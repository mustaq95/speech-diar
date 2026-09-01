"""ADEO Qwen3-ASR -- execution code, reached through a per-model proxy on the
ADEO inference host. Audio LEAVES this host, so this engine is `online`.

Returns the endpoint's native JSON untouched; only `adapter.py` reads it.

Not the LiteLLM gateway. `GET {LITELLM_BASE_URL}/v1/models` lists hamsa-stt,
hamsa-tts, inception-stt, inception-tts and five text models, and no ASR model
beyond those -- this one lives at its own proxy path with its OWN bearer token,
which is why URL and key are a pair here instead of reusing LITELLM_*.

**`language` is never sent, and that is measured, not assumed.** Probed against
the live endpoint on 2026-09-01 with a 61 s code-switched Arabic/English clip:

  * omitted, `language=ar` and `language=en` returned BYTE-IDENTICAL text
    (sha256 92f045d76da8030b, 821 chars, all three).
  * `language=auto` -> HTTP 400, "Unsupported language: 'auto'", listing the 57
    codes it would accept. `auto` is not among them.
  * `language=ar,en` -> HTTP 400 the same way. There is no multi-language syntax.

So vLLM validates the field and the model then ignores it: sending a value can
only ever reject a request, never change one. The model code-switches natively
(Arabic in Arabic script, English in Latin, in one transcript) with the field
absent, which is exactly the behaviour this comparison wants. Do not "add
language support" here -- there is nothing to support, and adding it back
reintroduces the 400.

Native response shape: `{"text": str, "usage": {"type": "duration",
"seconds": int}}`. No segments, no per-word timings, no detected-language field.

Configuration (`.env`, read via `packages/config/settings.py`):
  ADEO_QWEN3_ASR_URL / ADEO_QWEN3_ASR_API_KEY -- proxy endpoint and its bearer
  ADEO_QWEN3_ASR_MODEL                        -- served model id
  ADEO_QWEN3_ASR_TIMEOUT_SEC                  -- request timeout
  VERIFY_SSL / LITELLM_CA_BUNDLE              -- TLS for the ADEO host
"""

import os
from typing import Any

from packages.audio import ensure_canonical_wav
from packages.config.settings import get_settings

from .._openai_asr import OpenAiAsrError, httpx_verify, post_transcription

#: A list with one entry, matching every other engine here, so `adapter.adapt`
#: and `raw_output` have one shape to read regardless of live or batch.
AdeoQwen3RawOutput = list[dict[str, Any]]

ENGINE = "adeo-qwen3-asr"


def _post(wav_bytes: bytes, filename: str) -> dict[str, Any]:
    settings = get_settings()
    if not settings.adeo_qwen3_asr_url or not settings.adeo_qwen3_asr_api_key:
        raise OpenAiAsrError(
            f"{ENGINE} is not configured -- set ADEO_QWEN3_ASR_URL and "
            "ADEO_QWEN3_ASR_API_KEY in .env"
        )
    return post_transcription(
        engine=ENGINE,
        url=settings.adeo_qwen3_asr_url,
        api_key=settings.adeo_qwen3_asr_api_key,
        model=settings.adeo_qwen3_asr_model,
        # No `language`: see the module docstring. Measured no-op, and `auto` 400s.
        wav_bytes=wav_bytes,
        filename=filename,
        # Greedy, so a re-run of the same audio is comparable to the last one.
        extra={"temperature": "0.0"},
        timeout=settings.adeo_qwen3_asr_timeout_sec,
        verify=httpx_verify(settings),
    )


def transcribe_bytes(wav_bytes: bytes, filename: str = "chunk.wav") -> dict[str, Any]:
    """Transcribe one already-short WAV -- the live chunk path."""
    return _post(wav_bytes, filename)


def text_of(part: dict[str, Any]) -> str:
    """The transcript text of one entry. The one place that knows where inside
    an entry this engine's own text lives."""
    return ((part.get("response") or {}).get("text") or "").strip()


def run(audio_path: str) -> AdeoQwen3RawOutput:
    """Transcribe `audio_path` in ONE whole-file call.

    No splitting: nothing was measured to suggest this endpoint truncates (the
    Inception safeguard is specific to that gateway), and cutting the audio would
    hand this engine a different input from the one its scores are compared
    against.
    """
    send_path, cleanup = ensure_canonical_wav(audio_path)
    try:
        with open(send_path, "rb") as fh:
            entry = _post(fh.read(), os.path.basename(send_path))
        entry["segment_index"] = 0
        return [entry]
    finally:
        if cleanup:
            try:
                os.unlink(cleanup)
            except OSError:
                pass
