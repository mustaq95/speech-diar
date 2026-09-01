"""ADEO Whisper -- execution code, reached through a per-model proxy on the same
ADEO inference host as `adeo_qwen3`. Audio LEAVES this host, so this engine is
`online`.

Returns the endpoint's native JSON untouched; only `adapter.py` reads it.

**This engine's response shape is UNVERIFIED, unlike every other engine in this
package.** Probed on 2026-09-01: the pod answered

    HTTP 504 {"detail": "Upstream service is unavailable", "status_code": 504}

on every attempt over ~2 minutes -- the transcription route AND its own
/v1/models -- so nothing about its body was ever observed. It is written against
the OpenAI transcription contract, which the SIBLING Qwen3 pod on this same host
is proven to speak (same route, same multipart fields, `{"text": ...}` back).
That is an informed shape, not a measured one. Probe it once the pod is up, and
correct this docstring with what it actually returns rather than leaving this
note to rot.

The 504 is a proxy-level failure, so it surfaces as `OpenAiAsrUpstreamError` with
the body attached and lands in `TranscriptResult.error`. `configured` checks
credentials only, deliberately: a pod that is merely DOWN must read as a failed
run with its status visible, not as "not configured", which would quietly hide
the engine from the comparison.

`model` is omitted when ADEO_WHISPER_MODEL is empty, matching the working curl
for this pod, which sends only `file` and `response_format`.

Configuration (`.env`, read via `packages/config/settings.py`):
  ADEO_WHISPER_URL / ADEO_WHISPER_API_KEY -- proxy endpoint and its bearer
  ADEO_WHISPER_MODEL                      -- served model id, or "" to omit
  ADEO_WHISPER_LANGUAGE                   -- "auto"/"" omits it (Whisper detects)
  ADEO_WHISPER_TIMEOUT_SEC                -- request timeout
  VERIFY_SSL / LITELLM_CA_BUNDLE          -- TLS for the ADEO host
"""

import os
from typing import Any

from packages.audio import ensure_canonical_wav
from packages.config.settings import get_settings

from .._openai_asr import (
    OpenAiAsrError,
    httpx_verify,
    post_transcription,
    resolve_language,
)

AdeoWhisperRawOutput = list[dict[str, Any]]

ENGINE = "adeo-whisper"


def _post(wav_bytes: bytes, filename: str) -> dict[str, Any]:
    settings = get_settings()
    if not settings.adeo_whisper_url or not settings.adeo_whisper_api_key:
        raise OpenAiAsrError(
            f"{ENGINE} is not configured -- set ADEO_WHISPER_URL and "
            "ADEO_WHISPER_API_KEY in .env"
        )
    return post_transcription(
        engine=ENGINE,
        url=settings.adeo_whisper_url,
        api_key=settings.adeo_whisper_api_key,
        # Empty omits the field, which is what this pod's working curl does.
        model=settings.adeo_whisper_model or None,
        # Omitted by default so Whisper auto-detects rather than being forced
        # onto one language across code-switched audio.
        language=resolve_language(settings.adeo_whisper_language),
        wav_bytes=wav_bytes,
        filename=filename,
        extra={"response_format": "json"},
        timeout=settings.adeo_whisper_timeout_sec,
        verify=httpx_verify(settings),
    )


def transcribe_bytes(wav_bytes: bytes, filename: str = "chunk.wav") -> dict[str, Any]:
    """Transcribe one already-short WAV -- the live chunk path."""
    return _post(wav_bytes, filename)


def text_of(part: dict[str, Any]) -> str:
    return ((part.get("response") or {}).get("text") or "").strip()


def run(audio_path: str) -> AdeoWhisperRawOutput:
    """Transcribe `audio_path` in ONE whole-file call. See `adeo_qwen3.runner.run`
    for why nothing is split here."""
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
