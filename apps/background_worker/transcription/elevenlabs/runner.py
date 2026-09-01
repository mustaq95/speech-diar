"""ElevenLabs Scribe -- execution code. Audio LEAVES this host, so `online`.

One whole-file multipart POST, JSON back. NOT OpenAI-compatible despite the
resemblance, which is why this does not use `_openai_asr`:

  * auth is the `xi-api-key` HEADER, not `Authorization: Bearer`
  * the model field is `model_id`, not `model`
  * the language field is `language_code` (ISO-639-3, e.g. "ara"), not `language`

Returns the endpoint's native JSON untouched; only `adapter.py` reads it.

Probed against the live API on 2026-09-01 with a 61 s code-switched
Arabic/English clip:

  * `model_id` is validated server-side. A bogus id returns HTTP 400 naming the
    valid ones ('scribe_v1', 'scribe_v1_experimental', 'scribe_v2'), so a typo
    fails loudly instead of silently falling back to an older model.
  * With `language_code` OMITTED it auto-detected `language_code: "ara"` at
    `language_probability: 0.968` and still kept the English passages in Latin
    script -- genuine code-switching, WER 0.564, the best of every engine probed.
  * The response carries per-word entries typed `word`, `spacing` and
    `audio_event` (e.g. "[phone chimes]"), each with start/end/logprob.

Those `audio_event` entries are inside `text` too, and they are deliberately NOT
stripped: they are what the engine reported, and removing them here would be this
code editing a measurement. They count against the engine's own WER, which is
honest -- the reference contains no such marker.

The per-word timings are kept in `raw_output` and are NOT written to
`TranscriptResult.words`: that column is the CTC aligner's output for every
engine, and mixing one vendor's timings into it would make the column mean two
different things depending on the row.

Configuration (`.env`, read via `packages/config/settings.py`):
  ELEVENLABS_API_KEY        -- the xi-api-key credential
  ELEVENLABS_STT_URL        -- endpoint
  ELEVENLABS_STT_MODEL      -- scribe_v2 (validated server-side)
  ELEVENLABS_STT_LANGUAGE   -- "auto"/"" omits language_code (auto-detect)
  ELEVENLABS_STT_TIMEOUT_SEC
"""

import os
import threading
import time
from typing import Any

import httpx

from packages.audio import ensure_canonical_wav
from packages.config.settings import Settings, get_settings

from .._openai_asr import AUTO_LANGUAGES

ElevenLabsRawOutput = list[dict[str, Any]]

ENGINE = "elevenlabs-scribe-v2"

_client: httpx.Client | None = None
_client_lock = threading.Lock()


def _http_client() -> httpx.Client:
    """One pooled client, for the same latency reason as every other engine here:
    a per-call client re-handshakes TLS and that time is reported as the
    engine's."""
    global _client
    with _client_lock:
        if _client is None:
            _client = httpx.Client(
                limits=httpx.Limits(max_connections=16, max_keepalive_connections=16,
                                    keepalive_expiry=30),
            )
        return _client


class ElevenLabsError(RuntimeError):
    """Base for failures talking to the ElevenLabs Scribe API."""


class ElevenLabsUpstreamError(ElevenLabsError):
    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"{ENGINE} {status_code}: {body[:500]}")
        self.status_code = status_code
        self.body = body


def requested_language(settings: Settings) -> str | None:
    """The `language_code` to send, or None to omit it (auto-detect)."""
    language = (settings.elevenlabs_stt_language or "").strip()
    return None if language.lower() in AUTO_LANGUAGES else language


def _post(wav_bytes: bytes, filename: str) -> dict[str, Any]:
    settings = get_settings()
    if not settings.elevenlabs_api_key:
        raise ElevenLabsError(
            f"{ENGINE} is not configured -- set ELEVENLABS_API_KEY in .env"
        )
    data: dict[str, str] = {"model_id": settings.elevenlabs_stt_model}
    language = requested_language(settings)
    if language:
        data["language_code"] = language

    client = _http_client()
    started = time.perf_counter()
    try:
        response = client.post(
            settings.elevenlabs_stt_url,
            headers={"xi-api-key": settings.elevenlabs_api_key},
            data=data,
            files={"file": (filename, wav_bytes, "audio/wav")},
            timeout=settings.elevenlabs_stt_timeout_sec,
        )
    except httpx.TimeoutException as exc:
        raise ElevenLabsError(
            f"{ENGINE} did not respond within {settings.elevenlabs_stt_timeout_sec}s"
        ) from exc
    except httpx.HTTPError as exc:
        raise ElevenLabsError(f"{ENGINE} request failed: {exc}") from exc

    latency_ms = int((time.perf_counter() - started) * 1000)

    # The body carries the useful part: a rejected model_id lists the valid ones.
    if response.status_code >= 400:
        raise ElevenLabsUpstreamError(response.status_code, response.text)

    try:
        payload = response.json()
        text = payload["text"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ElevenLabsError(
            f"unexpected response shape from {ENGINE}: {response.text[:500]}"
        ) from exc
    if not isinstance(text, str):
        raise ElevenLabsError(f"{ENGINE} returned a non-string text: {text!r}")

    return {"response": payload, "latency_ms": latency_ms}


def transcribe_bytes(wav_bytes: bytes, filename: str = "chunk.wav") -> dict[str, Any]:
    """Transcribe one already-short WAV -- the live chunk path."""
    return _post(wav_bytes, filename)


def text_of(part: dict[str, Any]) -> str:
    return ((part.get("response") or {}).get("text") or "").strip()


def run(audio_path: str) -> ElevenLabsRawOutput:
    """Transcribe `audio_path` in ONE whole-file call."""
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
