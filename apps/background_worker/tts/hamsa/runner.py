"""TryHamsa TTS — execution code for the streaming synthesis engine.

A request/response-shaped POST whose body is delivered as a stream: the
server writes PCM16 audio in ~3200-byte chunks as it synthesizes, so the
first bytes exist well before the whole utterance does. Returns the response
untouched (raw bytes, status, headers); only `adapter.py` is allowed to
understand that shape.

Measured against the live endpoint on 2026-08-20 (17-word Arabic sentence):
first chunk at ~307ms into a ~1965ms total synthesis, ~16% of the total — a
real, meaningfully-earlier-than-total `first_audio_ms`, not noise.

Two things about the response are misleading and must not be trusted:
  * `Content-Type: text/event-stream; charset=utf-8` is claimed but the body
    is raw PCM16 binary, not SSE. Consumed with `.iter_bytes()`, never
    line/event parsing.
  * No field anywhere states the sample rate. `HAMSA_TTS_SAMPLE_RATE`
    defaults to 16000 as a documented ASSUMPTION (see settings.py), not a
    measured fact — it is what the adapter uses to build a WAV header.

Configuration (`.env`, read via `packages/config/settings.py`):
  HAMSA_TTS_API_URL / HAMSA_TTS_KEY / HAMSA_TTS_BEARER_TOKEN — endpoint + auth
                                        (both headers are required; omitting
                                        the bearer token gives a 401)
  HAMSA_TTS_DIALECT / HAMSA_TTS_LANGUAGE_ID — request fields, not per-call
  HAMSA_TTS_SSL_VERIFY               — false for a cert this host doesn't trust
  HAMSA_TTS_TIMEOUT_SEC              — request timeout
"""

import threading
import time
from typing import Any

import httpx

from packages.config.settings import Settings, get_settings

#: One pooled client for the whole process, rebuilt only if the TLS setting
#: changes. Same reasoning as `transcription/inception/runner.py`'s
#: `_http_client`: a fresh client per call re-pays the TLS handshake, measured
#: there at 73ms of a 107ms round trip against a comparable gateway.
_client: httpx.Client | None = None
_client_verify: object = object()  # sentinel: no client built yet
_client_lock = threading.Lock()


def _http_client(settings: Settings) -> httpx.Client:
    global _client, _client_verify
    verify = settings.hamsa_tts_ssl_verify
    with _client_lock:
        if _client is None or _client_verify != verify:
            if _client is not None:
                _client.close()
            _client = httpx.Client(verify=verify)
            _client_verify = verify
        return _client


class HamsaTtsError(RuntimeError):
    """Base for failures talking to the TryHamsa TTS endpoint."""


class HamsaTtsConnectionError(HamsaTtsError):
    """The endpoint was unreachable (DNS / network / TLS)."""


class HamsaTtsUpstreamError(HamsaTtsError):
    """The endpoint answered with a non-2xx status."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"hamsa-tts {status_code}: {body[:500]}")
        self.status_code = status_code
        self.body = body


def run(text: str, voice: str) -> dict[str, Any]:
    """Synthesize `text` with speaker `voice`, returning Hamsa's native bytes
    and headers untouched, alongside what was measured.

    `dialect` and `language_id` come from settings, not from the caller: they
    are the TTS deployment's own config, not something to vary per request.
    """
    settings = get_settings()
    if not settings.hamsa_tts_api_url or not settings.hamsa_tts_key:
        raise HamsaTtsError(
            "hamsa-tts is not configured — set HAMSA_TTS_API_URL and HAMSA_TTS_KEY in .env"
        )

    client = _http_client(settings)
    headers = {
        "X-API-Key": settings.hamsa_tts_key,
        "Authorization": f"Bearer {settings.hamsa_tts_bearer_token}",
    }
    payload = {
        "text": text,
        "speaker": voice,
        "dialect": settings.hamsa_tts_dialect,
        "language_id": settings.hamsa_tts_language_id,
        "mulaw": False,
        "stream": True,
    }

    chunks: list[bytes] = []
    first_audio_ms: int | None = None
    started = time.perf_counter()
    try:
        with client.stream(
            "POST",
            settings.hamsa_tts_api_url,
            headers=headers,
            json=payload,
            timeout=settings.hamsa_tts_timeout_sec,
        ) as response:
            if response.status_code >= 400:
                response.read()
                raise HamsaTtsUpstreamError(response.status_code, response.text)
            response_headers = dict(response.headers)
            status_code = response.status_code
            for chunk in response.iter_bytes():
                if not chunk:
                    continue
                if first_audio_ms is None:
                    first_audio_ms = int((time.perf_counter() - started) * 1000)
                chunks.append(chunk)
    except httpx.ConnectError as exc:
        raise HamsaTtsConnectionError(
            f"hamsa-tts is not reachable at {settings.hamsa_tts_api_url}: {exc}"
        ) from exc
    except httpx.HTTPError as exc:
        raise HamsaTtsConnectionError(f"hamsa-tts request failed: {exc}") from exc

    synth_ms = int((time.perf_counter() - started) * 1000)

    return {
        "audio": b"".join(chunks),
        "status_code": status_code,
        "headers": response_headers,
        "synth_ms": synth_ms,
        "first_audio_ms": first_audio_ms,
        "voice": voice,
    }
