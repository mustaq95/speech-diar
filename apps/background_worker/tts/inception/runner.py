"""Inception-TTS — execution code, reached through the LiteLLM gateway.

A single-shot synthesis API (OpenAI-compatible `/v1/audio/speech`: JSON
`model` + `input` + `voice` + `response_format`, audio body back), not a
streaming one in any meaningful sense: measured ~6.7s wall clock for a
10-word Arabic sentence with the whole body buffered before any bytes are
visible. `client.stream()` is still used, for the same code path as the
Hamsa runner and an honestly-measured `first_audio_ms` — which will land at
≈ `synth_ms` for this engine, and that is the correct, expected figure, not a
bug.

Returns the gateway's native bytes untouched; only `adapter.py` reads their
container.

**The gateway's `Content-Type` header lies.** It is always `audio/mpeg`
regardless of what container is actually returned, confirmed wrong even when
`response_format: "wav"` was requested and ffprobe showed real RIFF/WAVE,
pcm_s16le, mono, 24000Hz. Never branch on that header; the adapter sniffs
magic bytes instead.

Configuration (`.env`, read via `packages/config/settings.py`):
  LITELLM_BASE_URL / LITELLM_API_KEY     — gateway endpoint and credential
                                            (shared with inception-stt)
  TTS_SPEECH_PATH / INCEPTION_TTS_MODEL  — path on that gateway, model id
  INCEPTION_TTS_RESPONSE_FORMAT          — requested container ("wav")
  INCEPTION_TTS_TIMEOUT_SEC              — request timeout
  VERIFY_SSL / LITELLM_CA_BUNDLE         — TLS for this gateway only
"""

import threading
import time
from typing import Any

import httpx

from packages.config.settings import Settings, get_settings

#: Pooled client, same reasoning as `transcription/inception/runner.py`'s
#: `_http_client`: a fresh client per call re-pays the TLS handshake.
_client: httpx.Client | None = None
_client_verify: object = object()  # sentinel: no client built yet
_client_lock = threading.Lock()


def _http_client(settings: Settings) -> httpx.Client:
    global _client, _client_verify
    verify = settings.litellm_httpx_verify
    with _client_lock:
        if _client is None or _client_verify != verify:
            if _client is not None:
                _client.close()
            _client = httpx.Client(verify=verify)
            _client_verify = verify
        return _client


class InceptionTtsError(RuntimeError):
    """Base for failures talking to the Inception-TTS gateway."""


class InceptionTtsTimeoutError(InceptionTtsError):
    """The gateway did not answer within INCEPTION_TTS_TIMEOUT_SEC."""


class InceptionTtsConnectionError(InceptionTtsError):
    """The gateway was unreachable (DNS / network / TLS)."""


class InceptionTtsUpstreamError(InceptionTtsError):
    """The gateway answered with a non-2xx status."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"inception-tts {status_code}: {body[:500]}")
        self.status_code = status_code
        self.body = body


class InceptionTtsResponseError(InceptionTtsError):
    """The gateway answered 2xx with a body that was not the expected audio."""


def run(text: str, voice: str) -> dict[str, Any]:
    """Synthesize `text` with `voice`, returning the gateway's native bytes
    and headers untouched, alongside what was measured."""
    settings = get_settings()
    if not settings.tts_speech_url or not settings.litellm_api_key:
        raise InceptionTtsError(
            "inception-tts is not configured — set LITELLM_BASE_URL and LITELLM_API_KEY in .env"
        )

    client = _http_client(settings)
    payload = {
        "model": settings.inception_tts_model,
        "input": text,
        "voice": voice,
        "response_format": settings.inception_tts_response_format,
        # No "stream" flag. It is not in this gateway's documented parameter
        # set, it contradicts this engine's delivery="single" label, and it is
        # provably inert here (measured with it set: first byte still lands at
        # ~= total, the whole body buffered). If the gateway ever DID honour it
        # the body would be SSE, the adapter's magic-byte sniff would return
        # None, and every synthesis would fail on a 200.
    }

    chunks: list[bytes] = []
    first_audio_ms: int | None = None
    started = time.perf_counter()
    try:
        with client.stream(
            "POST",
            settings.tts_speech_url,
            headers={"Authorization": f"Bearer {settings.litellm_api_key}"},
            json=payload,
            timeout=settings.inception_tts_timeout_sec,
        ) as response:
            if response.status_code >= 400:
                response.read()
                raise InceptionTtsUpstreamError(response.status_code, response.text)
            response_headers = dict(response.headers)
            status_code = response.status_code
            for chunk in response.iter_bytes():
                if not chunk:
                    continue
                if first_audio_ms is None:
                    first_audio_ms = int((time.perf_counter() - started) * 1000)
                chunks.append(chunk)
    except httpx.TimeoutException as exc:
        raise InceptionTtsTimeoutError(
            f"inception-tts did not respond within {settings.inception_tts_timeout_sec}s"
        ) from exc
    except httpx.ConnectError as exc:
        raise InceptionTtsConnectionError(
            f"inception-tts is not reachable at {settings.tts_speech_url}: {exc}"
        ) from exc
    except httpx.HTTPError as exc:
        raise InceptionTtsConnectionError(f"inception-tts request failed: {exc}") from exc

    synth_ms = int((time.perf_counter() - started) * 1000)

    return {
        "audio": b"".join(chunks),
        "status_code": status_code,
        "headers": response_headers,
        "synth_ms": synth_ms,
        "first_audio_ms": first_audio_ms,
        "voice": voice,
    }
