"""TryHamsa voice cloning — execution code for the two pod calls.

Both are plain JSON POSTs against the SAME pod `hamsa-tts` synthesizes through,
with the same credential pair, and both return the pod's native output
untouched; only `adapter.py` is allowed to understand it.

Endpoint paths were taken from the pod's own `/openapi.json`, probed
2026-09-09, NOT from the vendor's written guide -- the two disagree:

    guide says              pod actually serves
    POST /v1/voice-clone    POST /tts/voice_clone
    POST /v1/voices         POST /tts/load_voice_cloning
    POST /v1/speech         POST /tts/stream   (what hamsa-tts already calls)

The guide also documents a `GET`-able voice list. There is none: every unknown
path on this pod answers 405 from a catch-all `OPTIONS /{path}` handler, which
is what makes a missing route look like an existing one. Nothing here discovers
what voices the pod holds, so nothing in this repo may claim to.

Measured behaviour that the caller has to cope with:
  * Extraction on a pod without the cloning model loaded returns a bare
    `text/plain` 500 "Internal Server Error" in ~80ms, for ANY `audio_url`
    including a syntactically invalid one -- it never attempts the download.
    That is NOT the "error describing the cause" the guide promises, so the
    upstream body is carried through verbatim rather than interpreted.
  * `audio_url` is fetched BY THE POD, from wherever the pod runs. A URL that
    resolves on this host proves nothing about whether the pod can reach it.

Configuration (`.env`, read via `packages/config/settings.py`):
  HAMSA_VOICE_CLONE_URL / HAMSA_LOAD_VOICE_URL   the two endpoints
  HAMSA_TTS_KEY / HAMSA_TTS_BEARER_TOKEN         the pod's credential pair,
                                                 shared with hamsa-tts (both
                                                 required; omitting the bearer
                                                 is a 401)
  HAMSA_TTS_SSL_VERIFY                           false for a cert this host
                                                 does not trust
  HAMSA_VOICE_CLONE_TIMEOUT_SEC                  extraction runs a model over
                                                 the whole clip, so it gets its
                                                 own budget
"""

import threading
import time
from typing import Any

import httpx

from packages.config.settings import Settings, get_settings

#: One pooled client for the whole process, rebuilt only if the TLS setting
#: changes. Same reasoning as `tts/hamsa/runner.py`'s own `_http_client`: a
#: fresh client per call re-pays the TLS handshake.
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


class VoiceCloneError(RuntimeError):
    """Base for failures talking to the cloning endpoints."""


class VoiceCloneConnectionError(VoiceCloneError):
    """The pod was unreachable (DNS / network / TLS)."""


class VoiceCloneUpstreamError(VoiceCloneError):
    """The pod answered with a non-2xx status.

    The body is carried verbatim and truncated, never summarised: the failure
    that actually happens here is an opaque `text/plain` 500, and rewording it
    into something friendlier would hide the one clue there is.
    """

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"voice-clone {status_code}: {body[:500] or '(empty body)'}")
        self.status_code = status_code
        self.body = body


def _headers(settings: Settings) -> dict[str, str]:
    return {
        "X-API-Key": settings.hamsa_tts_key or "",
        "Authorization": f"Bearer {settings.hamsa_tts_bearer_token}",
        "content-type": "application/json",
    }


def _post(url: str, payload: dict[str, Any], settings: Settings) -> tuple[Any, dict[str, Any], int]:
    """POST `payload`, returning (parsed body or raw text, headers, status).

    The body is returned as parsed JSON when it is JSON and as the raw string
    when it is not. `/tts/load_voice_cloning` answers a bare `null` on success,
    which is valid JSON and must survive as `None` rather than being coerced
    into an empty dict.
    """
    client = _http_client(settings)
    try:
        response = client.post(
            url,
            headers=_headers(settings),
            json=payload,
            timeout=settings.hamsa_voice_clone_timeout_sec,
        )
    except httpx.ConnectError as exc:
        raise VoiceCloneConnectionError(f"voice-clone is not reachable at {url}: {exc}") from exc
    except httpx.HTTPError as exc:
        raise VoiceCloneConnectionError(f"voice-clone request failed: {exc}") from exc

    if response.status_code >= 400:
        raise VoiceCloneUpstreamError(response.status_code, response.text)

    try:
        body: Any = response.json()
    except ValueError:
        body = response.text
    return body, dict(response.headers), response.status_code


def extract(audio_url: str, prompt_text: str) -> dict[str, Any]:
    """Turn a reference clip into voice tokens. `POST /tts/voice_clone`.

    Returns the pod's native body untouched alongside what was measured. The
    pod downloads `audio_url` itself; nothing is uploaded in this request.
    """
    settings = get_settings()
    if not settings.hamsa_voice_clone_url:
        raise VoiceCloneError(
            "voice cloning is not configured — set HAMSA_VOICE_CLONE_URL in .env"
        )

    started = time.perf_counter()
    body, headers, status_code = _post(
        settings.hamsa_voice_clone_url,
        {"audio_url": audio_url, "prompt_text": prompt_text},
        settings,
    )
    return {
        "body": body,
        "headers": headers,
        "status_code": status_code,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


def register(
    speaker_id: str,
    global_token_ids: Any,
    semantic_token_ids: Any,
    dialect: str,
    prompt_text: str,
) -> dict[str, Any]:
    """File already-extracted tokens under a speaker name.
    `POST /tts/load_voice_cloning`.

    The token arrays are passed through EXACTLY as extraction returned them.
    They are the pod's own opaque representation and this repo never inspects,
    reorders or re-types them -- the same rule that keeps every other adapter
    off another engine's native shape.
    """
    settings = get_settings()
    if not settings.hamsa_load_voice_url:
        raise VoiceCloneError(
            "voice cloning is not configured — set HAMSA_LOAD_VOICE_URL in .env"
        )

    started = time.perf_counter()
    body, headers, status_code = _post(
        settings.hamsa_load_voice_url,
        {
            "speaker_id": speaker_id,
            "global_token_ids": global_token_ids,
            "semantic_token_ids": semantic_token_ids,
            "dialect": dialect,
            "prompt_text": prompt_text,
        },
        settings,
    )
    return {
        "body": body,
        "headers": headers,
        "status_code": status_code,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }
