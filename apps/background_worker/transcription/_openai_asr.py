"""Shared transport for the ASR engines that speak OpenAI's transcription API.

Three engines here post the same multipart shape to the same route
(`POST .../v1/audio/transcriptions` with `file` + optional `model`/`language`,
JSON back): `adeo-qwen3-asr`, `adeo-whisper` and `moss-transcribe`. This module
is their shared TRANSPORT, in the spirit of `models/_nim_shared.py` -- it makes
the call and hands back the gateway's own JSON untouched. It is not an adapter
and never reads the payload beyond confirming `text` is a string, so each
engine's `adapter.py` remains the only code that understands its shape.

Deliberately NOT used by `inception/`: that engine carries a truncation
safeguard, its own split/concurrency policy and its own measured pooling
constants, and folding it in here would put one engine's workaround in every
engine's path.

`language` is passed by the caller or omitted entirely. Two of the three callers
never pass it at all, because it was MEASURED to be validated-then-ignored by
their vLLM builds -- see each runner's docstring.
"""

import threading
import time
from typing import Any

import httpx

from packages.config.settings import Settings

#: Values that mean "let the model detect it", which omit `language` from the
#: request. Same vocabulary as `inception.runner.AUTO_LANGUAGES`, and the reason
#: is the same: "auto" is the natural thing to write in `.env` but is NOT a code
#: any of these endpoints accepts -- both vLLM pods here reject it with a 400
#: against a 57-code list. Sending it verbatim would fail every request.
AUTO_LANGUAGES = {"", "auto", "detect"}


class OpenAiAsrError(RuntimeError):
    """Base for failures talking to an OpenAI-compatible transcription endpoint."""


class OpenAiAsrTimeoutError(OpenAiAsrError):
    """The endpoint did not answer within its configured timeout."""


class OpenAiAsrConnectionError(OpenAiAsrError):
    """The endpoint was unreachable (DNS / network / TLS)."""


class OpenAiAsrUpstreamError(OpenAiAsrError):
    """The endpoint answered with a non-2xx status."""

    def __init__(self, engine: str, status_code: int, body: str) -> None:
        super().__init__(f"{engine} {status_code}: {body[:500]}")
        self.status_code = status_code
        self.body = body


class OpenAiAsrResponseError(OpenAiAsrError):
    """The endpoint answered 2xx with a body that was not the expected JSON."""


#: One pooled client per process, rebuilt only when the TLS setting changes.
#: A per-call client re-handshakes every request; measured against the sibling
#: Inception gateway that cost 73 ms of a 107 ms round trip, reported on the
#: scorecard as the ENGINE's latency. httpx.Client is documented thread-safe,
#: which the live chunk path (Starlette threadpool) depends on.
_client: httpx.Client | None = None
_client_verify: object = object()
_client_lock = threading.Lock()


def _http_client(verify: bool | str) -> httpx.Client:
    global _client, _client_verify
    with _client_lock:
        if _client is None or _client_verify != verify:
            if _client is not None:
                _client.close()
            _client = httpx.Client(
                verify=verify,
                limits=httpx.Limits(max_connections=16, max_keepalive_connections=16,
                                    keepalive_expiry=30),
            )
            _client_verify = verify
        return _client


def resolve_language(configured: str | None) -> str | None:
    """The `language` value to send, or None to omit the field."""
    language = (configured or "").strip()
    return None if language.lower() in AUTO_LANGUAGES else language


def post_transcription(
    *,
    engine: str,
    url: str,
    wav_bytes: bytes,
    filename: str,
    timeout: float,
    api_key: str | None = None,
    model: str | None = None,
    language: str | None = None,
    extra: dict[str, str] | None = None,
    verify: bool | str = True,
) -> dict[str, Any]:
    """One transcription call, with its measured latency attached.

    Returns one entry in the shape every engine here persists:
    `{"response": <the endpoint's own JSON, untouched>, "latency_ms": int}`.
    The engine's JSON is kept whole under "response" and never merged with what
    we measured, so `TranscriptResult.raw_output` stays separable into what the
    engine said and what we added.
    """
    data: dict[str, str] = dict(extra or {})
    if model:
        data["model"] = model
    if language:
        data["language"] = language

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    client = _http_client(verify)
    # Timed around the request only: the client is fetched first so pool setup on
    # the very first call is never charged to the engine.
    started = time.perf_counter()
    try:
        response = client.post(
            url, headers=headers, data=data,
            files={"file": (filename, wav_bytes, "audio/wav")}, timeout=timeout,
        )
    except httpx.TimeoutException as exc:
        raise OpenAiAsrTimeoutError(f"{engine} did not respond within {timeout}s") from exc
    except httpx.ConnectError as exc:
        raise OpenAiAsrConnectionError(f"{engine} is not reachable at {url}: {exc}") from exc
    except httpx.HTTPError as exc:
        raise OpenAiAsrConnectionError(f"{engine} request failed: {exc}") from exc

    latency_ms = int((time.perf_counter() - started) * 1000)

    # Not raise_for_status(): httpx renders only "Client error '400 Bad Request'",
    # while these endpoints put the part that matters in the body -- the 400 that
    # rejects `language` enumerates every code it WOULD accept. That message ends
    # up in TranscriptResult.error and on the panel.
    if response.status_code >= 400:
        raise OpenAiAsrUpstreamError(engine, response.status_code, response.text)

    try:
        payload = response.json()
        text = payload["text"]
    except (ValueError, KeyError, TypeError) as exc:
        raise OpenAiAsrResponseError(
            f"unexpected response shape from {engine}: {response.text[:500]}"
        ) from exc
    if not isinstance(text, str):
        raise OpenAiAsrResponseError(f"{engine} returned a non-string text: {text!r}")

    return {"response": payload, "latency_ms": latency_ms}


def httpx_verify(settings: Settings) -> bool | str:
    """TLS setting for the ADEO inference host, reusing the gateway's own switch.

    Same host family and same internal CA as the LiteLLM gateway, so it follows
    VERIFY_SSL / LITELLM_CA_BUNDLE rather than introducing a third TLS flag
    nobody would remember to set.
    """
    return settings.litellm_httpx_verify
