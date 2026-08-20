"""Inception-STT — execution code, reached through the LiteLLM gateway.

A request/response transcription API (OpenAI-compatible: multipart `model` +
`file` + optional `language`, JSON back), not a streaming one. Audio LEAVES this
host, so this engine is `online`.

Returns the gateway's native per-segment JSON untouched; only `adapter.py`
reads that shape.

**This engine silently drops content, non-monotonically.** Measured against the
live gateway on 2026-08-19. It is deterministic (6/6 identical requests returned
byte-identical text), so this is a property of the engine, not flakiness -- but
length changes what comes back in a way no truncation model explains:

  * Four adjacent 5 s windows each transcribed fine (15, 10, 15, 15 words), yet
    the first two of them as ONE 10 s clip returned an empty string.
  * The same four as one 20 s clip returned 15 words -- exactly the content of
    the LAST 5 s, with the first 15 s absent.
  * Monotonicity is violated repeatedly: at one offset, 5 s returned 6 words,
    8 s returned 0, and 15 s returned 10. A clip cannot contain less speech than
    a clip it strictly contains.

Splitting one fixed 100 s span at different lengths, total words recovered:

    3 s -> 219 (1/34 blank)      10 s ->  28 (8/10 blank)
    5 s -> 201 (1/20 blank)      25 s ->  85 (1/4  blank)
    8 s ->  84 (8/13 blank)     100 s ->  51 (one call)

So short pieces are the only reliable regime, and `BATCH_SEGMENT_SECONDS`
defaults to 3 s for that reason -- the same length the live chunk path uses, so
both paths chunk identically and their numbers stay comparable. A test with one
short clip cannot catch a regression here: it needs a span long enough to be
split, compared against a single call on the same audio.

Boundaries are fixed, never nudged into a silence: see `split_wav_fixed`.

The gateway also returns `audio_duration`, `usage` and `word_timestamps` keys
whose values were observed to be null (and, on one earlier call, populated).
Nothing here depends on them; whatever arrives is persisted verbatim in
`TranscriptResult.raw_output` and read by nobody else.

Configuration (`.env`, read via `packages/config/settings.py`):
  LITELLM_BASE_URL / LITELLM_API_KEY  — gateway endpoint and credential
  STT_TRANSCRIPTION_PATH / STT_MODEL  — path on that gateway, model id
  STT_DEFAULT_LANGUAGE                — "" or "auto" = auto-detect (see below)
  STT_TIMEOUT_SECONDS                 — per-segment request timeout
  BATCH_SEGMENT_SECONDS               — the truncation guard
  BATCH_MAX_CONCURRENCY               — segments in flight at once
  VERIFY_SSL / LITELLM_CA_BUNDLE      — TLS for this gateway only
"""

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from typing import Any

import httpx

from packages.audio import ensure_canonical_wav, split_wav_fixed
from packages.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

#: One entry per segment. The gateway's own JSON is kept intact under
#: "response"; everything we measured or decided sits beside it, never merged
#: into it. Folding our fields into the engine's dict would make
#: `TranscriptResult.raw_output` a mixture of what the engine said and what we
#: added, with no way to tell them apart later -- and this platform's whole
#: contract is that native output stays native.
#:
#: A list even for short audio, so `adapter.py` has one shape to read.
InceptionRawOutput = list[dict[str, Any]]

#: Values of STT_DEFAULT_LANGUAGE that mean "let the model detect it", both of
#: which omit `language` from the request entirely.
#:
#: "auto" is in here because it is the natural thing to write in `.env` and is
#: NOT a language code the gateway would understand. Sending it verbatim would
#: make it a decoder prompt, which is how a forced language turns into
#: hallucinated output for the whole recording -- exactly the
#: cohere_transcribe_language failure. Mixed Arabic/English audio, which is what
#: this comparison exists to measure, must never force one.
AUTO_LANGUAGES = {"", "auto", "detect"}


#: One pooled client for the whole process, so the TCP+TLS handshake is paid once
#: instead of once per call.
#:
#: This is a LATENCY fix, not tidiness. `httpx.post(...)` — the module-level
#: function — builds and discards a Client per call, which means a fresh handshake
#: to the gateway every time. Measured against this gateway on the same 3 s chunk:
#: a new client per call ran a 107 ms median round trip, a pooled connection 35 ms.
#: The 73 ms difference was our own connection setup, reported on the scorecard as
#: the engine's chunk latency.
#:
#: httpx.Client is documented as thread-safe, which matters because the segmented
#: batch path calls it from a ThreadPoolExecutor and the live chunk endpoint calls
#: it from Starlette's threadpool.
_client: httpx.Client | None = None
_client_verify: object = object()  # sentinel: no client built yet
_client_lock = threading.Lock()


def _http_client(settings: Settings) -> httpx.Client:
    """The pooled client, rebuilt only if the TLS setting changed.

    Keyed on `verify` because that is the one field baked into a Client that a
    test or a config reload can legitimately change; everything else travels
    per-request.
    """
    global _client, _client_verify
    verify = settings.litellm_httpx_verify
    with _client_lock:
        if _client is None or _client_verify != verify:
            if _client is not None:
                _client.close()
            _client = httpx.Client(
                verify=verify,
                limits=httpx.Limits(
                    max_connections=settings.stt_max_connections,
                    max_keepalive_connections=settings.stt_max_connections,
                    # Must exceed the live chunk interval or the connection dies
                    # between chunks and every call re-handshakes.
                    keepalive_expiry=settings.stt_keepalive_expiry_sec,
                ),
            )
            _client_verify = verify
        return _client


class InceptionError(RuntimeError):
    """Base for failures talking to the Inception-STT gateway."""


class InceptionTimeoutError(InceptionError):
    """The gateway did not answer within STT_TIMEOUT_SECONDS."""


class InceptionConnectionError(InceptionError):
    """The gateway was unreachable (DNS / network / TLS)."""


class InceptionUpstreamError(InceptionError):
    """The gateway answered with a non-2xx status."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"inception-stt {status_code}: {body[:500]}")
        self.status_code = status_code
        self.body = body


class InceptionResponseError(InceptionError):
    """The gateway answered 2xx with a body that was not the expected JSON."""


def requested_language(settings: Settings) -> str | None:
    """The `language` value to send, or None to omit the field.

    Split out from `_post` so a test can assert the sentinel without a request.
    """
    language = (settings.stt_default_language or "").strip()
    return None if language.lower() in AUTO_LANGUAGES else language


def _post(settings: Settings, wav_bytes: bytes, filename: str) -> dict[str, Any]:
    """One transcription call, with its measured latency attached."""
    data: dict[str, str] = {"model": settings.stt_model}
    language = requested_language(settings)
    if language:
        data["language"] = language

    client = _http_client(settings)
    # Timed around the request only. The client is fetched above so pool setup on
    # the very first call is not charged to the engine.
    started = time.perf_counter()
    try:
        response = client.post(
            settings.stt_endpoint_url,
            headers={"Authorization": f"Bearer {settings.litellm_api_key}"},
            data=data,
            files={"file": (filename, wav_bytes, "audio/wav")},
            timeout=settings.stt_timeout_seconds,
        )
    except httpx.TimeoutException as exc:
        raise InceptionTimeoutError(
            f"inception-stt did not respond within {settings.stt_timeout_seconds}s"
        ) from exc
    except httpx.ConnectError as exc:
        raise InceptionConnectionError(
            f"inception-stt is not reachable at {settings.stt_endpoint_url}: {exc}"
        ) from exc
    except httpx.HTTPError as exc:
        raise InceptionConnectionError(f"inception-stt request failed: {exc}") from exc

    latency_ms = int((time.perf_counter() - started) * 1000)

    # Not raise_for_status(): httpx renders only "Client error '400 Bad Request'
    # for url ...", while the gateway puts the part that matters in the body.
    # That message ends up in TranscriptResult.error and on the panel.
    if response.status_code >= 400:
        raise InceptionUpstreamError(response.status_code, response.text)

    try:
        payload = response.json()
        text = payload["text"]
    except (ValueError, KeyError, TypeError) as exc:
        raise InceptionResponseError(
            f"unexpected response shape from inception-stt: {response.text[:500]}"
        ) from exc
    if not isinstance(text, str):
        raise InceptionResponseError(f"inception-stt returned a non-string text: {text!r}")

    return {"response": payload, "latency_ms": latency_ms}


def transcribe_bytes(wav_bytes: bytes, filename: str = "chunk.wav") -> dict[str, Any]:
    """Transcribe one already-short WAV, returning one InceptionRawOutput entry.

    Used by the live chunk path, whose chunks are short by construction and need
    no splitting.
    """
    return _post(get_settings(), wav_bytes, filename)


def text_of(part: dict[str, Any]) -> str:
    """The transcript text of one segment entry. The one place that knows where
    inside an entry the engine's own text lives."""
    return ((part.get("response") or {}).get("text") or "").strip()


def run(audio_path: str) -> InceptionRawOutput:
    """Transcribe `audio_path`, splitting it under the truncation ceiling.

    Segments are transcribed concurrently (`BATCH_MAX_CONCURRENCY`) but the
    returned list stays in AUDIO order, not completion order: `adapter.py`
    joins it into one transcript, and out-of-order pieces would scramble it.
    """
    settings = get_settings()
    if not settings.stt_endpoint_url or not settings.litellm_api_key:
        raise InceptionError(
            "inception-stt is not configured — set LITELLM_BASE_URL and LITELLM_API_KEY in .env"
        )

    send_path, cleanup = ensure_canonical_wav(audio_path)
    try:
        pieces = split_wav_fixed(send_path, settings.batch_segment_seconds)
        logger.info(
            "inception-stt: %d segment(s) of <=%.0fs for %s",
            len(pieces), settings.batch_segment_seconds, os.path.basename(audio_path),
        )

        def transcribe(index_and_piece: tuple[int, tuple[bytes, float, float]]) -> dict[str, Any]:
            index, (wav_bytes, start, end) = index_and_piece
            entry = _post(settings, wav_bytes, f"segment{index}.wav")
            # Offsets travel with each piece so a caller can place the text in
            # the recording without re-deriving the split.
            entry["segment_index"] = index
            entry["segment_start"] = start
            entry["segment_end"] = end
            return entry

        if len(pieces) == 1:
            return [transcribe((0, pieces[0]))]

        workers = max(1, min(settings.batch_max_concurrency, len(pieces)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # executor.map preserves input order regardless of completion order.
            return list(pool.map(transcribe, enumerate(pieces)))
    finally:
        if cleanup:
            with suppress(OSError):
                os.unlink(cleanup)
