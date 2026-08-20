"""Server-side state for one live read-aloud comparison.

The read-aloud flow captures a transcript from each engine WHILE someone is
speaking, then scores what was captured. That transcript has to live somewhere
between the first chunk and the moment the recording is finalized.

**It lives here, on the server, not in the browser.** The obvious shortcut is to
let the page accumulate its own text and post it at the end (the reference client
did exactly that, as a `live_text` form field). This does not, for one reason: the
numbers are the product. Latency is measured where the call is made, and a
transcript that arrives as a request body is a claim rather than a measurement.
Keeping the accumulation next to the calls means what gets scored is what the
engine actually returned, in the order it returned it.

Redis rather than a process dict, because the API can run as more than one
process and a session must not depend on which one a request lands on. Keys carry
a TTL (`LIVE_SESSION_TTL_SEC`) so an abandoned recording expires on its own —
nothing here needs cleaning up by hand.

Layout, one session:

    ts:{sid}:meta            hash  — chunk interval, engine ids, reference text
    ts:{sid}:{asr_id}:parts  list  — one JSON entry per chunk, RPUSHed

Per-engine LISTS, not one JSON blob: two engines append concurrently for the
whole recording, and a read-modify-write of a shared document would drop chunks
under exactly the load this is built for. RPUSH is atomic, so each engine's parts
accumulate independently and are ordered on read by the index the caller
measured, never by arrival.
"""

import json
import logging
import time
import uuid
from typing import Any

from redis import Redis

from packages.config.settings import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()
_redis = Redis.from_url(_settings.redis_url)

_PREFIX = "ts"


def _meta_key(session_id: str) -> str:
    return f"{_PREFIX}:{session_id}:meta"


def _parts_key(session_id: str, asr_id: str) -> str:
    return f"{_PREFIX}:{session_id}:{asr_id}:parts"


def create(asr_ids: list[str], chunk_interval_sec: float, reference_text: str = "") -> str:
    """Open a session and return its id.

    `started_at` is stamped here so the session's own elapsed time is measured
    from the server, not reconstructed from client clocks.
    """
    session_id = uuid.uuid4().hex
    ttl = _settings.live_session_ttl_sec
    mapping = {
        "asr_ids": json.dumps(asr_ids),
        "chunk_interval_sec": str(chunk_interval_sec),
        "reference_text": reference_text,
        "started_at": str(time.time()),
    }
    pipe = _redis.pipeline()
    pipe.hset(_meta_key(session_id), mapping=mapping)
    pipe.expire(_meta_key(session_id), ttl)
    pipe.execute()
    logger.info("live transcript session %s open for %s", session_id, ",".join(asr_ids))
    return session_id


def exists(session_id: str) -> bool:
    return bool(_redis.exists(_meta_key(session_id)))


def meta(session_id: str) -> dict[str, Any] | None:
    """The session's own record, or None once it has expired or been closed."""
    raw = _redis.hgetall(_meta_key(session_id))
    if not raw:
        return None
    decoded = {key.decode(): value.decode() for key, value in raw.items()}
    return {
        "asr_ids": json.loads(decoded.get("asr_ids", "[]")),
        "chunk_interval_sec": float(decoded.get("chunk_interval_sec", 0) or 0),
        "reference_text": decoded.get("reference_text", ""),
        "started_at": float(decoded.get("started_at", 0) or 0),
    }


def append_part(
    session_id: str,
    asr_id: str,
    *,
    index: int,
    text: str,
    latency_ms: int,
    raw: Any | None = None,
) -> None:
    """Record one engine's result for one chunk, with the latency measured for it.

    Blank text is recorded too, not skipped: a chunk the engine returned nothing
    for is a real outcome (this gateway does it — see the inception runner's
    measurements) and the chunk count has to stay honest about how many calls
    were actually made.

    `raw` is the engine's NATIVE response for this chunk, carried through so
    finalize can persist it to `TranscriptResult.raw_output`. Without it a live
    row would keep only the adapted string, and this platform's rule is that
    native output survives precisely because the adapters drop real data an
    evaluation tool should not lose — Inception's per-chunk `audio_duration`,
    `usage` and `word_timestamps`, Hamsa's own language detection and timings.
    The stored-audio pipeline already keeps it; the live path used to not.
    """
    entry = json.dumps({"i": index, "text": text, "ms": latency_ms, "raw": raw})
    key = _parts_key(session_id, asr_id)
    pipe = _redis.pipeline()
    pipe.rpush(key, entry)
    pipe.expire(key, _settings.live_session_ttl_sec)
    pipe.execute()


def parts(session_id: str, asr_id: str) -> list[dict[str, Any]]:
    """One engine's chunks, ordered by the index the caller measured them at.

    Sorted on read rather than trusted from the list order: chunks are posted
    concurrently and a slow call can land after a later one.
    """
    entries = [json.loads(raw) for raw in _redis.lrange(_parts_key(session_id, asr_id), 0, -1)]
    return sorted(entries, key=lambda part: part["i"])


def transcript(session_id: str, asr_id: str) -> dict[str, Any]:
    """One engine's accumulated transcript plus the timings behind it.

    `text` is the chunk texts joined with a single space, in audio order, with
    nothing else done to them — no de-duplication across a boundary and no
    punctuation repair. A word split by a chunk boundary stays split, because
    that cost belongs to the transport being measured.
    """
    collected = parts(session_id, asr_id)
    latencies = [part["ms"] for part in collected]
    texts = [part["text"].strip() for part in collected if part["text"].strip()]
    # Native payloads in measured-index order, every chunk represented — including
    # the ones that came back empty, since "the engine returned nothing here" is
    # itself a result worth being able to re-read.
    raw = [part.get("raw") for part in collected]
    return {
        "text": " ".join(texts).strip(),
        "raw": raw if any(item is not None for item in raw) else None,
        "chunk_count": len(collected),
        "chunk_latencies_ms": latencies,
        "first_latency_ms": latencies[0] if latencies else None,
        # Integer ms: the underlying measurements are integer ms, so a fractional
        # mean would imply precision the numbers do not have.
        "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else None,
        "total_latency_ms": sum(latencies) if latencies else None,
    }


def set_reference(session_id: str, reference_text: str) -> None:
    pipe = _redis.pipeline()
    pipe.hset(_meta_key(session_id), "reference_text", reference_text)
    pipe.expire(_meta_key(session_id), _settings.live_session_ttl_sec)
    pipe.execute()


def close(session_id: str) -> None:
    """Drop a session's keys once it has been persisted.

    Best-effort: the TTL would clear them anyway, so a failure here is not worth
    failing a finalize over.
    """
    session = meta(session_id)
    asr_ids = session["asr_ids"] if session else []
    keys = [_meta_key(session_id)] + [_parts_key(session_id, asr_id) for asr_id in asr_ids]
    _redis.delete(*keys)
