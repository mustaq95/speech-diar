"""Translates Hamsa's native WebSocket messages into a transcript string.

Native shape: one message per detected speech segment, each

    {"type": "transcription", "data": {"transcription": "...", "language": "ar", ...}}

with the payload sometimes at the top level instead of under `data` (the
deployments differ, and the reference client tolerates both). Non-transcription
messages — handshake acks, status, anything else — carry no speech and are
skipped rather than coerced into text.

Segments are joined in ARRIVAL ORDER, which is chronological: the server emits
each segment as its audio is consumed. Nothing here reorders, deduplicates or
cleans the text; whatever the engine said is what gets aligned and displayed.

Hamsa can also return its own word timestamps on some deployments. They are
deliberately ignored: this platform times every transcript with one aligner
(`../ctc_aligner/`) so the online and offline modes produce comparable output
rather than two differently-derived timelines.
"""

from .runner import HamsaRawOutput


def adapt(raw: HamsaRawOutput) -> str:
    """Concatenate every transcription segment into one transcript string."""
    segments: list[str] = []
    for message in raw:
        if message.get("type") != "transcription":
            continue
        inner = message.get("data") if isinstance(message.get("data"), dict) else message
        text = inner.get("transcription") or inner.get("text") or ""
        if text.strip():
            segments.append(text.strip())
    return " ".join(segments).strip()
