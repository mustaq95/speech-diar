"""Translates Inception-STT's per-segment responses into a transcript string.

Native shape: the list `runner.run` returns, one entry per audio segment,

    [{"response": {"text": ..., "task": ..., "audio_duration": ...,
                   "usage": ..., "word_timestamps": ...},
      "latency_ms": ..., "segment_index": ...,
      "segment_start": ..., "segment_end": ...}, ...]

The gateway's own JSON is the "response" value, kept intact; the sibling keys are
what the runner measured. Only `text` is read here. `usage` is token accounting,
not speech, and `word_timestamps` was observed null from this gateway and is
relied on nowhere. Everything dropped survives verbatim in
`TranscriptResult.raw_output`, which is what
`GET /evaluations/{id}/transcript/{asr_id}/raw` serves.

Segment texts are joined with a single space and otherwise passed through exactly
as generated — no normalization, no punctuation repair, no de-duplication across
a boundary. A word split by a segment boundary stays split. That is the engine's
transport showing through, and smoothing it here would be this code editing a
measurement.
"""

from .runner import InceptionRawOutput, text_of


def adapt(raw: InceptionRawOutput) -> str:
    segments = sorted(raw, key=lambda part: part.get("segment_index", 0))
    return " ".join(text for part in segments if (text := text_of(part))).strip()
