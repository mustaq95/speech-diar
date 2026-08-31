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


def segment_count(raw: InceptionRawOutput) -> int:
    """How many pieces this engine actually cut the recording into.

    Here rather than in the pipeline because the pipeline is not allowed to know
    what the native output looks like -- `len(raw)` is only meaningful to code
    that knows this engine returns one entry per segment.

    A batch run's chunk count used to be left NULL and rendered as 0, which reads
    as "the engine produced nothing" for a run that in fact made N calls.
    """
    return len(raw)
