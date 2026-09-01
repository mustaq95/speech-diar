"""Translates ADEO Qwen3-ASR's response into a transcript string.

Native shape: the list `runner.run` returns, one entry per call,

    [{"response": {"text": ..., "usage": {"type": "duration", "seconds": ...}},
      "latency_ms": ..., "segment_index": ...}, ...]

The endpoint's own JSON is the "response" value, kept intact; the sibling keys
are what the runner measured. Only `text` is read here -- `usage` is duration
accounting, not speech. Everything dropped survives verbatim in
`TranscriptResult.raw_output`.

Text is passed through exactly as generated: no normalization, no punctuation
repair, no script folding. This engine emits Arabic and English in one string
and that mixture is the measurement.
"""

from .runner import AdeoQwen3RawOutput, text_of


def adapt(raw: AdeoQwen3RawOutput) -> str:
    segments = sorted(raw, key=lambda part: part.get("segment_index", 0))
    return " ".join(text for part in segments if (text := text_of(part))).strip()


def segment_count(raw: AdeoQwen3RawOutput) -> int:
    """How many pieces this engine cut the recording into -- always 1 for the
    whole-file batch call, reported rather than left NULL so the column does not
    render as 0 ("the engine produced nothing")."""
    return len(raw)
