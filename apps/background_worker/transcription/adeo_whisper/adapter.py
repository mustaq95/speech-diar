"""Translates ADEO Whisper's response into a transcript string.

Native shape: the list `runner.run` returns, one entry per call,

    [{"response": {"text": ...}, "latency_ms": ..., "segment_index": ...}, ...]

Only `text` is read here; everything else survives verbatim in
`TranscriptResult.raw_output`. Whisper builds commonly also return `language`
and, with a verbose format, `segments` -- neither is requested and neither is
relied on, so a build that omits them still adapts cleanly.

This shape is inferred from the sibling Qwen3 pod on the same host, NOT observed:
this pod was 504 throughout probing. See `runner.py`. If it turns out to answer
differently, `post_transcription` raises `OpenAiAsrResponseError` naming the body
rather than silently storing an empty transcript.

Text is passed through exactly as generated.
"""

from .runner import AdeoWhisperRawOutput, text_of


def adapt(raw: AdeoWhisperRawOutput) -> str:
    segments = sorted(raw, key=lambda part: part.get("segment_index", 0))
    return " ".join(text for part in segments if (text := text_of(part))).strip()


def segment_count(raw: AdeoWhisperRawOutput) -> int:
    return len(raw)
