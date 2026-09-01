"""Translates VibeVoice-ASR's segment list into a transcript string.

Native shape (the same object the diarization adapter reads):

    {"audio_duration_sec": 61.44,
     "segments": [{"Start": 1.78, "End": 16.0, "Speaker": 0,
                   "Content": "... Arabic and English in one string ..."}, ...]}

Only `Content` is read here. `Start`/`End`/`Speaker` are the diarization half and
survive verbatim in `TranscriptResult.raw_output`.

**`Speaker` is optional and is never read here anyway**, but `Content` is fetched
with `.get()` for the same reason: the observed payload contained a segment with
no `Speaker` key, so this payload is not uniformly shaped and indexing it would
raise on real output.

`Content` is passed through exactly as generated, including the non-speech
markers this model emits (`[Music]`, `[Unintelligible Speech]`). They are part of
what the engine produced and they cost it real WER against a reference that has
no such marker; removing them here would be this code flattering the engine.

Segments are joined with a single space in the order returned. No merging, no
de-duplication across a boundary.
"""

from .runner import VibeVoiceRawOutput


def adapt(raw: VibeVoiceRawOutput) -> str:
    if not isinstance(raw, dict):
        return ""
    segments = raw.get("segments") or []
    return " ".join(
        text
        for segment in segments
        if isinstance(segment, dict) and (text := str(segment.get("Content") or "").strip())
    ).strip()


def segment_count(raw: VibeVoiceRawOutput) -> int | None:
    """How many segments the model returned.

    Not a chunk count: nothing split this audio, the MODEL chose these
    boundaries. Reported because the column is "how many pieces this run was
    made of" and one whole-file call that yields six turns is honestly six.
    """
    if not isinstance(raw, dict):
        return None
    segments = raw.get("segments")
    return len(segments) if isinstance(segments, list) else None
