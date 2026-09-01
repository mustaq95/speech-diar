"""Translates MOSS-Transcribe-Diarize's generated string into a transcript.

Native shape (the same object the diarization adapter reads):

    {"text": "[2.16][S01] ... [6.32][6.64][S01] ... ", "usage": {...}}

MOSS emits ASR, diarization and timestamps as ONE generated string: each turn is
`[start][Snn] text[end]`. The diarization adapter reads the speaker and time
markers; this one reads the words between them and drops the markers, because
`TranscriptResult.text` is scored against a reference that contains no such
markup and leaving them in would charge the engine insertions for the format it
was asked to produce.

That is the one and only thing removed. The words themselves are passed through
untouched -- including the English that this model renders as Arabic
translation, and any stray token it emits. Those are the measurement.

Turns are joined with a single space, in the order generated. No merging of
adjacent turns, no punctuation repair.
"""

import re
from typing import Any

from .runner import MossRawOutput

#: `[12.34]` timestamps and `[S01]` speaker tags, the two markers MOSS interleaves
#: with its words. Anchored to those exact forms rather than "any bracketed run"
#: so a bracketed non-speech token the model emits as CONTENT survives.
_MARKER = re.compile(r"\[\d+(?:\.\d+)?\]|\[S\d+\]")


def _strip_markers(text: str) -> str:
    return re.sub(r"\s+", " ", _MARKER.sub(" ", text)).strip()


def adapt(raw: MossRawOutput) -> str:
    if not isinstance(raw, dict):
        return ""
    return _strip_markers(str(raw.get("text") or ""))


def turns(raw: MossRawOutput) -> list[dict[str, Any]]:
    """The `[start][Snn] text[end]` turns, parsed.

    Unused by the transcript surface, which scores one flat string. It exists so
    the speaker/time data MOSS genuinely produced is reachable from this side
    without re-parsing `raw_output` in code that is not an adapter.
    """
    out: list[dict[str, Any]] = []
    if not isinstance(raw, dict):
        return out
    for start, speaker, content in re.findall(
        r"\[(\d+(?:\.\d+)?)\]\[(S\d+)\]([^\[]*)", str(raw.get("text") or "")
    ):
        if text := content.strip():
            out.append({"start": float(start), "speaker": speaker, "text": text})
    return out
