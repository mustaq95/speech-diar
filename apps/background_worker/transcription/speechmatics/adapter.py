"""Translates Speechmatics' json-v2 transcript into a transcript string.

Native shape (the API's own body, plus the two keys the runner attached):

    {"format": "2.9",
     "job": {...},
     "metadata": {"language_pack_info": {"language_description": "Arabic and
                    English", "writing_direction": ..., "itn": ...},
                  "language_identification": {"predicted_language": "ar"},
                  "transcription_config": {...}, ...},
     "results": [{"type": "word"|"punctuation",
                  "start_time": ..., "end_time": ...,
                  "attaches_to": "previous",           # punctuation only
                  "alternatives": [{"content": ..., "confidence": ...,
                                    "language": "ar_en", "speaker": "UU"}]}, ...],
     "job_id": ..., "turnaround_ms": ...}

Only the first alternative's `content` is read. Confidences, per-token times,
per-token language tags and the speaker labels are real data this contract has
no field for; all of it survives verbatim in `TranscriptResult.raw_output`.

**Punctuation is attached, not space-joined.** A `punctuation` result carries
`attaches_to: "previous"`, and honouring it is what makes the output read as
"الحكومية، نجحوا" rather than "الحكومية ، نجحوا". This is not cosmetic: CER is
computed over characters, so a spurious space before every comma is a real
character error charged to the engine for a separator this adapter invented.
Word content itself is passed through untouched.
"""

from typing import Any

from .runner import SpeechmaticsRawOutput


def _content(result: dict[str, Any]) -> str:
    alternatives = result.get("alternatives") or []
    if not alternatives:
        return ""
    return str((alternatives[0] or {}).get("content") or "")


def adapt(raw: SpeechmaticsRawOutput) -> str:
    if not isinstance(raw, dict):
        return ""
    out: list[str] = []
    for result in raw.get("results") or []:
        if not isinstance(result, dict):
            continue
        text = _content(result)
        if not text:
            continue
        if result.get("attaches_to") == "previous" and out:
            out[-1] += text
        else:
            out.append(text)
    return " ".join(out).strip()


def adapt_stream(frames: list[dict]) -> str:
    """Concatenate the text from a replay-live pass through Speechmatics RT.

    Different native shape from `adapt` above: the batch runner returns one
    json-v2 body with a `results` array of tokens; the realtime WebSocket
    returns a series of `AddTranscript` frames each with its own `metadata`
    block. `metadata.transcript` is the already-joined-and-punctuated string
    for that segment (the RT protocol does the token join server-side, so
    the `results` array's alternate-content shape does not apply here).

    Order is arrival order (which is also chronological).
    """
    return " ".join(
        text
        for frame in frames
        if (text := str((frame.get("metadata") or {}).get("transcript") or "").strip())
    ).strip()


def language_pack(raw: SpeechmaticsRawOutput) -> str | None:
    """The language pack the job actually ran, as the API describes it.

    Worth reaching for because it is how you tell a bilingual `ar_en` run from a
    monolingual one that silently dropped the English -- the two differ in
    content, not in status.
    """
    if not isinstance(raw, dict):
        return None
    info = (raw.get("metadata") or {}).get("language_pack_info") or {}
    return info.get("language_description")


def word_count(raw: SpeechmaticsRawOutput) -> int | None:
    """How many `word` results the job returned. Punctuation is excluded: it is
    not speech and counting it would inflate the figure."""
    if not isinstance(raw, dict):
        return None
    results = raw.get("results")
    if not isinstance(results, list):
        return None
    return sum(1 for r in results if isinstance(r, dict) and r.get("type") == "word")
