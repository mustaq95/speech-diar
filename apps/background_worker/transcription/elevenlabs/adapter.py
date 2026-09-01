"""Translates ElevenLabs Scribe's response into a transcript string.

Native shape: the list `runner.run` returns, one entry per call,

    [{"response": {"text": ..., "language_code": "ara",
                   "language_probability": 0.968,
                   "words": [{"text": ..., "start": ..., "end": ...,
                              "type": "word"|"spacing"|"audio_event",
                              "logprob": ...}, ...]},
      "latency_ms": ..., "segment_index": ...}, ...]

Only `text` is read. `language_code` / `language_probability` are the engine's
own detection result and `words` are its timings; both are real data this
contract has no field for, and both survive verbatim in
`TranscriptResult.raw_output`, served by
`GET /evaluations/{id}/transcript/{asr_id}/raw`.

`text` is passed through exactly as returned, INCLUDING the bracketed
`audio_event` markers ("[phone chimes]"). Stripping them would be this code
improving a measurement: they are part of what the engine emitted and they cost
it real WER against a reference that contains no such marker.
"""

from .runner import ElevenLabsRawOutput, text_of


def adapt(raw: ElevenLabsRawOutput) -> str:
    segments = sorted(raw, key=lambda part: part.get("segment_index", 0))
    return " ".join(text for part in segments if (text := text_of(part))).strip()


def segment_count(raw: ElevenLabsRawOutput) -> int:
    return len(raw)


def adapt_stream(frames: list[dict]) -> str:
    """Concatenate the text from a replay-live pass through Scribe realtime.

    Different native shape from `adapt` above: those come from the batch HTTP
    product (one `{"response": {"text": ...}, ...}` per POST), these are the
    realtime WebSocket's own frames (`committed_transcript` /
    `committed_transcript_with_timestamps`, each with `text` at the top
    level). Only committed frames are here -- partials are dropped upstream
    for the same reason a live run does not score them.

    Order is arrival order (which is also chronological, because the server
    emits one segment as its audio is consumed).
    """
    return " ".join(
        text
        for frame in frames
        if (text := str(frame.get("text") or "").strip())
    ).strip()


def detected_language(raw: ElevenLabsRawOutput) -> str | None:
    """The language this engine says it detected, or None if absent.

    Here rather than in the pipeline because only an adapter may read the native
    shape. Nothing stores this today; it exists so the detection result is
    reachable without re-parsing `raw_output` elsewhere.
    """
    for part in raw:
        code = (part.get("response") or {}).get("language_code")
        if code:
            return str(code)
    return None
