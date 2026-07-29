"""Translates the CTC aligner's native spans into the shared contract.

Native shape: one dict per word from `postprocess_results`, carrying
`{"text", "start", "end", "score"}`. The `text` is the ORIGINAL word, not the
romanized form the aligner matched on — romanization exists only to get Latin
and Arabic script into one CTC vocabulary, and the panel must display what the
ASR actually said.

A word the aligner could not place keeps its position in the list with
`s`/`e` left None. It is neither dropped nor given an interpolated time: the UI
shows it as un-highlightable rather than implying a measurement nobody made.

`<star>` entries are the library's own gap markers between aligned regions, not
speech, and are skipped.
"""

from packages.shared_contracts.schemas import TranscriptWord

from .runner import CtcAlignerRawOutput

_STAR = "<star>"


def adapt(raw: CtcAlignerRawOutput) -> list[TranscriptWord]:
    words: list[TranscriptWord] = []
    for span in raw:
        text = (span.get("text") or "").strip()
        if not text or text == _STAR:
            continue
        start, end = span.get("start"), span.get("end")
        words.append(
            TranscriptWord(
                w=text,
                s=float(start) if start is not None else None,
                e=float(end) if end is not None else None,
                score=float(span["score"]) if span.get("score") is not None else None,
            )
        )
    return words
