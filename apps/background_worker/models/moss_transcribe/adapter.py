"""Translates MOSS-Transcribe-Diarize's native output into the shared contract.

Native shape: a single generated string in which every turn is
`[start][Sxx]text[end]`, concatenated with no separator, e.g.

    [0.48][S01]Welcome everyone[1.66][12.26][S02]The pipeline is ready[13.81]

The transcript text is dropped here -- the contract is diarization-only, the
same call every other ASR-backed engine in this platform makes (azure,
azure-batch, vibevoice, nim-sortformer-*).

Unlike vibevoice (whose runner parses), the parse lives in the adapter: the
native shape is the raw string, and the adapter is the only code allowed to
understand it.

The model also emits optional acoustic event annotations ([Music], [Silence],
...). Those carry no `[Sxx]` tag and so do not match `_SEG_RE` at all: they are
skipped rather than assigned a speaker the model never reported. An event
occurring *inside* a turn is kept as part of that turn's (discarded) text --
`.*?` only stops at the next `[<number>]`, and an event tag is not numeric.

`Sxx` is re-based to a zero-based index by first appearance: the model emits
speaker ids as generated *text*, so nothing guarantees they start at S01 or run
contiguously. One segment per turn -- no merging, even for adjacent
same-speaker turns.
"""

import re

from packages.shared_contracts.schemas import DiarizationModelRun, DiarizationSegment

from ..base_model import ModelAdapter
from .runner import MossTranscribeRawOutput

_SEG_RE = re.compile(
    r"\[(\d+(?:\.\d+)?)\]"  # start seconds
    r"\[S(\d+)\]"  # speaker tag
    r".*?"  # transcript -- matched so the end timestamp anchors, then dropped
    r"\[(\d+(?:\.\d+)?)\]",  # end seconds
    re.DOTALL,
)


class MossTranscribeAdapter(ModelAdapter[MossTranscribeRawOutput]):
    name = "MOSS-Transcribe-Diarize"
    short = "MOSS"
    description = "OpenMOSS MOSS-Transcribe-Diarize 0.9B · single-pass joint ASR + diarization + timestamping"

    def audio_duration_sec(self, raw: MossTranscribeRawOutput) -> float | None:
        return raw.get("audio_duration_sec")

    def adapt(self, raw: MossTranscribeRawOutput) -> DiarizationModelRun:
        speaker_index: dict[str, int] = {}
        segs: list[DiarizationSegment] = []

        for match in _SEG_RE.finditer(raw.get("text") or ""):
            start, label, end = match.group(1), match.group(2), match.group(3)
            spk = speaker_index.setdefault(label, len(speaker_index))
            segs.append(DiarizationSegment(spk=spk, s=float(start), e=float(end)))

        return DiarizationModelRun(
            id="moss-transcribe",
            name=self.name,
            short=self.short,
            description=self.description,
            segs=segs,
            num_spk=len(speaker_index),
        )
