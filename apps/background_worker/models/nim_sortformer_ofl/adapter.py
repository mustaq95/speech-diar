"""Translates the offline Parakeet-Sortformer NIM's native word list into the
shared contract.

The NIM emits word-level `speaker_tag`s (0-based, proto3 default 0 — never
"missing"); the adapter groups consecutive same-tag words into one segment
per contiguous speaker run and re-bases tags to zero-based indices by first
appearance. One segment per model-reported turn — no merging.
"""

from packages.shared_contracts.schemas import DiarizationModelRun, DiarizationSegment

from .. import _nim_shared
from ..base_model import ModelAdapter
from .runner import NimSortformerOflRawOutput


class NimSortformerOflAdapter(ModelAdapter[NimSortformerOflRawOutput]):
    name = "Parakeet-Sortformer NIM (Offline)"
    short = "Parakeet Offline"
    description = "NVIDIA Parakeet 1.1B RNNT Multilingual + Sortformer diarizer · NIM offline batch"

    def audio_duration_sec(self, raw: NimSortformerOflRawOutput) -> float | None:
        return raw.get("audio_duration_sec")

    def adapt(self, raw: NimSortformerOflRawOutput) -> DiarizationModelRun:
        speaker_index: dict[int, int] = {}
        segs: list[DiarizationSegment] = []

        turns = _nim_shared.group_words_into_turns(raw.get("words", []))
        for speaker_tag, start_sec, end_sec in turns:
            spk = speaker_index.setdefault(speaker_tag, len(speaker_index))
            segs.append(DiarizationSegment(spk=spk, s=start_sec, e=end_sec))

        return DiarizationModelRun(
            id="nim-sortformer-ofl",
            name=self.name,
            short=self.short,
            description=self.description,
            segs=segs,
            num_spk=len(speaker_index),
        )
