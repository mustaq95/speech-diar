"""Translates DiariZen's native RTTM output into the shared contract.

RTTM line shape: `SPEAKER <uniq_id> <channel> <start> <dur> <NA> <NA> <spk_label> <NA> <NA>`.
Speaker labels are re-based to zero-based indices by first appearance. One
segment per RTTM line -- no merging, even for adjacent same-speaker lines.
"""

from packages.shared_contracts.schemas import DiarizationModelRun, DiarizationSegment

from ..base_model import ModelAdapter
from .runner import DiarizenRawOutput


class DiarizenAdapter(ModelAdapter[DiarizenRawOutput]):
    name = "DiariZen"
    short = "DiariZen"
    description = "WavLM-Large + Conformer local EEND, followed by global clustering · unbounded speaker count"

    def audio_duration_sec(self, raw: DiarizenRawOutput) -> float | None:
        return raw.get("audio_duration_sec")

    def adapt(self, raw: DiarizenRawOutput) -> DiarizationModelRun:
        speaker_index: dict[str, int] = {}
        segs: list[DiarizationSegment] = []

        for line in raw.get("rttm", "").splitlines():
            fields = line.split()
            if len(fields) < 8 or fields[0] != "SPEAKER":
                continue
            start = float(fields[3])
            duration = float(fields[4])
            label = fields[7]
            spk = speaker_index.setdefault(label, len(speaker_index))
            segs.append(DiarizationSegment(spk=spk, s=start, e=start + duration))

        return DiarizationModelRun(
            id="diarizen",
            name=self.name,
            short=self.short,
            description=self.description,
            segs=segs,
            num_spk=len(speaker_index),
        )
