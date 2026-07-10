"""Translates the NeMo Clustering Diarizer's native RTTM output into the
shared contract.

RTTM line shape: `SPEAKER <uniq_id> <channel> <start> <dur> <NA> <NA> <spk_label> <NA> <NA>`.
Speaker labels (e.g. "speaker_0", "speaker_3") are re-based to zero-based
indices by first appearance. One segment per RTTM line -- no merging, even
for adjacent same-speaker lines.
"""

from packages.shared_contracts.schemas import DiarizationModelRun, DiarizationSegment

from ..base_model import ModelAdapter
from .runner import NemoClusteringRawOutput


class NemoClusteringAdapter(ModelAdapter[NemoClusteringRawOutput]):
    name = "NeMo Clustering Diarizer"
    short = "NeMo Clustering"
    description = "MarbleNet VAD + TitaNet embeddings + spectral clustering \u00b7 cascaded, unbounded speaker count"

    def audio_duration_sec(self, raw: NemoClusteringRawOutput) -> float | None:
        return raw.get("audio_duration_sec")

    def adapt(self, raw: NemoClusteringRawOutput) -> DiarizationModelRun:
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
            id="nemo-clustering",
            name=self.name,
            short=self.short,
            description=self.description,
            segs=segs,
            num_spk=len(speaker_index),
        )
