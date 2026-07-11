"""Translates the 3D-Speaker CAM++ Clustering Diarizer's native RTTM output
into the shared contract.

RTTM line shape: `SPEAKER <uniq_id> <channel> <start> <dur> <NA> <NA> <spk_label> <NA> <NA>`.
Speaker labels (integer cluster ids from 3D-Speaker's clustering step) are
re-based to zero-based indices by first appearance. One segment per RTTM
line -- no merging, even for adjacent same-speaker lines.
"""

from packages.shared_contracts.schemas import DiarizationModelRun, DiarizationSegment

from ..base_model import ModelAdapter
from .runner import Speaker3dClusteringRawOutput


class Speaker3dClusteringAdapter(ModelAdapter[Speaker3dClusteringRawOutput]):
    name = "3D-Speaker CAM++ Clustering"
    short = "3D-Speaker"
    description = "VAD + CAM++ speaker embeddings + clustering · ASR-free, unbounded speaker count"

    def audio_duration_sec(self, raw: Speaker3dClusteringRawOutput) -> float | None:
        return raw.get("audio_duration_sec")

    def adapt(self, raw: Speaker3dClusteringRawOutput) -> DiarizationModelRun:
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
            id="3d-speaker-clustering",
            name=self.name,
            short=self.short,
            description=self.description,
            segs=segs,
            num_spk=len(speaker_index),
        )
