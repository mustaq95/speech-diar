"""Translates PyAnnote 3.1 native output into the shared contract."""

from packages.shared_contracts.schemas import DiarizationModelRun, DiarizationSegment

from ..base_model import ModelAdapter
from .runner import Pyannote31RawOutput


class Pyannote31Adapter(ModelAdapter[Pyannote31RawOutput]):
    name = "PyAnnote 3.1"
    short = "PyAnnote 3.1"
    description = "End-to-end neural speaker diarization · speaker-diarization-3.1 (segmentation-3.0 + WeSpeaker embeddings + agglomerative clustering)"

    def adapt(self, raw: Pyannote31RawOutput) -> DiarizationModelRun:
        speaker_index: dict[str, int] = {}
        segs = []
        for track in raw:
            spk = speaker_index.setdefault(track["label"], len(speaker_index))
            segs.append(DiarizationSegment(spk=spk, s=track["start"], e=track["end"]))
        return DiarizationModelRun(
            id="pyannote-3-1",
            name=self.name,
            short=self.short,
            description=self.description,
            segs=segs,
            num_spk=len(speaker_index),
        )
