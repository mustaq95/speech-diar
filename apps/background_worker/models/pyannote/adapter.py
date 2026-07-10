"""Translates PyAnnote native output into the shared contract."""

from packages.shared_contracts.schemas import DiarizationModelRun, DiarizationSegment

from ..base_model import ModelAdapter
from .runner import PyAnnoteRawOutput


class PyAnnoteAdapter(ModelAdapter[PyAnnoteRawOutput]):
    name = "PyAnnote Community-1"
    short = "PyAnnote"
    description = "End-to-end neural speaker diarization · community-1 (v4)"

    def adapt(self, raw: PyAnnoteRawOutput) -> DiarizationModelRun:
        speaker_index: dict[str, int] = {}
        segs = []
        for track in raw:
            spk = speaker_index.setdefault(track["label"], len(speaker_index))
            segs.append(DiarizationSegment(spk=spk, s=track["start"], e=track["end"]))
        return DiarizationModelRun(
            id="pyannote",
            name=self.name,
            short=self.short,
            description=self.description,
            segs=segs,
            num_spk=len(speaker_index),
        )
