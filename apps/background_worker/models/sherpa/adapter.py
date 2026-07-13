"""Translates sherpa-onnx native output into the shared contract."""

from packages.shared_contracts.schemas import DiarizationModelRun, DiarizationSegment

from ..base_model import ModelAdapter
from .runner import SherpaRawOutput


class SherpaAdapter(ModelAdapter[SherpaRawOutput]):
    name = "Sherpa-ONNX"
    short = "Sherpa"
    description = "Pyannote segmentation-3.0 + NeMo TitaNet-small (EN) embeddings, clustered (onnxruntime)"

    def adapt(self, raw: SherpaRawOutput) -> DiarizationModelRun:
        speaker_index: dict[int, int] = {}
        segs = []
        for segment in raw:
            spk = speaker_index.setdefault(segment["speaker"], len(speaker_index))
            segs.append(DiarizationSegment(spk=spk, s=segment["start"], e=segment["end"]))
        return DiarizationModelRun(
            id="sherpa",
            name=self.name,
            short=self.short,
            description=self.description,
            segs=segs,
            num_spk=len(speaker_index),
        )
