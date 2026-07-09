"""Translates WhisperX native output into the shared contract.

WhisperX emits ASR segments with string speaker labels ("SPEAKER_00"); the
adapter maps labels to zero-based indices and drops everything the contract
does not need (words, confidences, language).
"""

from packages.shared_contracts.schemas import DiarizationModelRun, DiarizationSegment

from ..base_model import ModelAdapter
from .runner import WhisperXRawOutput


class WhisperXAdapter(ModelAdapter[WhisperXRawOutput]):
    name = "WhisperX Diarization"
    short = "WhisperX"
    description = "ASR-aligned diarization"

    def adapt(self, raw: WhisperXRawOutput) -> DiarizationModelRun:
        speaker_index: dict[str, int] = {}
        segs = []
        for segment in raw.get("segments", []):
            label = segment.get("speaker", "SPEAKER_00")
            spk = speaker_index.setdefault(label, len(speaker_index))
            segs.append(DiarizationSegment(spk=spk, s=segment["start"], e=segment["end"]))
        return DiarizationModelRun(
            id="whisperx",
            name=self.name,
            short=self.short,
            description=self.description,
            segs=segs,
            num_spk=len(speaker_index),
        )
