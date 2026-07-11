"""Translates VibeVoice-ASR's native output into the shared contract.

Native segment shape: `{"Start": float, "End": float, "Speaker": int,
"Content": str}`. `Content` (the transcript) is dropped here -- the contract
is diarization-only, the same call every other ASR-backed engine in this
platform makes (azure, azure-batch, nim-sortformer-*).

`Speaker` is already an int, but it is re-based to a zero-based index by first
appearance anyway: the model emits speaker ids as generated *text*, so nothing
guarantees they start at 0 or run contiguously. One segment per entry -- no
merging, even for adjacent same-speaker entries.
"""

from packages.shared_contracts.schemas import DiarizationModelRun, DiarizationSegment

from ..base_model import ModelAdapter
from .runner import VibeVoiceRawOutput


class VibeVoiceAdapter(ModelAdapter[VibeVoiceRawOutput]):
    name = "VibeVoice-ASR"
    short = "VibeVoice"
    description = "Microsoft VibeVoice-ASR 8B · single-pass joint ASR + diarization + timestamping"

    def audio_duration_sec(self, raw: VibeVoiceRawOutput) -> float | None:
        return raw.get("audio_duration_sec")

    def adapt(self, raw: VibeVoiceRawOutput) -> DiarizationModelRun:
        speaker_index: dict[int, int] = {}
        segs: list[DiarizationSegment] = []

        for entry in raw.get("segments", []):
            spk = speaker_index.setdefault(entry["Speaker"], len(speaker_index))
            segs.append(
                DiarizationSegment(spk=spk, s=float(entry["Start"]), e=float(entry["End"]))
            )

        return DiarizationModelRun(
            id="vibevoice",
            name=self.name,
            short=self.short,
            description=self.description,
            segs=segs,
            num_spk=len(speaker_index),
        )
