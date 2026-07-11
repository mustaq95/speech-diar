"""Translates VibeVoice-ASR's native output into the shared contract.

Native segment shape: `{"Start": float, "End": float, "Speaker": int,
"Content": str}`. `Content` (the transcript) is dropped here -- the contract
is diarization-only, the same call every other ASR-backed engine in this
platform makes (azure, azure-batch, nim-sortformer-*).

Non-speech segments carry NO `Speaker` key at all: the model was trained to
tag intervals it hears as `[Silence]`, `[Music]`, `[Noise]`, `[Human Sounds]`,
`[Environmental Sounds]` or `[Unintelligible Speech]` instead of hallucinating
words over them, and those entries are speaker-less. They are skipped here.
Nothing else is honest: they are not speech turns, and minting a speaker index
for silence would fabricate a speaker the model never reported.

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
            if "Speaker" not in entry:
                continue  # non-speech event ([Silence], [Music], ...) -- no speaker to report
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
