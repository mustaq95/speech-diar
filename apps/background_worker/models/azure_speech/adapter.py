"""Translates Azure ConversationTranscriber output into the shared contract.

Azure emits ASR phrases with string speaker ids ("Guest-1", "Unknown") and
100-nanosecond tick offsets; the adapter maps speaker ids to zero-based
indices by first appearance and converts ticks to seconds. One segment per
Azure-reported phrase — no merging, so the KPI (how the model itself splits
speaker turns) stays raw.
"""

from packages.shared_contracts.schemas import DiarizationModelRun, DiarizationSegment

from ..base_model import ModelAdapter
from .runner import AzureSpeechRawOutput

TICKS_PER_SEC = 10_000_000


def duration_sec(raw: AzureSpeechRawOutput) -> float:
    """Total audio duration: from the file header, else the last phrase end."""
    from_header = raw.get("audio_duration_sec")
    if from_header:
        return float(from_header)
    ends = [
        (p["offset_ticks"] + p["duration_ticks"]) / TICKS_PER_SEC
        for p in raw.get("phrases", [])
    ]
    return max(ends, default=0.0)


class AzureSpeechAdapter(ModelAdapter[AzureSpeechRawOutput]):
    name = "Azure Speech Diarization"
    short = "Azure"
    description = "Azure AI Speech · real-time conversation transcription"

    def audio_duration_sec(self, raw: AzureSpeechRawOutput) -> float | None:
        return duration_sec(raw) or None

    def adapt(self, raw: AzureSpeechRawOutput) -> DiarizationModelRun:
        speaker_index: dict[str, int] = {}
        segs: list[DiarizationSegment] = []

        phrases = sorted(raw.get("phrases", []), key=lambda p: p["offset_ticks"])
        for phrase in phrases:
            spk = speaker_index.setdefault(phrase["speaker_id"], len(speaker_index))
            s = phrase["offset_ticks"] / TICKS_PER_SEC
            e = s + phrase["duration_ticks"] / TICKS_PER_SEC
            segs.append(DiarizationSegment(spk=spk, s=s, e=e))

        return DiarizationModelRun(
            id="azure",
            name=self.name,
            short=self.short,
            description=self.description,
            segs=segs,
            num_spk=len(speaker_index),
        )
