"""Translates Azure batch transcription output into the shared contract.

The batch result has 1-based integer speaker ids and 100-nanosecond tick
offsets per recognized phrase; the adapter re-bases speakers to zero and
converts ticks to seconds. One segment per Azure-reported phrase — no
merging, so the KPI (how the model itself splits speaker turns) stays raw.
"""

from datetime import datetime

from packages.shared_contracts.schemas import DiarizationModelRun, DiarizationSegment

from ..base_model import ModelAdapter
from .runner import AzureBatchRawOutput

TICKS_PER_SEC = 10_000_000


def duration_sec(raw: AzureBatchRawOutput) -> float:
    ticks = raw.get("durationInTicks") or raw.get("durationTicks")
    if ticks:
        return float(ticks) / TICKS_PER_SEC
    ends = [
        (p.get("offsetInTicks", 0) + p.get("durationInTicks", 0)) / TICKS_PER_SEC
        for p in raw.get("recognizedPhrases", [])
    ]
    return max(ends, default=0.0)


class AzureBatchAdapter(ModelAdapter[AzureBatchRawOutput]):
    name = "Azure Speech Batch Diarization"
    short = "Azure Batch"
    description = "Azure AI Speech · batch transcription v3.2 · diarization"

    def audio_duration_sec(self, raw: AzureBatchRawOutput) -> float | None:
        return duration_sec(raw) or None

    def processing_ms(self, raw: AzureBatchRawOutput) -> int | None:
        """Azure's own created->completed bracket for the batch job, when present."""
        timing = raw.get("_azureJobTiming") or {}
        created, finished = timing.get("createdDateTime"), timing.get("lastActionDateTime")
        if not created or not finished:
            return None
        try:
            start = datetime.fromisoformat(created.replace("Z", "+00:00"))
            end = datetime.fromisoformat(finished.replace("Z", "+00:00"))
        except ValueError:
            return None
        return max(0, int((end - start).total_seconds() * 1000))

    def adapt(self, raw: AzureBatchRawOutput) -> DiarizationModelRun:
        speaker_index: dict[int, int] = {}
        segs: list[DiarizationSegment] = []

        phrases = sorted(raw.get("recognizedPhrases", []), key=lambda p: p.get("offsetInTicks", 0))
        for phrase in phrases:
            label = phrase.get("speaker", 0)
            spk = speaker_index.setdefault(label, len(speaker_index))
            s = phrase.get("offsetInTicks", 0) / TICKS_PER_SEC
            e = s + phrase.get("durationInTicks", 0) / TICKS_PER_SEC
            segs.append(DiarizationSegment(spk=spk, s=s, e=e))

        return DiarizationModelRun(
            id="azure-batch",
            name=self.name,
            short=self.short,
            description=self.description,
            segs=segs,
            num_spk=len(speaker_index),
        )
