"""Translates the ADEO gateway's pyannoteAI DiarizationJob into the shared contract."""

from packages.shared_contracts.schemas import DiarizationModelRun, DiarizationSegment

from ..base_model import ModelAdapter
from .runner import PyannoteAdeoRawOutput


class PyannoteAdeoAdapter(ModelAdapter[PyannoteAdeoRawOutput]):
    name = "PyAnnote Community-1 (ADEO)"
    short = "PyAnnote Community-1 (ADEO)"
    description = "community-1 on the ADEO on-prem deployment, via the LiteLLM gateway (/v1/diarize)"

    def audio_duration_sec(self, raw: PyannoteAdeoRawOutput) -> float | None:
        return raw.get("audio_duration_sec")

    def adapt(self, raw: PyannoteAdeoRawOutput) -> DiarizationModelRun:
        job = raw["job"]
        if job.get("status") != "succeeded":
            raise ValueError(f"pyannote-adeo job {job.get('jobId')} status {job.get('status')!r}")
        # `diarization` keeps overlapping turns, matching what the in-process
        # pyannote row shows. `exclusiveDiarization` stays in raw_output only.
        speaker_index: dict[str, int] = {}
        segs = []
        for turn in job["output"]["diarization"]:
            spk = speaker_index.setdefault(turn["speaker"], len(speaker_index))
            segs.append(DiarizationSegment(spk=spk, s=turn["start"], e=turn["end"]))
        return DiarizationModelRun(
            id="pyannote-adeo",
            name=self.name,
            short=self.short,
            description=self.description,
            segs=segs,
            num_spk=len(speaker_index),
        )
