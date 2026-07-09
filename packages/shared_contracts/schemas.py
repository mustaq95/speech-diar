"""Source of truth for every data structure that crosses a service boundary.

`schemas.py` (Pydantic, used by backend_api and background_worker) and
`types.ts` (TypeScript, consumed by apps/frontend) describe the SAME wire
format and must be kept in sync. The wire format is camelCase JSON; the
Python models use snake_case attributes with camelCase aliases.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

ModelStatus = Literal["queued", "running", "done", "failed"]


class ContractModel(BaseModel):
    """Base for all shared contracts: camelCase on the wire, snake_case in Python."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class DiarizationSegment(ContractModel):
    """One contiguous stretch of speech attributed to a single speaker."""

    spk: int = Field(ge=0, description="Zero-based speaker index, stable within one model run")
    s: float = Field(ge=0, description="Start time in seconds from the beginning of the audio")
    e: float = Field(ge=0, description="End time in seconds (exclusive), always >= s")


class DiarizationModelRun(ContractModel):
    """The complete diarization output of one model over one audio file."""

    id: str = Field(description="Stable model identifier, e.g. 'pyannote'")
    name: str = Field(description="Full display name, e.g. 'PyAnnote Audio 3.1'")
    short: str = Field(description="Short label used in pills and compact rows")
    description: str = Field(description="One-line description shown in settings/model cards")
    segs: list[DiarizationSegment] = Field(default_factory=list, description="Speech segments, sorted by start time")
    num_spk: int = Field(default=0, ge=0, description="Distinct speakers detected (max spk + 1)")
    status: ModelStatus | None = Field(default=None, description="queued|running|done|failed; the real, true state")
    error: str | None = Field(default=None, description="Failure reason when status == 'failed'")
    started_at: datetime | None = Field(default=None, description="When the worker started running this model")
    finished_at: datetime | None = Field(default=None, description="When the worker finished running this model")
    processing_ms: int | None = Field(
        default=None,
        description=(
            "Wall-clock run time in ms, started_at->finished_at (excludes queue wait). "
            "For azure-batch, Azure's own reported duration when available."
        ),
    )


class DiarizationEvaluation(ContractModel):
    """Everything the frontend needs to render one evaluation session."""

    audio_file_id: int = Field(description="Primary key of the AudioFile this evaluation belongs to")
    duration_sec: float = Field(ge=0, description="Audio duration in seconds")
    upload_ms: int | None = Field(default=None, description="Client-perceived upload time (browser start -> API response)")
    models: list[DiarizationModelRun]


class QueuedModel(ContractModel):
    """One model's initial state right after being enqueued."""

    id: str
    status: ModelStatus


class UploadAck(ContractModel):
    """Immediate response to `POST /upload` — the client then polls
    `GET /evaluations/{audioFileId}` for the real result."""

    audio_file_id: int
    models: list[QueuedModel]


class UploadTimingUpdate(ContractModel):
    """Body of `PATCH /evaluations/{audioFileId}` — the client's one-time
    measured upload time, persisted so it survives reopening the project."""

    upload_ms: int = Field(ge=0)


class ModelMetadata(ContractModel):
    """One entry in the `GET /models` registry — what a model IS, not a run's output."""

    id: str
    name: str
    short: str
    description: str
    available: bool = Field(description="False for engines that are pluggable stubs (report failed, never fake segments)")


def normalize_model_run(run: DiarizationModelRun) -> DiarizationModelRun:
    """Enforce the contract's invariants (sorted segments, correct num_spk).

    Every worker-side adapter should pass its output through this before the
    result is persisted or served, mirroring `normalizeModelRun()` on the
    frontend side. This may only sort and recount — never merge or drop a
    segment a model actually produced.
    """
    segs = sorted(run.segs, key=lambda seg: seg.s)
    num_spk = max((seg.spk for seg in segs), default=-1) + 1
    return run.model_copy(update={"segs": segs, "num_spk": num_spk})
