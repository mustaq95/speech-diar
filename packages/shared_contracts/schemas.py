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

ModelLifecycleState = Literal["unloaded", "starting", "ready", "in_use", "stopping", "unhealthy"]

#: Which half of the transcript pipeline a run is in. The two stages are two
#: separate RQ jobs (ASR, then alignment), each timed on its own.
TranscriptStage = Literal["asr", "align"]

#: Where the ASR ran. `online` streamed the audio to a remote endpoint;
#: `offline` kept it on this host. On a produced transcript this is derived from
#: the engine that actually ran, not from the mode a new run would use.
TranscriptionMode = Literal["online", "offline"]


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
    loading_started_at: datetime | None = Field(
        default=None,
        description=(
            "When the GPU supervisor granted a slot and began waiting for the model's "
            "container to become healthy. None for models that never needed a cold start. "
            "The interval [loadingStartedAt, startedAt) is cold-start wait, not inference."
        ),
    )
    started_at: datetime | None = Field(
        default=None, description="When inference actually began (container confirmed healthy)"
    )
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


class TranscriptWord(ContractModel):
    """One word of the transcript, with the timing the aligner gave it.

    `s`/`e` are None when the aligner could not place the word (its characters
    are outside the CTC vocabulary, e.g. a Latin token in an Arabic-script
    model). Such a word is still carried, unplaced, rather than dropped or
    given a guessed time.
    """

    w: str = Field(description="The word exactly as the ASR emitted it, for display")
    s: float | None = Field(default=None, ge=0, description="Start time in seconds, or None if unaligned")
    e: float | None = Field(default=None, ge=0, description="End time in seconds, or None if unaligned")
    score: float | None = Field(
        default=None,
        description=(
            "The aligner's score for this word: a mean log-probability, so <= 0 and "
            "higher is better. NOT a 0..1 confidence -- do not render it as a percentage."
        ),
    )


class TranscriptRun(ContractModel):
    """One engine's live-speech transcript of one audio file: ASR text plus
    word-level timings, with each stage's real measured cost.

    Separate from `DiarizationModelRun` on purpose — this is transcription, not
    diarization, and the diarization contract has no place for text.

    One transcript per (audio file, engine): a recording holds an online AND an
    offline run at once, which is the comparison this platform exists to make.
    `GET /evaluations/{id}/transcript` therefore returns a list of these.
    """

    audio_file_id: int
    status: ModelStatus = Field(description="queued|running|done|failed across both stages")
    stage: TranscriptStage | None = Field(
        default=None, description="Stage currently running, or the stage that failed; None once done"
    )
    asr_id: str = Field(description="Engine that actually ran: 'hamsa' | 'cohere-transcribe'")
    mode: TranscriptionMode = Field(
        description="Where the ASR ran, derived from asr_id -- NOT from the current setting"
    )
    asr_name: str = Field(description="Display name of the ASR engine, e.g. 'TryHamsa STT'")
    aligner_name: str = Field(description="Display name of the forced aligner")
    text: str = Field(default="", description="Full ASR transcript; present as soon as the ASR stage finishes")
    words: list[TranscriptWord] = Field(default_factory=list, description="Empty until the alignment stage finishes")
    asr_ms: int | None = Field(default=None, description="Measured wall-clock time of the ASR stage in ms")
    align_ms: int | None = Field(default=None, description="Measured wall-clock time of the alignment stage in ms")
    error: str | None = Field(default=None, description="Failure reason when status == 'failed'")


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


class ModelContainerStatus(ContractModel):
    """One entry in `GET /models/status` — a model's real-time GPU-residency
    lifecycle state, platform-wide (shared across every evaluation, not
    scoped to one upload), derived fresh from Docker + RQ at request time
    rather than read from a stored mirror. Models with no container to
    manage (pyannote runs in-process) have no entry — never fabricate a
    state for them."""

    model_id: str
    state: ModelLifecycleState
    active_job_count: int = Field(ge=0, description="In-flight jobs on this model; >0 means untouchable by eviction/idle-unload")
    queued_job_count: int = Field(ge=0, description="Jobs waiting on this model because the residency cap is full")
    last_error: str | None = Field(default=None, description="Most recent failure reason, if any (e.g. a failed cold start)")


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
