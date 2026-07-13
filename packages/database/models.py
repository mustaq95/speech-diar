"""SQLAlchemy ORM tables.

EvaluationResult.payload stores the unified contract
(packages/shared_contracts DiarizationModelRun) as JSON — the database never
stores a model's raw native output.
"""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, JSON, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    audio_files: Mapped[list["AudioFile"]] = relationship(back_populates="owner")


class AudioFile(Base):
    __tablename__ = "audio_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    filename: Mapped[str] = mapped_column(String(512))
    # Each lane owns its own location; a row only has the key(s) for the
    # lane(s) actually used — never both copied from one store to the other.
    s3_key: Mapped[str | None] = mapped_column(String(1024), unique=True, nullable=True)
    blob_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    blob_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    duration_sec: Mapped[float] = mapped_column(Float)
    # Client-perceived upload time (browser upload start -> API response),
    # persisted via PATCH once the client has measured it.
    upload_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    owner: Mapped[User] = relationship(back_populates="audio_files")
    results: Mapped[list["EvaluationResult"]] = relationship(back_populates="audio_file")


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    audio_file_id: Mapped[int] = mapped_column(ForeignKey("audio_files.id"), index=True)
    model_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued|running|done|failed
    error: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # DiarizationModelRun JSON
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # Worker-measured timing: started/finished bracket the run itself
    # (excludes queue wait); processing_ms = finished - started.
    queued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Set when the supervisor grants a GPU slot and begins waiting for the
    # model's container to report healthy (see apps/background_worker/
    # supervisor/). started_at is stamped only once the container is
    # confirmed ready and inference actually begins, so
    # processing_ms = finished_at - started_at never includes cold-start
    # wait -- that interval is [loading_started_at, started_at) instead.
    loading_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    processing_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    audio_file: Mapped[AudioFile] = relationship(back_populates="results")


class ModelContainerState(Base):
    """One row per GPU container-managed local model (see
    apps/background_worker/supervisor/registry.py MANAGED_CONTAINERS).
    pyannote has no row: it runs in-process in the worker, not in a
    container the supervisor can start/stop.

    Deliberately stores ONLY what no other system can know — job accounting
    and in-flight claims. Everything else (container running/healthy, queue
    depth, the UI's lifecycle state) is derived at read time from its owner
    (Docker, RQ) by apps/background_worker/supervisor/state.py. An earlier
    version mirrored a full lifecycle_state machine here; every live bug it
    produced was mirror-vs-reality drift, repaired by an ever-growing set of
    daemon sweeps. Claims carry timestamps and their validity is evaluated
    when read (expired or worker-dead => treated as absent), so they cannot
    orphan and need no repair sweep.
    """

    __tablename__ = "model_container_state"

    model_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    active_job_count: Mapped[int] = mapped_column(Integer, default=0)  # >0 => untouchable by eviction/idle-unload
    # Cold-start claim: set when admission grants a slot, cleared atomically
    # by mark_job_started (claim converts to active_job_count) or
    # mark_unhealthy. Valid only while young enough AND a live busy worker
    # is actually processing a job for this model — see state.py.
    starting_since: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Eviction claim: set by the evicting process just before `docker stop`,
    # cleared right after. TTL-expires by interpretation if the evictor dies
    # mid-stop (docker stop is idempotent, so a successor can just re-stop).
    evicting_since: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_job_finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_unhealthy_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
