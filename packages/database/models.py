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
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    processing_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    audio_file: Mapped[AudioFile] = relationship(back_populates="results")
