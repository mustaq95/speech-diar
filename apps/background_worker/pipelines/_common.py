"""Shared helpers for both pipelines.

Deliberately has NO storage imports, so importing this module can never
blur the local/Azure lane boundary — each pipeline reaches its own store
directly.
"""

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from packages.database.models import EvaluationResult


def get_result_row(session: Session, audio_file_id: int, model_id: str) -> EvaluationResult | None:
    """None if the row is gone (e.g. an orphaned/stale job outlived its
    EvaluationResult) — callers log and return instead of crashing the job."""
    return session.query(EvaluationResult).filter_by(audio_file_id=audio_file_id, model_id=model_id).one_or_none()


def mark_running(session: Session, result: EvaluationResult) -> None:
    result.status = "running"
    result.started_at = datetime.now(timezone.utc)
    session.commit()


def mark_loading(session: Session, result: EvaluationResult) -> None:
    """Local-lane, GPU-supervisor-managed models only: status becomes
    "running" (the contract's ModelStatus is unchanged — the UI's existing
    running/queued/done/failed vocabulary still applies) but `started_at`
    stays unset. `loading_started_at` marks the start of the cold-start
    wait, kept OUT of `started_at`->`finished_at` (and therefore out of
    `processing_ms`) by construction. Call `mark_inference_started` once the
    container is confirmed healthy and inference is actually about to
    begin."""
    result.status = "running"
    result.loading_started_at = datetime.now(timezone.utc)
    session.commit()


def mark_inference_started(session: Session, result: EvaluationResult) -> None:
    result.started_at = datetime.now(timezone.utc)
    session.commit()


def _as_aware_utc(value: datetime) -> datetime:
    """SQLite (unlike Postgres TIMESTAMPTZ) hands back naive datetimes for
    any row fetched by a session that didn't itself set the column — e.g. a
    fresh session re-querying a row a different session wrote. Only the test
    suite hits this; assume naive means UTC, matching how these columns are
    always written."""
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def mark_done(session: Session, result: EvaluationResult, payload: dict, processing_ms: int | None = None) -> None:
    finished = datetime.now(timezone.utc)
    result.status = "done"
    result.finished_at = finished
    if processing_ms is not None:
        result.processing_ms = processing_ms
    elif result.started_at is not None:
        result.processing_ms = int((finished - _as_aware_utc(result.started_at)).total_seconds() * 1000)
    result.payload = payload
    session.commit()


def mark_failed(session: Session, result: EvaluationResult, error: str) -> None:
    finished = datetime.now(timezone.utc)
    result.status = "failed"
    result.error = error[:2048]
    result.finished_at = finished
    if result.started_at is not None:
        result.processing_ms = int((finished - _as_aware_utc(result.started_at)).total_seconds() * 1000)
    session.commit()
