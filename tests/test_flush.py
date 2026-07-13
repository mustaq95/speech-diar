"""Boot-time flush: drains Redis and fails interrupted runs so every
`honcho start` begins with no stale jobs for any model."""

from datetime import timedelta

import pytest
from rq import Queue
from rq.registry import ScheduledJobRegistry
from sqlalchemy.orm import Session, sessionmaker

from apps.background_worker import flush
from packages.database.models import AudioFile, EvaluationResult


@pytest.fixture()
def patched_flush(
    fake_queue: Queue, db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(flush, "queue", fake_queue)
    monkeypatch.setattr(flush, "SessionLocal", db_session_factory)
    monkeypatch.setattr(flush, "init_db", lambda: None)  # SQLite schema already built by the fixture


def test_flush_drains_live_and_scheduled_jobs(patched_flush: None, fake_queue: Queue) -> None:
    fake_queue.enqueue("builtins.print", 1)
    # Same mechanism as an admission-denied job's retry: enqueue_in lands in
    # the ScheduledJobRegistry, not the live queue.
    fake_queue.enqueue_in(timedelta(minutes=5), "builtins.print", 2)
    assert fake_queue.get_job_ids()
    assert ScheduledJobRegistry(queue=fake_queue).get_job_ids()

    flush.flush()

    assert fake_queue.get_job_ids() == []
    assert ScheduledJobRegistry(queue=fake_queue).get_job_ids() == []


def test_flush_fails_stale_rows_and_keeps_terminal_ones(
    patched_flush: None, db_session_factory: sessionmaker[Session]
) -> None:
    with db_session_factory() as session:
        audio_file = AudioFile(owner_id=1, filename="clip.wav", duration_sec=10.0, s3_key="audio/1.wav")
        session.add(audio_file)
        session.commit()
        session.add_all(
            [
                EvaluationResult(audio_file_id=audio_file.id, model_id="m-queued", status="queued"),
                EvaluationResult(audio_file_id=audio_file.id, model_id="m-running", status="running"),
                EvaluationResult(
                    audio_file_id=audio_file.id, model_id="m-done", status="done", payload={"segments": []}
                ),
                EvaluationResult(
                    audio_file_id=audio_file.id, model_id="m-failed", status="failed", error="original error"
                ),
            ]
        )
        session.commit()
        audio_file_id = audio_file.id

    flush.flush()

    with db_session_factory() as session:
        rows = {
            row.model_id: row
            for row in session.query(EvaluationResult).filter_by(audio_file_id=audio_file_id).all()
        }
        for model_id in ("m-queued", "m-running"):
            assert rows[model_id].status == "failed"
            assert rows[model_id].error == flush.STALE_ERROR
            assert rows[model_id].finished_at is not None
        assert rows["m-done"].status == "done"
        assert rows["m-done"].payload == {"segments": []}
        assert rows["m-failed"].error == "original error"
