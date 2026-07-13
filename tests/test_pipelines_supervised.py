"""Integration tests for local_pipeline.py's GPU-supervisor-managed model
path — admission control, the loading/inference-started split, and release
on completion/failure. Uses the same FakeModel/`_seed` helpers as
test_pipelines.py but forces a model id into
apps.background_worker.supervisor.registry's managed set via monkeypatch,
since the real registry only recognizes the actual container-backed model
ids (nemo-clustering, vibevoice, ...).
"""

from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy.orm import Session, sessionmaker

from apps.background_worker.pipelines import local_pipeline
from apps.background_worker.supervisor.registry import ManagedContainer
from packages.database.models import AudioFile, EvaluationResult, ModelContainerState
from packages.shared_contracts.schemas import DiarizationModelRun, DiarizationSegment


class FakeModel:
    def __init__(self, *, error: Exception | None = None) -> None:
        self._error = error
        self.received_input: str | None = None
        self.runner = SimpleNamespace(run=self._run)
        self.adapter = SimpleNamespace(adapt=self._adapt, processing_ms=lambda raw: None)

    def _run(self, audio_input: str, params: dict[str, Any] | None = None) -> dict:
        self.received_input = audio_input
        if self._error:
            raise self._error
        return {"native": "payload"}

    def _adapt(self, raw: dict) -> DiarizationModelRun:
        return DiarizationModelRun(
            id="fake-managed", name="Fake Managed", short="Fake", description="test double",
            segs=[DiarizationSegment(spk=0, s=0.0, e=1.0)],
        )


def _seed(db_session_factory: sessionmaker[Session], *, model_id: str = "fake-managed") -> int:
    with db_session_factory() as session:
        audio_file = AudioFile(owner_id=1, filename="clip.wav", duration_sec=10.0, s3_key="audio/1.wav")
        session.add(audio_file)
        session.commit()
        session.add(EvaluationResult(audio_file_id=audio_file.id, model_id=model_id, status="queued"))
        session.add(ModelContainerState(model_id=model_id))
        session.commit()
        return audio_file.id


def _status(db_session_factory: sessionmaker[Session], audio_file_id: int, model_id: str) -> EvaluationResult:
    with db_session_factory() as session:
        row = session.query(EvaluationResult).filter_by(audio_file_id=audio_file_id, model_id=model_id).one()
        session.expunge(row)
        return row


def _container_row(db_session_factory: sessionmaker[Session], model_id: str) -> ModelContainerState:
    with db_session_factory() as session:
        row = session.get(ModelContainerState, model_id)
        session.expunge(row)
        return row


@pytest.fixture()
def _managed(monkeypatch: pytest.MonkeyPatch) -> ManagedContainer:
    """Registers "fake-managed" as a supervisor-managed model for the
    duration of one test, and stubs the container start/health-check calls
    plus admission's docker/RQ snapshots so nothing shells out to a real
    `docker` CLI or touches real Redis."""
    from apps.background_worker.supervisor import state

    cfg = ManagedContainer(
        model_id="fake-managed", container_name="fake-managed-container",
        health_url="http://localhost:9999/health/ready", cold_start_timeout_sec=30,
    )
    monkeypatch.setattr(local_pipeline, "is_managed", lambda model_id: model_id == "fake-managed")
    monkeypatch.setattr(local_pipeline, "managed_container", lambda model_id: cfg)
    monkeypatch.setattr(local_pipeline.containers, "start_container", lambda name: None)
    monkeypatch.setattr(local_pipeline.containers, "ensure_ready", lambda url, timeout_sec: None)
    monkeypatch.setattr(state, "live_busy_model_ids", lambda: {"fake-managed"})
    monkeypatch.setattr(local_pipeline.admission.containers, "running_containers", lambda: set())
    monkeypatch.setattr(state.containers, "running_containers", lambda: set())
    fake_settings = type(
        "S",
        (),
        {
            "admission_requeue_delay_sec": 10,
            "max_resident_models": 2,
            "unhealthy_retry_backoff_sec": 120,
            "supervisor_stale_grace_sec": 60,
            "container_stop_grace_sec": 30,
        },
    )()
    monkeypatch.setattr(local_pipeline, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(local_pipeline.admission, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(state, "get_settings", lambda: fake_settings)
    return cfg


def test_managed_model_success_stamps_loading_and_started_and_releases(
    db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch, _managed: ManagedContainer
) -> None:
    audio_file_id = _seed(db_session_factory)
    fake_model = FakeModel()

    monkeypatch.setattr(local_pipeline, "SessionLocal", db_session_factory)
    monkeypatch.setattr(local_pipeline, "REGISTRY", {"fake-managed": fake_model})
    monkeypatch.setattr(local_pipeline, "download_to", lambda key, path: None)

    local_pipeline.run_local_model(audio_file_id, "fake-managed")

    result = _status(db_session_factory, audio_file_id, "fake-managed")
    assert result.status == "done"
    assert result.loading_started_at is not None
    assert result.started_at is not None
    assert result.processing_ms is not None
    # loading_started_at must be <= started_at: cold-start wait precedes inference.
    assert result.loading_started_at <= result.started_at

    row = _container_row(db_session_factory, "fake-managed")
    assert row.starting_since is None  # claim converted by mark_job_started
    assert row.active_job_count == 0  # then released on completion
    assert row.last_job_finished_at is not None


def test_managed_model_denied_admission_re_enqueues_without_running(
    db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch, _managed: ManagedContainer
) -> None:
    audio_file_id = _seed(db_session_factory)
    fake_model = FakeModel()
    enqueued: list[tuple] = []

    monkeypatch.setattr(local_pipeline, "SessionLocal", db_session_factory)
    monkeypatch.setattr(local_pipeline, "REGISTRY", {"fake-managed": fake_model})
    monkeypatch.setattr(local_pipeline, "download_to", lambda key, path: pytest.fail("must not download when denied a slot"))
    monkeypatch.setattr(local_pipeline.admission, "try_acquire", lambda session, model_id: type("D", (), {"granted": False, "evicted_model_id": None})())
    monkeypatch.setattr(local_pipeline.queue, "enqueue_in", lambda delay, func, *args: enqueued.append(args))

    local_pipeline.run_local_model(audio_file_id, "fake-managed")

    assert fake_model.received_input is None
    assert enqueued == [(audio_file_id, "fake-managed")]
    result = _status(db_session_factory, audio_file_id, "fake-managed")
    assert result.status == "queued"  # never touched -- mark_loading was never called
    # Queue depth is derived from RQ at read time (state.queued_counts), so a
    # denied claim writes nothing here -- a prior design incremented a stored
    # counter on every denial, which double-counted one job across retries.


def test_managed_model_engine_failure_still_releases_the_slot(
    db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch, _managed: ManagedContainer
) -> None:
    audio_file_id = _seed(db_session_factory)
    fake_model = FakeModel(error=RuntimeError("boom"))

    monkeypatch.setattr(local_pipeline, "SessionLocal", db_session_factory)
    monkeypatch.setattr(local_pipeline, "REGISTRY", {"fake-managed": fake_model})
    monkeypatch.setattr(local_pipeline, "download_to", lambda key, path: None)

    local_pipeline.run_local_model(audio_file_id, "fake-managed")

    result = _status(db_session_factory, audio_file_id, "fake-managed")
    assert result.status == "failed"
    assert "boom" in result.error

    row = _container_row(db_session_factory, "fake-managed")
    assert row.active_job_count == 0  # released, not stuck holding the slot


def test_managed_model_container_never_ready_marks_failed_and_unhealthy(
    db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch, _managed: ManagedContainer
) -> None:
    from apps.background_worker.supervisor.containers import ContainerNotReadyError

    audio_file_id = _seed(db_session_factory)
    fake_model = FakeModel()

    monkeypatch.setattr(local_pipeline, "SessionLocal", db_session_factory)
    monkeypatch.setattr(local_pipeline, "REGISTRY", {"fake-managed": fake_model})
    monkeypatch.setattr(local_pipeline, "download_to", lambda key, path: pytest.fail("must not download without a healthy container"))

    def _never_ready(url: str, timeout_sec: int) -> None:
        raise ContainerNotReadyError("timed out")

    monkeypatch.setattr(local_pipeline.containers, "ensure_ready", _never_ready)

    local_pipeline.run_local_model(audio_file_id, "fake-managed")

    assert fake_model.received_input is None
    result = _status(db_session_factory, audio_file_id, "fake-managed")
    assert result.status == "failed"
    assert "timed out" in result.error

    row = _container_row(db_session_factory, "fake-managed")
    assert row.starting_since is None  # claim cleared, not left dangling
    assert row.last_unhealthy_attempt_at is not None
    assert "timed out" in row.last_error


def test_admission_exception_re_enqueues_the_job_instead_of_killing_it(
    db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch, _managed: ManagedContainer
) -> None:
    """Regression test, caught live: an exception escaping try_acquire (a
    docker CLI timeout while evicting a victim) made RQ mark the job failed
    with no retry, leaving the evaluation stuck "Queued" forever. Any
    unexpected admission failure must put the job back in the bounce loop,
    exactly like a denial."""
    audio_file_id = _seed(db_session_factory)
    fake_model = FakeModel()
    enqueued: list[tuple] = []

    monkeypatch.setattr(local_pipeline, "SessionLocal", db_session_factory)
    monkeypatch.setattr(local_pipeline, "REGISTRY", {"fake-managed": fake_model})
    monkeypatch.setattr(local_pipeline, "download_to", lambda key, path: pytest.fail("must not download after an admission failure"))

    def _boom(session, model_id):
        raise TimeoutError("docker stop timed out mid-eviction")

    monkeypatch.setattr(local_pipeline.admission, "try_acquire", _boom)
    monkeypatch.setattr(local_pipeline.queue, "enqueue_in", lambda delay, func, *args: enqueued.append(args))

    local_pipeline.run_local_model(audio_file_id, "fake-managed")  # must not raise

    assert fake_model.received_input is None
    assert enqueued == [(audio_file_id, "fake-managed")]
    result = _status(db_session_factory, audio_file_id, "fake-managed")
    assert result.status == "queued"  # honest: still waiting, not falsely failed
