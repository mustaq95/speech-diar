"""Unit tests for apps/background_worker/supervisor/daemon.py — which is now
idle-unload only. The four staleness-repair sweeps the old daemon carried are
gone by design: state is derived at read time (see state.py / the admission
tests), so there is no stored mirror left to drift and nothing to repair.

Everything external is stubbed (docker via `daemon.containers`, RQ via
`daemon.state` / `state.get_settings`) — no test here may touch real Docker
or Redis.
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.orm import Session, sessionmaker

from apps.background_worker.supervisor import daemon, state
from packages.database.models import ModelContainerState


def _seed_row(db_session_factory: sessionmaker[Session], model_id: str, **kwargs) -> None:
    with db_session_factory() as session:
        session.add(ModelContainerState(model_id=model_id, **kwargs))
        session.commit()


def _row(db_session_factory: sessionmaker[Session], model_id: str) -> ModelContainerState:
    with db_session_factory() as session:
        row = session.get(ModelContainerState, model_id)
        session.expunge(row)
        return row


@pytest.fixture(autouse=True)
def _wire_test_session(db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(daemon, "SessionLocal", db_session_factory)
    fake_settings = type(
        "S",
        (),
        {
            "idle_unload_timeout_sec": 600,
            "container_stop_grace_sec": 30,
            "supervisor_stale_grace_sec": 60,
            "unhealthy_retry_backoff_sec": 120,
        },
    )()
    monkeypatch.setattr(daemon, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(state, "get_settings", lambda: fake_settings)
    # Default snapshots: nothing busy, nothing queued, nothing running.
    monkeypatch.setattr(daemon.state, "live_busy_model_ids", lambda: set())
    monkeypatch.setattr(daemon.state, "queued_counts", lambda: {})
    monkeypatch.setattr(daemon.containers, "running_containers", lambda: set())


def _stub_stop(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    stopped: list[str] = []
    monkeypatch.setattr(daemon.containers, "stop_container", lambda name: stopped.append(name))
    return stopped


def test_sweep_stops_a_model_idle_past_its_timeout(db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch) -> None:
    stopped = _stub_stop(monkeypatch)
    monkeypatch.setattr(daemon.containers, "running_containers", lambda: {"nemo-clustering"})
    long_idle = datetime.now(timezone.utc) - timedelta(seconds=900)  # past the 600s timeout
    _seed_row(db_session_factory, "nemo-clustering", last_job_finished_at=long_idle)

    daemon.sweep_once()

    assert stopped == ["nemo-clustering"]
    row = _row(db_session_factory, "nemo-clustering")
    assert row.evicting_since is None  # claim cleared once the stop completed


def test_sweep_leaves_a_recently_finished_model_alone(db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch) -> None:
    stopped = _stub_stop(monkeypatch)
    monkeypatch.setattr(daemon.containers, "running_containers", lambda: {"nemo-clustering"})
    just_finished = datetime.now(timezone.utc) - timedelta(seconds=30)
    _seed_row(db_session_factory, "nemo-clustering", last_job_finished_at=just_finished)

    daemon.sweep_once()

    assert stopped == []


def test_sweep_never_stops_a_model_with_queued_jobs(db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch) -> None:
    """A model with jobs waiting for it is about to be needed — stopping it
    would force an immediate cold restart, the user's exact complaint."""
    stopped = _stub_stop(monkeypatch)
    monkeypatch.setattr(daemon.containers, "running_containers", lambda: {"nemo-clustering"})
    monkeypatch.setattr(daemon.state, "queued_counts", lambda: {"nemo-clustering": 2})
    long_idle = datetime.now(timezone.utc) - timedelta(seconds=900)
    _seed_row(db_session_factory, "nemo-clustering", last_job_finished_at=long_idle)

    daemon.sweep_once()

    assert stopped == []


def test_sweep_never_stops_a_model_with_real_active_jobs(db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch) -> None:
    stopped = _stub_stop(monkeypatch)
    monkeypatch.setattr(daemon.containers, "running_containers", lambda: {"nemo-clustering"})
    monkeypatch.setattr(daemon.state, "live_busy_model_ids", lambda: {"nemo-clustering"})
    long_idle = datetime.now(timezone.utc) - timedelta(seconds=900)
    _seed_row(db_session_factory, "nemo-clustering", active_job_count=1, last_job_finished_at=long_idle)

    daemon.sweep_once()

    assert stopped == []


def test_sweep_never_stops_a_model_that_is_cold_starting(db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch) -> None:
    stopped = _stub_stop(monkeypatch)
    monkeypatch.setattr(daemon.containers, "running_containers", lambda: {"nemo-clustering"})
    monkeypatch.setattr(daemon.state, "live_busy_model_ids", lambda: {"nemo-clustering"})
    long_idle = datetime.now(timezone.utc) - timedelta(seconds=900)
    _seed_row(
        db_session_factory,
        "nemo-clustering",
        starting_since=datetime.now(timezone.utc),
        last_job_finished_at=long_idle,
    )

    daemon.sweep_once()

    assert stopped == []


def test_sweep_ignores_containers_that_are_not_running(db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch) -> None:
    stopped = _stub_stop(monkeypatch)
    long_idle = datetime.now(timezone.utc) - timedelta(seconds=900)
    _seed_row(db_session_factory, "nemo-clustering", last_job_finished_at=long_idle)

    daemon.sweep_once()  # running_containers() is empty (autouse default)

    assert stopped == []


def test_sweep_leaves_an_operator_started_container_alone(db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch) -> None:
    """A running container that never finished a job through us has
    last_job_finished_at = None — only unload what demonstrably went idle
    at a known time."""
    stopped = _stub_stop(monkeypatch)
    monkeypatch.setattr(daemon.containers, "running_containers", lambda: {"nemo-clustering"})
    _seed_row(db_session_factory, "nemo-clustering", last_job_finished_at=None)

    daemon.sweep_once()

    assert stopped == []


def test_sweep_survives_a_failed_docker_stop(db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing stop leaves the eviction claim in place to TTL-expire on
    its own; the sweep must not crash and must not clear the claim as if
    the stop had succeeded."""

    def _boom(name: str) -> None:
        raise RuntimeError("docker daemon unreachable")

    monkeypatch.setattr(daemon.containers, "stop_container", _boom)
    monkeypatch.setattr(daemon.containers, "running_containers", lambda: {"nemo-clustering"})
    long_idle = datetime.now(timezone.utc) - timedelta(seconds=900)
    _seed_row(db_session_factory, "nemo-clustering", last_job_finished_at=long_idle)

    daemon.sweep_once()

    row = _row(db_session_factory, "nemo-clustering")
    assert row.evicting_since is not None  # left to TTL-expire, not falsely cleared
