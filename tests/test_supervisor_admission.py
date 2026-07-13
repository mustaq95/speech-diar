"""Unit tests for apps/background_worker/supervisor/admission.py — cap
enforcement, never-evict-while-active, idle-eviction-under-contention, the
unhealthy-retry backoff, and read-time claim validity (TTL-expired or
worker-dead claims are treated as absent, never repaired).

Residency is derived, not stored: tests stub the two snapshots try_acquire
takes — `state.live_busy_model_ids()` (RQ) and
`containers.running_containers()` (docker) — and seed only claims/counters.

Honest limitation: the in-memory SQLite test DB parses `SELECT ... FOR
UPDATE` but does not actually lock (SQLite has no row-level locking), so the
*concurrent*-claims race (two worker processes claiming the last slot at the
same instant) cannot be genuinely exercised here — these tests verify the
decision logic sequentially. The real row-locking behavior needs a manual
check against Postgres (concurrent try_acquire calls, assert exactly one
wins) whenever the locked section changes.
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.orm import Session, sessionmaker

from apps.background_worker.supervisor import admission, state
from packages.database.models import ModelContainerState


def _seed_row(
    db_session_factory: sessionmaker[Session],
    model_id: str,
    *,
    active_job_count: int = 0,
    starting_since: datetime | None = None,
    evicting_since: datetime | None = None,
    last_unhealthy_attempt_at: datetime | None = None,
) -> None:
    with db_session_factory() as session:
        session.add(
            ModelContainerState(
                model_id=model_id,
                active_job_count=active_job_count,
                starting_since=starting_since,
                evicting_since=evicting_since,
                last_unhealthy_attempt_at=last_unhealthy_attempt_at,
            )
        )
        session.commit()


def _row(db_session_factory: sessionmaker[Session], model_id: str) -> ModelContainerState:
    with db_session_factory() as session:
        row = session.get(ModelContainerState, model_id)
        session.expunge(row)
        return row


@pytest.fixture(autouse=True)
def _cap_at_two(monkeypatch: pytest.MonkeyPatch) -> None:
    """max_resident_models=2 unless overridden. Patched into both modules
    that read settings (each holds its own `from ... import get_settings`
    reference)."""
    fake_settings = type(
        "S",
        (),
        {
            "max_resident_models": 2,
            "unhealthy_retry_backoff_sec": 120,
            "supervisor_stale_grace_sec": 60,
            "container_stop_grace_sec": 30,
        },
    )()
    monkeypatch.setattr(admission, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(state, "get_settings", lambda: fake_settings)


@pytest.fixture(autouse=True)
def _no_real_docker_or_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default snapshots: nothing busy, nothing running. Individual tests
    override via _snapshots(). Guarantees no test here ever touches a real
    docker CLI or Redis."""
    monkeypatch.setattr(state, "live_busy_model_ids", lambda: set())
    monkeypatch.setattr(admission.containers, "running_containers", lambda: set())
    monkeypatch.setattr(state.containers, "running_containers", lambda: set())


def _snapshots(monkeypatch: pytest.MonkeyPatch, *, busy: set[str] = frozenset(), running: set[str] = frozenset()) -> None:
    monkeypatch.setattr(state, "live_busy_model_ids", lambda: set(busy))
    monkeypatch.setattr(admission.containers, "running_containers", lambda: set(running))
    monkeypatch.setattr(state.containers, "running_containers", lambda: set(running))


def _stub_stop(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    stopped: list[str] = []
    monkeypatch.setattr(admission.containers, "stop_container", lambda name: stopped.append(name))
    return stopped


def test_try_acquire_grants_immediately_under_cap(db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_stop(monkeypatch)
    _seed_row(db_session_factory, "nemo-clustering")

    with db_session_factory() as session:
        decision = admission.try_acquire(session, "nemo-clustering")

    assert decision.granted is True
    assert decision.evicted_model_id is None
    # Every grant takes the cold-start claim -- it's what protects the model
    # from eviction until mark_job_started converts it to an active count.
    assert _row(db_session_factory, "nemo-clustering").starting_since is not None


def test_try_acquire_grants_for_an_already_resident_model_without_evicting(
    db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    stopped = _stub_stop(monkeypatch)
    _snapshots(monkeypatch, running={"nemo-clustering"})
    _seed_row(db_session_factory, "nemo-clustering")

    with db_session_factory() as session:
        decision = admission.try_acquire(session, "nemo-clustering")

    assert decision.granted is True
    assert stopped == []  # already resident -- no eviction needed


def test_try_acquire_denies_when_cap_full_and_no_idle_victim(db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch) -> None:
    stopped = _stub_stop(monkeypatch)
    _snapshots(monkeypatch, busy={"nemo-clustering", "3d-speaker-clustering"}, running={"nemo-clustering", "3d-speaker-clustering"})
    _seed_row(db_session_factory, "nemo-clustering", active_job_count=1)
    _seed_row(db_session_factory, "3d-speaker-clustering", active_job_count=1)
    _seed_row(db_session_factory, "vibevoice")

    with db_session_factory() as session:
        decision = admission.try_acquire(session, "vibevoice")

    assert decision.granted is False
    assert stopped == []
    assert _row(db_session_factory, "vibevoice").starting_since is None  # never claimed


def test_try_acquire_evicts_idle_model_when_cap_full(db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch) -> None:
    """An idle-resident model (container running, zero jobs, no claims) is
    fair game for eviction under contention, even before its idle-unload
    timeout -- deliberate policy, not a bug."""
    stopped = _stub_stop(monkeypatch)
    _snapshots(monkeypatch, busy={"3d-speaker-clustering"}, running={"nemo-clustering", "3d-speaker-clustering"})
    _seed_row(db_session_factory, "nemo-clustering", active_job_count=0)  # idle-resident
    _seed_row(db_session_factory, "3d-speaker-clustering", active_job_count=1)
    _seed_row(db_session_factory, "vibevoice")

    with db_session_factory() as session:
        decision = admission.try_acquire(session, "vibevoice")

    assert decision.granted is True
    assert decision.evicted_model_id == "nemo-clustering"
    assert stopped == ["nemo-clustering"]
    victim = _row(db_session_factory, "nemo-clustering")
    assert victim.evicting_since is None  # cleared after the stop completed
    assert _row(db_session_factory, "vibevoice").starting_since is not None


def test_try_acquire_never_evicts_a_model_that_is_still_cold_starting(
    db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression scenario caught live under the old design: a multi-model
    upload evicted models mid-cold-start (`docker start` then `docker stop`
    within the same second). A valid cold-start claim (fresh timestamp +
    live busy worker on that model) must never be an eviction target."""
    stopped = _stub_stop(monkeypatch)
    now = datetime.now(timezone.utc)
    _snapshots(monkeypatch, busy={"nemo-clustering", "3d-speaker-clustering"})
    _seed_row(db_session_factory, "nemo-clustering", starting_since=now)
    _seed_row(db_session_factory, "3d-speaker-clustering", starting_since=now)
    _seed_row(db_session_factory, "vibevoice")

    with db_session_factory() as session:
        decision = admission.try_acquire(session, "vibevoice")

    assert decision.granted is False
    assert stopped == []
    for model_id in ("nemo-clustering", "3d-speaker-clustering"):
        assert _row(db_session_factory, model_id).starting_since is not None  # untouched


def test_try_acquire_never_evicts_a_model_with_an_active_job(db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch) -> None:
    """The hard requirement: real active jobs make a model untouchable, no
    matter how long it's been resident."""
    stopped = _stub_stop(monkeypatch)
    _snapshots(monkeypatch, busy={"nemo-clustering", "3d-speaker-clustering"}, running={"nemo-clustering", "3d-speaker-clustering"})
    _seed_row(db_session_factory, "nemo-clustering", active_job_count=1)
    _seed_row(db_session_factory, "3d-speaker-clustering", active_job_count=3)
    _seed_row(db_session_factory, "vibevoice")

    with db_session_factory() as session:
        decision = admission.try_acquire(session, "vibevoice")

    assert decision.granted is False
    assert stopped == []


def test_try_acquire_ignores_a_ttl_expired_starting_claim(db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch) -> None:
    """A claim older than cold_start_timeout + grace no longer holds a slot,
    even if some worker is (implausibly) still busy on that model -- the
    hard ceiling from the old deadline sweep, now evaluated at read time."""
    _stub_stop(monkeypatch)
    hours_ago = datetime.now(timezone.utc) - timedelta(hours=2)  # past every model's 1800s budget + grace
    _snapshots(monkeypatch, busy={"nemo-clustering", "3d-speaker-clustering"})
    _seed_row(db_session_factory, "nemo-clustering", starting_since=hours_ago)
    _seed_row(db_session_factory, "3d-speaker-clustering", starting_since=hours_ago)
    _seed_row(db_session_factory, "vibevoice")

    with db_session_factory() as session:
        decision = admission.try_acquire(session, "vibevoice")

    assert decision.granted is True  # both stale claims ignored, cap has room


def test_try_acquire_ignores_a_starting_claim_with_no_live_worker(
    db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression scenario caught live: a honcho restart killed the worker
    mid-VibeVoice-cold-start, and the orphaned claim blocked a GPU slot for
    the full 30-minute budget. With no live busy worker on the model, the
    claim is treated as absent immediately -- no sweep, no waiting."""
    _stub_stop(monkeypatch)
    just_now = datetime.now(timezone.utc) - timedelta(seconds=30)  # fresh, but its worker is gone
    _snapshots(monkeypatch, busy=set())
    _seed_row(db_session_factory, "vibevoice", starting_since=just_now)
    _seed_row(db_session_factory, "nemo-clustering", starting_since=just_now)
    _seed_row(db_session_factory, "3d-speaker-clustering")

    with db_session_factory() as session:
        decision = admission.try_acquire(session, "3d-speaker-clustering")

    assert decision.granted is True


def test_try_acquire_repairs_an_orphaned_active_job_count(db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker killed mid-inference leaves active_job_count stuck >0 with
    nothing running. try_acquire zeroes it opportunistically (no live busy
    worker on that model), inside the same locked transaction."""
    _stub_stop(monkeypatch)
    _snapshots(monkeypatch, busy=set())
    _seed_row(db_session_factory, "nemo-clustering", active_job_count=2)
    _seed_row(db_session_factory, "vibevoice")

    with db_session_factory() as session:
        decision = admission.try_acquire(session, "vibevoice")

    assert decision.granted is True
    assert _row(db_session_factory, "nemo-clustering").active_job_count == 0


def test_try_acquire_respects_unhealthy_retry_backoff(db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_stop(monkeypatch)
    recently_failed = datetime.now(timezone.utc) - timedelta(seconds=10)  # well inside the 120s backoff
    _seed_row(db_session_factory, "vibevoice", last_unhealthy_attempt_at=recently_failed)

    with db_session_factory() as session:
        decision = admission.try_acquire(session, "vibevoice")

    assert decision.granted is False


def test_try_acquire_retries_unhealthy_model_after_backoff_elapses(db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_stop(monkeypatch)
    long_ago = datetime.now(timezone.utc) - timedelta(seconds=300)  # past the 120s backoff
    _seed_row(db_session_factory, "vibevoice", last_unhealthy_attempt_at=long_ago)

    with db_session_factory() as session:
        decision = admission.try_acquire(session, "vibevoice")

    assert decision.granted is True


def test_try_acquire_unknown_model_id_raises(db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_stop(monkeypatch)
    with db_session_factory() as session, pytest.raises(ValueError, match="No ModelContainerState row"):
        admission.try_acquire(session, "not-a-managed-model")


def test_mark_job_started_converts_the_claim_into_an_active_count(db_session_factory: sessionmaker[Session]) -> None:
    """One transaction: claim cleared, count incremented -- there is never a
    moment where the model holds neither protection."""
    _seed_row(db_session_factory, "nemo-clustering", starting_since=datetime.now(timezone.utc))

    with db_session_factory() as session:
        admission.mark_job_started(session, "nemo-clustering")

    row = _row(db_session_factory, "nemo-clustering")
    assert row.starting_since is None
    assert row.active_job_count == 1


def test_release_decrements_and_stamps_finish_time_only_when_no_jobs_remain(db_session_factory: sessionmaker[Session]) -> None:
    _seed_row(db_session_factory, "nemo-clustering", active_job_count=2)

    with db_session_factory() as session:
        admission.release(session, "nemo-clustering")

    row = _row(db_session_factory, "nemo-clustering")
    assert row.active_job_count == 1
    assert row.last_job_finished_at is None  # one job still running

    with db_session_factory() as session:
        admission.release(session, "nemo-clustering")

    row = _row(db_session_factory, "nemo-clustering")
    assert row.active_job_count == 0
    assert row.last_job_finished_at is not None


def test_release_never_goes_negative(db_session_factory: sessionmaker[Session]) -> None:
    _seed_row(db_session_factory, "nemo-clustering", active_job_count=0)

    with db_session_factory() as session:
        admission.release(session, "nemo-clustering")

    assert _row(db_session_factory, "nemo-clustering").active_job_count == 0


def test_try_acquire_denies_and_releases_its_claim_when_the_victim_stop_fails(
    db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test, caught live: `docker stop` on a slow NIM/Triton
    victim raised (CLI timeout), the exception escaped, and the acquiring
    job died permanently. A failed stop means the victim's memory was NOT
    freed, so the grant must be withdrawn: release the acquirer's own
    claim, deny (caller bounces), and leave the victim's eviction claim to
    TTL-expire — the next acquirer re-stops idempotently."""

    def _slow_stop(name: str) -> None:
        raise TimeoutError(f"docker stop {name} timed out")

    monkeypatch.setattr(admission.containers, "stop_container", _slow_stop)
    # running_containers() returns CONTAINER names -- nim-sortformer-str's
    # container is "parakeet-nim-str" (see registry.py).
    _snapshots(monkeypatch, busy={"3d-speaker-clustering"}, running={"parakeet-nim-str", "3d-speaker-clustering"})
    _seed_row(db_session_factory, "nim-sortformer-str", active_job_count=0)  # idle-resident victim
    _seed_row(db_session_factory, "3d-speaker-clustering", active_job_count=1)
    _seed_row(db_session_factory, "vibevoice")

    with db_session_factory() as session:
        decision = admission.try_acquire(session, "vibevoice")

    assert decision.granted is False
    assert _row(db_session_factory, "vibevoice").starting_since is None  # own claim released
    assert _row(db_session_factory, "nim-sortformer-str").evicting_since is not None  # left to TTL-expire


def test_mark_unhealthy_records_error_and_clears_the_claim(db_session_factory: sessionmaker[Session]) -> None:
    _seed_row(db_session_factory, "vibevoice", starting_since=datetime.now(timezone.utc))

    with db_session_factory() as session:
        admission.mark_unhealthy(session, "vibevoice", "CUDA out of memory")

    row = _row(db_session_factory, "vibevoice")
    assert row.starting_since is None
    assert row.last_error == "CUDA out of memory"
    assert row.last_unhealthy_attempt_at is not None
