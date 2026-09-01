"""STT_PRESTART_CONTAINERS: pre-start the transcript surface's GPU containers
and keep them resident.

These tests exist because of a real deadlock measured on 2026-09-01: the
transcript path never starts a container, a local engine reports itself
unconfigured while its container is down, and so it can never be enqueued. The
container was started by hand and the supervisor stopped it about six seconds
later, twice.

The subtle half is the PIN. Pre-starting alone is useless: idle is measured from
`last_job_finished_at`, which for these models was 697 and 795 HOURS old, so the
idle test fires on the first sweep no matter how large IDLE_UNLOAD_TIMEOUT_SEC
is. `test_a_pinned_model_is_not_idle_unloaded` fails if the pin is removed.
"""

from datetime import datetime, timedelta, timezone

import pytest

from apps.background_worker.supervisor import daemon
from packages.config.settings import Settings, get_settings
from packages.database.models import ModelContainerState


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_pinning_is_inert_unless_the_flag_is_on(monkeypatch) -> None:
    """Default off. A feature that overrides GPU residency policy must not
    switch itself on."""
    monkeypatch.setattr(daemon, "get_settings", lambda: Settings(stt_prestart_containers=False))
    assert daemon.stt_pinned_model_ids() == frozenset()


def test_pinned_set_is_derived_from_the_asr_registry(monkeypatch) -> None:
    """Container-backed ASR engines only -- not every managed container, and not
    a hardcoded list that would silently miss a newly registered engine."""
    monkeypatch.setattr(daemon, "get_settings", lambda: Settings(stt_prestart_containers=True))
    pinned = daemon.stt_pinned_model_ids()
    assert pinned == {"moss-transcribe", "vibevoice"}


def test_prestart_does_nothing_when_the_flag_is_off(monkeypatch) -> None:
    started: list[str] = []
    monkeypatch.setattr(daemon, "get_settings", lambda: Settings(stt_prestart_containers=False))
    monkeypatch.setattr(daemon.containers, "start_container", lambda n: started.append(n))
    daemon.prestart_stt_containers()
    assert started == []


def test_prestart_starts_and_waits_for_each_pinned_container(monkeypatch) -> None:
    started: list[str] = []
    waited: list[str] = []
    monkeypatch.setattr(daemon, "get_settings", lambda: Settings(stt_prestart_containers=True))
    monkeypatch.setattr(daemon.containers, "running_containers", lambda: set())
    monkeypatch.setattr(daemon.containers, "start_container", lambda n: started.append(n))
    monkeypatch.setattr(daemon.containers, "ensure_ready", lambda url, t: waited.append(url))
    daemon.prestart_stt_containers()
    assert sorted(started) == ["moss-transcribe", "vibevoice"]
    # Started is not ready: each one is waited on before the daemon moves on, or
    # the API would be asked "configured?" while the model is still loading.
    assert len(waited) == 2


def test_prestart_skips_a_container_that_is_already_running(monkeypatch) -> None:
    """`docker start` on a running container is harmless, but health is still
    confirmed rather than assumed."""
    started: list[str] = []
    waited: list[str] = []
    monkeypatch.setattr(daemon, "get_settings", lambda: Settings(stt_prestart_containers=True))
    monkeypatch.setattr(daemon.containers, "running_containers",
                        lambda: {"moss-transcribe", "vibevoice"})
    monkeypatch.setattr(daemon.containers, "start_container", lambda n: started.append(n))
    monkeypatch.setattr(daemon.containers, "ensure_ready", lambda url, t: waited.append(url))
    daemon.prestart_stt_containers()
    assert started == []
    assert len(waited) == 2


def test_one_unhealthy_container_does_not_stop_the_daemon(monkeypatch) -> None:
    """The daemon's real job is the sweep loop. A model that will not come up
    keeps reporting itself unconfigured, which is honest, but must not take the
    supervisor down with it."""
    monkeypatch.setattr(daemon, "get_settings", lambda: Settings(stt_prestart_containers=True))
    monkeypatch.setattr(daemon.containers, "running_containers", lambda: set())
    monkeypatch.setattr(daemon.containers, "start_container", lambda n: None)

    def boom(url, timeout):
        raise RuntimeError("never became healthy")

    monkeypatch.setattr(daemon.containers, "ensure_ready", boom)
    daemon.prestart_stt_containers()  # must not raise


def test_a_pinned_model_is_not_idle_unloaded(monkeypatch, db_session) -> None:
    """The load-bearing one.

    A pinned model whose last job finished WEEKS ago -- the real state of both
    these models -- must survive the sweep. Without the pin the idle test fires
    immediately and stops the container seconds after pre-start, which is the
    exact behaviour observed live before this flag existed.
    """
    stopped: list[str] = []
    monkeypatch.setattr(daemon, "get_settings", lambda: Settings(stt_prestart_containers=True))
    monkeypatch.setattr(daemon, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_session, "close", lambda: None)
    monkeypatch.setattr(daemon.containers, "running_containers",
                        lambda: {"moss-transcribe", "vibevoice"})
    monkeypatch.setattr(daemon.containers, "stop_container", lambda n: stopped.append(n))
    monkeypatch.setattr(daemon.state, "live_busy_model_ids", lambda: set())
    monkeypatch.setattr(daemon.state, "queued_counts", lambda: {})

    weeks_ago = datetime.now(timezone.utc) - timedelta(hours=700)
    for model_id in ("moss-transcribe", "vibevoice"):
        row = db_session.get(ModelContainerState, model_id)
        if row is None:
            row = ModelContainerState(model_id=model_id)
            db_session.add(row)
        row.last_job_finished_at = weeks_ago
        row.starting_since = None
        row.evicting_since = None
    db_session.commit()

    daemon.sweep_once()
    assert stopped == [], f"pinned models were idle-unloaded anyway: {stopped}"


def test_an_unpinned_model_is_still_idle_unloaded(monkeypatch, db_session) -> None:
    """The pin must be narrow. With the flag ON, a model that is NOT a
    container-backed ASR engine still obeys the idle timeout, or this feature
    would quietly disable idle-unload for the whole GPU."""
    stopped: list[str] = []
    monkeypatch.setattr(daemon, "get_settings", lambda: Settings(stt_prestart_containers=True))
    monkeypatch.setattr(daemon, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_session, "close", lambda: None)
    monkeypatch.setattr(daemon.containers, "running_containers", lambda: {"diarizen"})
    monkeypatch.setattr(daemon.containers, "stop_container", lambda n: stopped.append(n))
    monkeypatch.setattr(daemon.state, "live_busy_model_ids", lambda: set())
    monkeypatch.setattr(daemon.state, "queued_counts", lambda: {})

    row = db_session.get(ModelContainerState, "diarizen")
    if row is None:
        row = ModelContainerState(model_id="diarizen")
        db_session.add(row)
    row.last_job_finished_at = datetime.now(timezone.utc) - timedelta(hours=700)
    row.starting_since = None
    row.evicting_since = None
    db_session.commit()

    daemon.sweep_once()
    assert stopped == ["diarizen"]
