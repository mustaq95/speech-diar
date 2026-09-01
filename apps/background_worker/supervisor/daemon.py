"""Standalone idle-unload loop — the one genuine background job left.

Run as its own OS process (the `supervisor` line in `Procfile`, managed by
`honcho start` exactly like `web`/`worker`) — `rq.SimpleWorker` has no idle
hook and never forks, so this cannot be piggybacked onto the workers.

This used to hold four additional staleness-repair sweeps. They are gone by
design, not omission: supervisor state is now derived from its owners at
read time (`state.py`) and claims carry timestamps whose validity is
evaluated when read, so there is no stored mirror left to drift and nothing
to repair. Idle-unload remains because *someone* has to proactively issue
`docker stop` when a model has sat unused past its timeout — that's a
timer-driven action, not a state repair.
"""

import logging
import time
from datetime import datetime, timedelta, timezone

from packages.config.settings import get_settings
from packages.database.models import ModelContainerState
from packages.database.session import SessionLocal, init_db

from . import containers, state
from .registry import managed_container

logger = logging.getLogger(__name__)


def stt_pinned_model_ids() -> frozenset[str]:
    """Container-backed ASR engines to pre-start and keep resident, or empty.

    Derived from the ASR registry rather than hardcoded, so registering a new
    container-backed engine picks it up with no edit here. Empty unless
    STT_PRESTART_CONTAINERS is on, which is what keeps this whole feature inert
    by default.

    Imported lazily: `transcription/__init__` reaches back into this package for
    its health probe, and a module-level import here would close that loop.
    """
    if not get_settings().stt_prestart_containers:
        return frozenset()
    from apps.background_worker.transcription import ASR_ENGINES

    return frozenset(m for m in ASR_ENGINES if managed_container(m) is not None)


def prestart_stt_containers() -> None:
    """Start each pinned engine's container and wait for it to report healthy.

    Runs once, before the sweep loop, so the containers are warm before the API
    is asked whether those engines are configured.

    Deliberately does NOT go through `admission.try_acquire`: admission would
    apply the residency cap and vibevoice's exclusive-GPU rule and refuse to hold
    both at once, which is precisely what this flag is asking to override. See
    `stt_prestart_containers` in settings for what that costs.

    A container that will not come up is logged and skipped, never fatal: the
    daemon's real job is the sweep loop, and one unhealthy model must not stop it
    from running. That engine simply keeps reporting itself unconfigured, which
    is the honest state.
    """
    pinned = stt_pinned_model_ids()
    if not pinned:
        return
    logger.info("STT_PRESTART_CONTAINERS is on; pre-starting %d container(s): %s",
                len(pinned), ", ".join(sorted(pinned)))
    for model_id in sorted(pinned):
        cfg = managed_container(model_id)
        if cfg is None:
            continue
        try:
            if cfg.container_name not in containers.running_containers():
                containers.start_container(cfg.container_name)
            containers.ensure_ready(cfg.health_url, cfg.cold_start_timeout_sec)
            logger.info("Pre-started %s and it is healthy", model_id)
        except Exception:
            logger.exception(
                "Pre-start failed for %s; it will keep reporting itself unconfigured",
                model_id,
            )


def sweep_once() -> None:
    """Stop containers idle past `idle_unload_timeout_sec`. The predicate is
    re-derived from Docker/RQ each sweep, so it self-resets the moment a new
    job arrives — no per-model timers. The eviction claim taken here is the
    same one `admission.try_acquire` uses, so a mid-stop crash self-expires
    identically (successor re-stops; docker stop is idempotent)."""
    settings = get_settings()
    now = datetime.now(timezone.utc)
    pinned = stt_pinned_model_ids()
    busy_model_ids = state.live_busy_model_ids()
    running_names = containers.running_containers()
    queued = state.queued_counts()

    to_stop: list[tuple[str, str]] = []
    with SessionLocal() as session:
        rows = session.query(ModelContainerState).with_for_update().all()
        for row in rows:
            container_cfg = managed_container(row.model_id)
            if container_cfg is None or container_cfg.container_name not in running_names:
                continue
            # Pinned by STT_PRESTART_CONTAINERS. Without this the pre-start is
            # pointless: these models' `last_job_finished_at` is weeks old, so
            # the idle test below fires on the very first sweep and stops the
            # container seconds after it came up.
            if row.model_id in pinned:
                continue
            if state.starting_claim_valid(row, busy_model_ids, now):
                continue
            if state.evicting_claim_valid(row, now):
                continue
            if state.real_active_jobs(row, busy_model_ids) > 0:
                continue
            if queued.get(row.model_id, 0) > 0:
                continue
            # A running container that never finished a job through us (e.g.
            # started manually by an operator) is left alone — only unload
            # what demonstrably went idle at a known time.
            if row.last_job_finished_at is None:
                continue
            idle_for = now - state._as_aware_utc(row.last_job_finished_at)
            if idle_for < timedelta(seconds=settings.idle_unload_timeout_sec):
                continue
            row.evicting_since = now
            to_stop.append((row.model_id, container_cfg.container_name))
        session.commit()

    for model_id, container_name in to_stop:
        try:
            containers.stop_container(container_name)
        except Exception:
            logger.exception("Idle-unload: failed to stop %s (%s)", model_id, container_name)
            continue  # claim TTL-expires on its own; next sweep or acquirer retries
        with SessionLocal() as session:
            row = session.get(ModelContainerState, model_id, with_for_update=True)
            if row is not None:
                row.evicting_since = None
                session.commit()
        logger.info("Idle-unloaded %s", model_id)


def run_forever() -> None:
    init_db()  # creates tables + seeds model_container_state rows if missing (idempotent)
    settings = get_settings()
    logger.info("Supervisor daemon started, sweep interval %ss", settings.supervisor_sweep_interval_sec)
    prestart_stt_containers()
    while True:
        try:
            sweep_once()
        except Exception:
            logger.exception("Supervisor sweep failed")
        time.sleep(settings.supervisor_sweep_interval_sec)


if __name__ == "__main__":
    from packages.config.logging import configure_logging

    configure_logging()
    run_forever()
