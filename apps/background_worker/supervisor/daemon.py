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


def sweep_once() -> None:
    """Stop containers idle past `idle_unload_timeout_sec`. The predicate is
    re-derived from Docker/RQ each sweep, so it self-resets the moment a new
    job arrives — no per-model timers. The eviction claim taken here is the
    same one `admission.try_acquire` uses, so a mid-stop crash self-expires
    identically (successor re-stops; docker stop is idempotent)."""
    settings = get_settings()
    now = datetime.now(timezone.utc)
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
