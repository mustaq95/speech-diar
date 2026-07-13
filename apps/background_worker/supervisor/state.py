"""Derive supervisor state from its owners at read time.

Postgres (`ModelContainerState`) stores only job accounting and in-flight
claims — everything else is derived here, when asked, from the system that
actually owns it: Docker owns "is the container running", RQ owns "what is
queued and who is working". An earlier design mirrored a full
lifecycle-state machine into Postgres; every live incident it produced was
mirror-vs-reality drift, each repaired by another daemon sweep. Derived
state cannot drift, and claims whose validity is *evaluated at read time*
(expired or worker-dead => treated as absent) cannot orphan, so no repair
sweeps exist anymore.

Claim-validity predicates here are pure functions over
(row, snapshots, now) so `admission.try_acquire` can apply them to rows it
holds FOR UPDATE locks on, and `derive_statuses` can apply the identical
logic for the read-only /models/status endpoint — one definition of
"holds a slot", two call sites.

Tests monkeypatch this module's `containers`, `Worker`, and `queue`
attributes (same pattern as the daemon tests) — nothing here may be
exercised against real Docker/Redis in the suite.
"""

import os
import socket
from datetime import datetime, timedelta, timezone

from rq import Worker
from rq.registry import ScheduledJobRegistry
from sqlalchemy.orm import Session

from apps.background_worker.queue_app import queue
from packages.config.settings import get_settings
from packages.database.models import ModelContainerState
from packages.shared_contracts.schemas import ModelContainerStatus

from . import containers
from .registry import managed_container


def get_row(session: Session, model_id: str) -> ModelContainerState | None:
    return session.get(ModelContainerState, model_id)


def snapshot_all(session: Session) -> list[ModelContainerState]:
    return session.query(ModelContainerState).all()


def _as_aware_utc(value: datetime) -> datetime:
    """SQLite hands back naive datetimes in tests; Postgres is always aware."""
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def live_busy_model_ids() -> set[str]:
    """model_ids some live, busy RQ worker is processing a job for right
    now.

    Liveness is checked against the worker's OS PID, not just its Redis
    registration: `rq.SimpleWorker` heartbeats its registration with
    `job.timeout + 60` seconds of TTL when it picks up a job (verified in
    rq 2.10's `SimpleWorker.get_heartbeat_ttl`), so a hard-killed worker
    would otherwise look "busy" for up to `queue_job_timeout_sec` (75+
    minutes) — long enough to keep a dead cold-start claim alive for its
    whole TTL. Every worker runs on this same host (single-DGX deployment,
    all processes under one honcho — a stated constraint of this design),
    so `os.kill(pid, 0)` gives instant, accurate liveness; the hostname
    guard keeps the check honest rather than assumed. A worker whose PID is
    gone is dead *now*, and any claim leaning on it evaporates on the very
    next read."""
    busy: set[str] = set()
    for worker in Worker.all(connection=queue.connection):
        if worker.get_state() != "busy":
            continue
        if worker.hostname == socket.gethostname() and worker.pid is not None:
            try:
                os.kill(int(worker.pid), 0)
            except ProcessLookupError:
                continue  # registration outlived the process — dead worker
            except PermissionError:
                pass  # exists but owned by another user — alive
        job_id = worker.get_current_job_id()
        if job_id is None:
            continue
        job = queue.fetch_job(job_id)
        # run_local_model(audio_file_id, model_id) puts model_id at args[1].
        if job is not None and len(job.args) > 1:
            busy.add(job.args[1])
    return busy


def queued_counts() -> dict[str, int]:
    """Jobs waiting for each managed model, counted straight from RQ's live
    queue + ScheduledJobRegistry (where a denied job's `enqueue_in` retry
    sits between bounces)."""
    counts: dict[str, int] = {}
    job_ids = set(ScheduledJobRegistry(queue=queue).get_job_ids()) | set(queue.get_job_ids())
    for job_id in job_ids:
        job = queue.fetch_job(job_id)
        if job is not None and len(job.args) > 1:
            counts[job.args[1]] = counts.get(job.args[1], 0) + 1
    return counts


def starting_claim_valid(row: ModelContainerState, busy_model_ids: set[str], now: datetime) -> bool:
    """A cold-start claim counts only while it is young enough AND someone
    is actually doing the cold-starting. The worker-liveness half is what
    frees a slot within one RQ heartbeat TTL when a worker dies mid
    cold-start (observed live: a honcho restart during a VibeVoice load
    would otherwise have blocked a slot for the full 30-minute budget)."""
    if row.starting_since is None:
        return False
    container_cfg = managed_container(row.model_id)
    if container_cfg is None:
        return False
    ttl = timedelta(seconds=container_cfg.cold_start_timeout_sec + get_settings().supervisor_stale_grace_sec)
    if now - _as_aware_utc(row.starting_since) >= ttl:
        return False
    return row.model_id in busy_model_ids


def evicting_claim_valid(row: ModelContainerState, now: datetime) -> bool:
    """An eviction claim self-expires by interpretation: if the evictor died
    between commit and `docker stop`, the claim ages out and a successor
    simply re-stops the container (idempotent)."""
    if row.evicting_since is None:
        return False
    settings = get_settings()
    ttl = timedelta(seconds=settings.container_stop_grace_sec + settings.supervisor_stale_grace_sec)
    return now - _as_aware_utc(row.evicting_since) < ttl


def real_active_jobs(row: ModelContainerState, busy_model_ids: set[str]) -> int:
    """`active_job_count` is only believed while a live busy worker is
    actually on a job for this model — a worker killed mid-inference leaves
    the counter stuck >0 with nothing running, and this read-time discount
    replaces what used to be a dedicated repair sweep."""
    if row.active_job_count > 0 and row.model_id not in busy_model_ids:
        return 0
    return row.active_job_count


def holds_slot(
    row: ModelContainerState,
    busy_model_ids: set[str],
    running_names: set[str],
    now: datetime,
) -> bool:
    """Does this model currently occupy one of the MAX_RESIDENT_MODELS GPU
    slots? Cold-starting, actively inferring, and idle-but-resident all
    count; a model mid-eviction no longer does."""
    if starting_claim_valid(row, busy_model_ids, now):
        return True
    if real_active_jobs(row, busy_model_ids) > 0:
        return True
    container_cfg = managed_container(row.model_id)
    if container_cfg is not None and container_cfg.container_name in running_names:
        return not evicting_claim_valid(row, now)
    return False


def _derive_state(
    row: ModelContainerState,
    busy_model_ids: set[str],
    running_names: set[str],
    now: datetime,
) -> str:
    if starting_claim_valid(row, busy_model_ids, now):
        return "starting"
    if real_active_jobs(row, busy_model_ids) > 0:
        return "in_use"
    if evicting_claim_valid(row, now):
        return "stopping"
    container_cfg = managed_container(row.model_id)
    if container_cfg is not None and container_cfg.container_name in running_names:
        return "ready"
    if row.last_unhealthy_attempt_at is not None:
        backoff = timedelta(seconds=get_settings().unhealthy_retry_backoff_sec)
        if now - _as_aware_utc(row.last_unhealthy_attempt_at) < backoff:
            return "unhealthy"
    return "unloaded"


def derive_statuses(rows: list[ModelContainerState]) -> list[ModelContainerStatus]:
    """The /models/status payload, derived fresh from Docker + RQ + the
    claim rows at the moment of the request — never a stored snapshot.

    Rows for models no longer in the managed set are skipped, not reported:
    a model that moved off-host (vibevoice with VIBEVOICE_BASEURL set) may
    leave its old row behind in Postgres, but it has no container lifecycle
    anymore — omitting it lets the frontend's existing catalog-minus-status
    logic render it as "In-process" instead of a fabricated "unloaded"."""
    now = datetime.now(timezone.utc)
    busy_model_ids = live_busy_model_ids()
    running_names = containers.running_containers()
    queued = queued_counts()
    return [
        ModelContainerStatus(
            model_id=row.model_id,
            state=_derive_state(row, busy_model_ids, running_names, now),
            active_job_count=real_active_jobs(row, busy_model_ids),
            queued_job_count=queued.get(row.model_id, 0),
            last_error=row.last_error,
        )
        for row in rows
        if managed_container(row.model_id) is not None
    ]
