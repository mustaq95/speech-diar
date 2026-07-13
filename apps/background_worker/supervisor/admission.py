"""Admission control: cap how many managed models may be GPU-resident at
once, called synchronously from `pipelines/local_pipeline.py` around each
model's container call.

State store is Postgres, not Redis: this codebase has no existing
Redis-locking code (RQ is the only Redis consumer, purely as a job queue).
A `SELECT ... FOR UPDATE` held only across the "check cap, claim slot"
decision (milliseconds, committed immediately, never spanning a container
call) matches the short-lived-session discipline already established in
`pipelines/local_pipeline.py`.

Concurrency: the cap ("at most N models resident *total*") is a property of
the whole table, not one row, so the claim path locks every row for the
duration of the count-and-claim decision. A `SELECT ... FOR UPDATE` only
blocks on rows a query actually returns, so locking a filtered subset would
let two concurrent callers each see zero residents and both grant — the
whole-table lock was verified against real Postgres with concurrent
threads and must be re-verified whenever this locked section changes.

Residency itself is DERIVED, not stored: a model holds a slot iff it has a
valid cold-start claim, real active jobs, or its container is actually
running (one `docker ps`) — see `state.holds_slot`. The docker/RQ snapshots
are taken BEFORE the lock (both are read-only and the caller's own
busy-ness is stable for the job's whole duration); the claim writes that
make decisions visible to competitors all happen inside the locked
transaction, so races resolve toward under-admission, never over-admission
(a competitor who reads between our commit and our `docker stop` still
counts the victim as resident and simply waits one more bounce).
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from packages.config.settings import get_settings
from packages.database.models import ModelContainerState

from . import containers, state
from .registry import managed_container

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdmissionDecision:
    granted: bool
    evicted_model_id: str | None = None


def _as_aware_utc(value: datetime) -> datetime:
    """Same SQLite-vs-Postgres naive/aware guard as `state._as_aware_utc` —
    only the test suite hits this."""
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def try_acquire(session: Session, model_id: str) -> AdmissionDecision:
    """Attempt to claim a GPU slot for `model_id`. Must run inside its own
    short-lived `SessionLocal()` block — never held open across a container
    call. Returns `granted=False` if the cap is full and no evictable
    (idle-resident, unclaimed) model exists; the caller re-enqueues.

    Every grant takes (or refreshes) the `starting_since` claim, even when
    the container is already warm: the claim is what protects the model
    from eviction during the window between this grant and
    `mark_job_started` converting the claim into an active-job count. The
    calling RQ worker is itself busy on this model's job for that entire
    window, which is exactly what keeps the claim valid (see
    `state.starting_claim_valid`) — and what makes a claim from a *dead*
    prior attempt evaporate instead of wedging the slot.
    """
    settings = get_settings()
    busy_model_ids = state.live_busy_model_ids()
    running_names = containers.running_containers()
    now = datetime.now(timezone.utc)

    all_rows = {row.model_id: row for row in session.query(ModelContainerState).with_for_update().all()}
    target = all_rows.get(model_id)
    if target is None:
        raise ValueError(f"No ModelContainerState row for managed model {model_id!r} — was seed_rows() run?")

    if target.last_unhealthy_attempt_at is not None:
        backoff_until = _as_aware_utc(target.last_unhealthy_attempt_at) + timedelta(
            seconds=settings.unhealthy_retry_backoff_sec
        )
        if now < backoff_until:
            session.commit()
            return AdmissionDecision(granted=False)

    # Opportunistic repair: a worker killed mid-inference leaves
    # active_job_count stuck >0 with nothing running. Conservative by
    # design — if ANY live worker is busy on this model the count is left
    # alone (over-counting only ever under-admits), and it fully heals the
    # first time the model has no busy workers at all.
    for row in all_rows.values():
        if row.active_job_count > 0 and row.model_id not in busy_model_ids:
            logger.warning(
                "Repairing orphaned active_job_count for %s (%d -> 0): no live busy worker on it",
                row.model_id,
                row.active_job_count,
            )
            row.active_job_count = 0

    target_holds = state.holds_slot(target, busy_model_ids, running_names, now)
    other_holders = [
        row
        for row in all_rows.values()
        if row.model_id != model_id and state.holds_slot(row, busy_model_ids, running_names, now)
    ]

    evicted_model_id: str | None = None
    if not target_holds and len(other_holders) >= settings.max_resident_models:
        # Victim = idle-resident only: container running, no cold-start
        # claim (never evict a model mid-cold-start — reproduced live once),
        # no real active jobs (never evict mid-inference — hard rule).
        victim = next(
            (
                row
                for row in other_holders
                if state.real_active_jobs(row, busy_model_ids) == 0
                and not state.starting_claim_valid(row, busy_model_ids, now)
            ),
            None,
        )
        if victim is None:
            session.commit()  # persist the opportunistic repair even on deny
            return AdmissionDecision(granted=False)
        # Claim the eviction and our own slot in ONE transaction, so no
        # competitor can slip between "victim freed" and "we claimed".
        victim.evicting_since = now
        evicted_model_id = victim.model_id

    target.starting_since = now
    session.commit()

    if evicted_model_id is not None:
        victim_container = managed_container(evicted_model_id)
        try:
            if victim_container is not None:
                containers.stop_container(victim_container.container_name)
        except Exception:
            # The victim's memory was NOT freed — proceeding with the grant
            # anyway would risk exactly the OOM this system exists to
            # prevent. Release our own claim and deny; the caller re-enqueues
            # and retries. The victim's eviction claim is deliberately left
            # in place (we can't know how far the stop got) to TTL-expire on
            # its own; the next acquirer re-stops idempotently.
            logger.exception(
                "Evicting %s to admit %s: docker stop failed; denying this attempt", evicted_model_id, model_id
            )
            unclaimed = session.get(ModelContainerState, model_id, with_for_update=True)
            if unclaimed is not None:
                unclaimed.starting_since = None
            session.commit()
            return AdmissionDecision(granted=False)
        # Clear the eviction claim in a follow-up transaction. If we die
        # before this line, the claim TTL-expires on its own and the next
        # acquirer simply re-stops the container (docker stop is idempotent).
        cleared = session.get(ModelContainerState, evicted_model_id, with_for_update=True)
        if cleared is not None:
            cleared.evicting_since = None
        session.commit()
        logger.info("Evicted idle %s to admit %s", evicted_model_id, model_id)

    return AdmissionDecision(granted=True, evicted_model_id=evicted_model_id)


def mark_job_started(session: Session, model_id: str) -> None:
    """Called once the container is healthy and the job is about to invoke
    the model — converts the cold-start claim into an active-job count in
    one transaction, so there is never a moment where the model holds
    neither protection."""
    row = session.get(ModelContainerState, model_id, with_for_update=True)
    if row is None:
        return
    row.starting_since = None
    row.active_job_count += 1
    session.commit()


def mark_unhealthy(session: Session, model_id: str, error: str) -> None:
    row = session.get(ModelContainerState, model_id, with_for_update=True)
    if row is None:
        return
    row.starting_since = None
    row.last_unhealthy_attempt_at = datetime.now(timezone.utc)
    row.last_error = error[:2048]
    session.commit()


def release(session: Session, model_id: str) -> None:
    """Called once a job finishes (success or failure) — decrements the
    active-job counter and stamps `last_job_finished_at` for the idle-unload
    sweep. Never stops the container here — that's the daemon's job, kept
    out of this latency-sensitive completion path."""
    row = session.get(ModelContainerState, model_id, with_for_update=True)
    if row is None:
        return
    row.active_job_count = max(0, row.active_job_count - 1)
    if row.active_job_count == 0:
        row.last_job_finished_at = datetime.now(timezone.utc)
    session.commit()
