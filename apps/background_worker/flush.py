"""Boot-time flush — run by scripts/run_workers.sh before any worker starts.

Every `honcho start` is a fresh start: whatever the previous session left
behind (jobs in the live queue, admission-denied jobs parked in RQ's
ScheduledJobRegistry, started jobs orphaned by a killed worker) is deleted,
and the matching EvaluationResult rows still in queued/running are marked
failed so the UI shows a terminal state instead of a run that will never
happen.

Ordering: Redis is drained before the DB is reconciled. Honcho starts web,
worker, and supervisor in parallel, so an upload accepted during the flush
window could have its row transiently marked failed — but its job survives
the drain and overwrites the row when it runs, so that order self-heals.
The reverse order could delete a fresh job while leaving its row queued
forever.
"""

import logging

from rq.registry import ScheduledJobRegistry, StartedJobRegistry

from apps.background_worker.pipelines._common import mark_failed
from apps.background_worker.queue_app import queue
from packages.database.models import EvaluationResult
from packages.database.session import SessionLocal, init_db

logger = logging.getLogger(__name__)

STALE_ERROR = "Flushed: worker pool restarted before this run finished"


def _drain_redis() -> int:
    """Delete every job in the live queue, ScheduledJobRegistry (admission
    bounces waiting for a GPU slot), and StartedJobRegistry (leftovers from
    workers that died mid-job). Finished/failed registries are history and
    stay untouched. Not queue.empty(): that only clears the live list, and
    its Lua script doesn't run on fakeredis."""
    scheduled = ScheduledJobRegistry(queue=queue)
    started = StartedJobRegistry(queue=queue)
    job_ids = set(queue.get_job_ids()) | set(scheduled.get_job_ids()) | set(started.get_job_ids())
    for job_id in job_ids:
        job = queue.fetch_job(job_id)
        if job is not None:
            job.delete()
        else:
            # Hash already expired; strip the raw queue/registry entries.
            queue.remove(job_id)
            scheduled.remove(job_id)
            started.remove(job_id)
    return len(job_ids)


def _fail_stale_results() -> int:
    with SessionLocal() as session:
        rows = session.query(EvaluationResult).filter(EvaluationResult.status.in_(("queued", "running"))).all()
        for row in rows:
            mark_failed(session, row, STALE_ERROR)
        return len(rows)


def flush() -> None:
    init_db()  # idempotent; tables may not exist yet on a first boot
    dropped = _drain_redis()
    failed = _fail_stale_results()
    logger.info("Flush complete: dropped %d stale job(s), failed %d interrupted run(s)", dropped, failed)


if __name__ == "__main__":
    from packages.config.logging import configure_logging

    configure_logging()
    flush()
