#!/usr/bin/env bash
# Runs WORKER_CONCURRENCY independent RQ worker processes against the shared
# "diarization" queue, so models queued for the same upload actually run at
# the same time instead of one-at-a-time.
#
# Each process uses rq.SimpleWorker (no forking — required for RQ on macOS);
# the parallelism instead comes from running N separate OS processes, which
# works everywhere because Redis hands each queued job to exactly one of them.
#
# Usage (from repo root): ./scripts/run_workers.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

PIDFILE=".worker_pool.pids"

# Self-heal: if a previous run of THIS checkout died ungracefully (crash,
# kill -9) without reaching the trap below, any worker PIDs it recorded here
# might still be running. Only ever touches PIDs we ourselves wrote to this
# file — never searches the system-wide process table — so this is safe even
# when other users run their own copy of this repo on the same shared
# machine (e.g. DGX Spark): it can't see, let alone kill, their processes.
if [ -f "$PIDFILE" ]; then
  echo "Found $PIDFILE from a previous run — cleaning up any leftover workers..."
  while read -r old_pid; do
    kill -9 "$old_pid" 2>/dev/null || true
  done < "$PIDFILE"
  rm -f "$PIDFILE"
fi

echo "Flushing stale jobs and reconciling interrupted runs..."
uv run python -m apps.background_worker.flush

CONCURRENCY="${WORKER_CONCURRENCY:-1}"
echo "Starting $CONCURRENCY worker process(es) on the 'diarization' queue..."

pids=()
for _ in $(seq 1 "$CONCURRENCY"); do
  # --with-scheduler: a job denied a GPU slot (see
  # apps/background_worker/supervisor/admission.py) is re-enqueued via
  # queue.enqueue_in(), which lands in RQ's ScheduledJobRegistry, not the
  # live queue -- nothing promotes it back without this flag. RQ's scheduler
  # takes a Redis lock so only one of these N processes is ever active as
  # the scheduler; the rest still just dequeue and run jobs as before.
  uv run rq worker --worker-class rq.SimpleWorker --with-scheduler diarization &
  pids+=("$!")
done
printf '%s\n' "${pids[@]}" > "$PIDFILE"

trap 'echo "Stopping workers..."; kill "${pids[@]}" 2>/dev/null; wait; rm -f "$PIDFILE"' INT TERM
wait
