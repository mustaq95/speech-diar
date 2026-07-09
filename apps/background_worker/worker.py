"""RQ task: dispatch one model's run to its correct storage lane.

Enqueued once PER MODEL (see `apps/backend_api/routers/upload.py`), so RQ
calls `run_model(audio_file_id, model_id)` in a fresh worker process for
each. `apps/background_worker/lanes.py` is the only place that decides
which pipeline — local (MinIO) or azure (Blob) — a model belongs to; this
function just dispatches, it never touches storage itself.

Run the worker(s) locally (from repo root):
  ./scripts/run_workers.sh

That starts `WORKER_CONCURRENCY` (see .env) independent `rq worker
--worker-class rq.SimpleWorker diarization` processes against the same
queue, so models queued for one upload run concurrently instead of one at
a time. A single `uv run rq worker --worker-class rq.SimpleWorker
diarization` also works, but only processes one job at a time.
"""

import logging

from apps.background_worker.lanes import lane_for
from apps.background_worker.pipelines.azure_pipeline import run_azure_model
from apps.background_worker.pipelines.local_pipeline import run_local_model
from packages.config.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def run_model(audio_file_id: int, model_id: str) -> None:
    lane = lane_for(model_id)
    if lane == "local":
        run_local_model(audio_file_id, model_id)
    elif lane == "azure":
        run_azure_model(audio_file_id, model_id)
    else:
        logger.warning("No lane mapped for model %r (audio_file_id=%s) — skipping", model_id, audio_file_id)
