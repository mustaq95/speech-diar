"""RQ queue definition — shared by the API (enqueues) and the worker (consumes).

One job per model (see `apps.background_worker.worker.run_model`), so each
model's status updates independently and a slow Azure batch job never
blocks the local models.
"""

from redis import Redis
from rq import Queue

from packages.config.settings import get_settings

redis_conn = Redis.from_url(get_settings().redis_url)
queue = Queue("diarization", connection=redis_conn)
