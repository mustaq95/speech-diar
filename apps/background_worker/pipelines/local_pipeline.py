"""Local-model lane: MinIO only.

Given an `AudioFile.s3_key` and one local model id, downloads the audio to a
context-managed scratch file, runs the model's runner + adapter, and writes
that model's status/result/timing to its own `EvaluationResult` row.

This module NEVER touches Azure Blob — see `azure_pipeline.py` for the Azure
lane, which is entirely separate and only ever handles `azure-batch`.

The DB session is deliberately not held open across `model.runner.run()`:
that call is GPU inference and can run for many minutes on real audio, well
past the `idle_in_transaction_session_timeout` Postgres backstop (see
`packages/database/session.py`) that guards against a session leaking a
pooled connection. Each phase (initial lookup, `mark_running`, the final
`mark_done`/`mark_failed`) gets its own short-lived session instead.

Models registered in `apps.background_worker.supervisor.registry` run
through the GPU-residency admission control below: a slot is claimed (or
the job is re-enqueued to wait for one) before the model's container is
contacted at all, and the container's health-wait happens outside any DB
session too, for the same reason `model.runner.run()` already is. `pyannote`
(and any future in-process, non-containerized model) has no registry entry
and takes the unmodified fast path — there is no container to admit it to.
"""

import logging
import tempfile
from datetime import timedelta
from pathlib import Path

from packages.database.session import SessionLocal
from packages.shared_contracts.schemas import normalize_model_run
from packages.storage.s3_client import download_to

from apps.background_worker.models import REGISTRY
from apps.background_worker.pipelines._common import (
    get_result_row,
    mark_done,
    mark_failed,
    mark_inference_started,
    mark_loading,
    mark_running,
)
from apps.background_worker.queue_app import queue
from apps.background_worker.supervisor import admission, containers
from apps.background_worker.supervisor.registry import is_managed, managed_container
from packages.config.settings import get_settings

logger = logging.getLogger(__name__)

# A model reporting a segment this many times past the audio's real duration
# is not a rounding artifact (those are sub-second): it's the engine
# hallucinating/generating past the end of the input, e.g. an autoregressive
# model that never emits an end-of-generation token near the audio's actual
# end. Such a run is marked failed rather than shown as valid data.
SEGMENT_OVERRUN_FACTOR = 3


def run_local_model(audio_file_id: int, model_id: str) -> None:
    with SessionLocal() as session:
        result = get_result_row(session, audio_file_id, model_id)
        if result is None:
            logger.warning("No EvaluationResult row for audio_file_id=%s model_id=%r — dropping stale job", audio_file_id, model_id)
            return
        audio_file = result.audio_file

        model = REGISTRY.get(model_id)
        if model is None:
            mark_failed(session, result, f"Unknown model id {model_id!r}")
            return
        if not audio_file.s3_key:
            mark_failed(session, result, "Audio was not stored in MinIO for the local lane")
            return

        s3_key = audio_file.s3_key
        suffix = Path(audio_file.filename).suffix or ".wav"
        duration_sec = audio_file.duration_sec

        if not is_managed(model_id):
            # Unmanaged (in-process) model, e.g. pyannote — no container, no
            # cap, no cold-start wait. Exactly today's behavior.
            mark_running(session, result)
        else:
            try:
                decision = admission.try_acquire(session, model_id)
            except Exception:
                # An unexpected admission failure (e.g. a docker CLI hiccup
                # while evicting a victim) must put the job back in the
                # bounce loop, never kill it: an escaped exception here made
                # RQ mark jobs failed with no retry, leaving their
                # evaluations stuck "Queued" forever — caught live. Bouncing
                # is correct even for a persistent fault (docker daemon
                # down): the job recovers the moment the fault clears, and
                # the row honestly reads "Queued" the whole time.
                logger.exception(
                    "Admission failed for %r (audio_file_id=%s); re-enqueueing instead of dying", model_id, audio_file_id
                )
                session.rollback()
                delay = get_settings().admission_requeue_delay_sec
                queue.enqueue_in(timedelta(seconds=delay), run_local_model, audio_file_id, model_id)
                return
            if not decision.granted:
                delay = get_settings().admission_requeue_delay_sec
                queue.enqueue_in(timedelta(seconds=delay), run_local_model, audio_file_id, model_id)
                return
            mark_loading(session, result)

    if is_managed(model_id):
        container_cfg = managed_container(model_id)
        try:
            containers.start_container(container_cfg.container_name)
            containers.ensure_ready(container_cfg.health_url, container_cfg.cold_start_timeout_sec)
        except Exception as exc:
            logger.exception("Container for %r failed to become ready for audio_file_id=%s", model_id, audio_file_id)
            with SessionLocal() as session:
                result = get_result_row(session, audio_file_id, model_id)
                mark_failed(session, result, str(exc))
                admission.mark_unhealthy(session, model_id, str(exc))
            return

        with SessionLocal() as session:
            result = get_result_row(session, audio_file_id, model_id)
            mark_inference_started(session, result)
            admission.mark_job_started(session, model_id)

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix) as scratch:
            download_to(s3_key, Path(scratch.name))
            raw = model.runner.run(scratch.name)
            run = normalize_model_run(model.adapter.adapt(raw))
        max_end = max((seg.e for seg in run.segs), default=0.0)
        if max_end > duration_sec * SEGMENT_OVERRUN_FACTOR:
            raise RuntimeError(
                f"{model_id} reported a segment ending at {max_end:.2f}s, far past the "
                f"audio's actual {duration_sec:.2f}s duration; model output "
                "discarded as unreliable rather than shown as a valid result"
            )
    except NotImplementedError:
        with SessionLocal() as session:
            result = get_result_row(session, audio_file_id, model_id)
            mark_failed(session, result, f"{model_id} has no runner implementation yet")
            if is_managed(model_id):
                admission.release(session, model_id)
        return
    except Exception as exc:
        logger.exception("Local model %r failed on audio_file_id=%s", model_id, audio_file_id)
        with SessionLocal() as session:
            result = get_result_row(session, audio_file_id, model_id)
            mark_failed(session, result, str(exc))
            if is_managed(model_id):
                admission.release(session, model_id)
        return

    with SessionLocal() as session:
        result = get_result_row(session, audio_file_id, model_id)
        mark_done(session, result, run.model_dump(by_alias=True, mode="json"))
        if is_managed(model_id):
            admission.release(session, model_id)
