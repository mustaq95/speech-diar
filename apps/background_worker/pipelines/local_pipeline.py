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
"""

import logging
import tempfile
from pathlib import Path

from packages.database.session import SessionLocal
from packages.shared_contracts.schemas import normalize_model_run
from packages.storage.s3_client import download_to

from apps.background_worker.models import REGISTRY
from apps.background_worker.pipelines._common import get_result_row, mark_done, mark_failed, mark_running

logger = logging.getLogger(__name__)


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
        mark_running(session, result)

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix) as scratch:
            download_to(s3_key, Path(scratch.name))
            raw = model.runner.run(scratch.name)
            run = normalize_model_run(model.adapter.adapt(raw))
    except NotImplementedError:
        with SessionLocal() as session:
            result = get_result_row(session, audio_file_id, model_id)
            mark_failed(session, result, f"{model_id} has no runner implementation yet")
        return
    except Exception as exc:
        logger.exception("Local model %r failed on audio_file_id=%s", model_id, audio_file_id)
        with SessionLocal() as session:
            result = get_result_row(session, audio_file_id, model_id)
            mark_failed(session, result, str(exc))
        return

    with SessionLocal() as session:
        result = get_result_row(session, audio_file_id, model_id)
        mark_done(session, result, run.model_dump(by_alias=True, mode="json"))
