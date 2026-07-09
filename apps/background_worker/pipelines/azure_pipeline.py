"""Azure lane: Blob only.

Given an `AudioFile.blob_key`, builds a read SAS URL and runs the
`azure-batch` runner directly against it (URL-based, no disk). Writes
status/result/timing to the model's own `EvaluationResult` row exactly like
the local lane.

This module NEVER touches MinIO — see `local_pipeline.py` for the primary
lane, which is entirely separate.
"""

import logging

from packages.database.session import SessionLocal
from packages.shared_contracts.schemas import normalize_model_run
from packages.storage.azure_blob import read_sas_url

from apps.background_worker.models import REGISTRY
from apps.background_worker.pipelines._common import get_result_row, mark_done, mark_failed, mark_running

logger = logging.getLogger(__name__)


def run_azure_model(audio_file_id: int, model_id: str) -> None:
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
        if not audio_file.blob_key:
            mark_failed(session, result, "Audio was not stored in Azure Blob for the azure lane")
            return

        mark_running(session, result)

        try:
            sas_url = read_sas_url(audio_file.blob_key)
            raw = model.runner.run(sas_url)
            run = normalize_model_run(model.adapter.adapt(raw))
        except Exception as exc:
            logger.exception("Azure model %r failed on audio_file_id=%s", model_id, audio_file_id)
            mark_failed(session, result, str(exc))
            return

        # Prefer Azure's own reported job duration; fall back to worker wall-clock.
        processing_ms = model.adapter.processing_ms(raw)
        mark_done(session, result, run.model_dump(by_alias=True, mode="json"), processing_ms=processing_ms)
