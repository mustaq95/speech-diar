"""Azure lane: Blob only.

Given an `AudioFile.blob_key`, builds a read SAS URL and runs the
`azure-batch` runner directly against it (URL-based, no disk). Writes
status/result/timing to the model's own `EvaluationResult` row exactly like
the local lane.

This module NEVER touches MinIO — see `local_pipeline.py` for the primary
lane, which is entirely separate.

DB sessions are short-lived and never held across the Azure call — the rule
`local_pipeline.py` documents at its top. A batch job on real audio runs for
minutes, far past the 60 s `idle_in_transaction_session_timeout` backstop in
`packages/database/session.py`: holding a session open across it got the
connection killed mid-job, so `mark_done` raised and the `mark_failed` meant
to catch that raised too, stranding the row at "running" forever.
"""

import audioop
import io
import logging
import wave
from pathlib import PurePosixPath

from packages.database.session import SessionLocal
from packages.shared_contracts.schemas import normalize_model_run
from packages.storage import azure_blob
from packages.storage.azure_blob import read_sas_url

from apps.background_worker.models import REGISTRY
from apps.background_worker.pipelines._common import get_result_row, mark_done, mark_failed, mark_running

logger = logging.getLogger(__name__)


def _header_is_valid(content: bytes) -> bool:
    """True if the audio is already mono and its WAV header's declared frame
    count matches what's actually on disk. Some encoders (streamed/live
    recordings) write a placeholder `data` chunk size, and Azure Batch's
    diarization only accepts mono input -- both are reported as the same
    generic "InvalidData - the recordings URI contains invalid data"."""
    buffer = io.BytesIO(content)
    try:
        with wave.open(buffer, "rb") as wav:
            n_channels = wav.getnchannels()
            bytes_per_frame = n_channels * wav.getsampwidth()
            header_frames = wav.getnframes()
            data_start = buffer.tell()
    except (wave.Error, EOFError):
        return False
    if n_channels > 1 or not bytes_per_frame:
        return False
    max_frames_in_buffer = max(0, len(content) - data_start) // bytes_per_frame
    return header_frames == max_frames_in_buffer


def _repaired_wav(content: bytes) -> bytes:
    """Downmix to mono (Azure Batch diarization rejects stereo outright) and
    rewrite the WAV header with the real `RIFF`/`data` chunk sizes."""
    buffer = io.BytesIO(content)
    with wave.open(buffer, "rb") as wav:
        n_channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        framerate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if n_channels > 1:
        frames = audioop.tomono(frames, sampwidth, 0.5, 0.5)
        n_channels = 1
    out = io.BytesIO()
    with wave.open(out, "wb") as fixed:
        fixed.setnchannels(n_channels)
        fixed.setsampwidth(sampwidth)
        fixed.setframerate(framerate)
        fixed.writeframes(frames)
    return out.getvalue()


def _fixed_key(blob_key: str) -> str:
    path = PurePosixPath(blob_key)
    return str(path.with_name(f"{path.stem}-fixed{path.suffix}"))


def _sas_url_for_azure_batch(blob_key: str) -> str:
    """azure-batch fetches audio directly from a URL, so a bad WAV header on
    the blob itself is what Azure rejects -- there's no local scratch file
    in this lane to patch, unlike the local lane's per-job temp file. Repairs
    the blob once (skipping already-well-formed ones) into a derived key and
    SASes off that instead of the original."""
    fixed_key = _fixed_key(blob_key)
    if azure_blob.blob_exists(fixed_key):
        return read_sas_url(fixed_key)
    content = azure_blob.open_stream(blob_key).readall()
    if _header_is_valid(content):
        return read_sas_url(blob_key)
    azure_blob.put_stream(io.BytesIO(_repaired_wav(content)), fixed_key)
    return read_sas_url(fixed_key)


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

        blob_key = audio_file.blob_key
        mark_running(session, result)

    try:
        sas_url = _sas_url_for_azure_batch(blob_key)
        raw = model.runner.run(sas_url)
        run = normalize_model_run(model.adapter.adapt(raw))
    except Exception as exc:
        logger.exception("Azure model %r failed on audio_file_id=%s", model_id, audio_file_id)
        with SessionLocal() as session:
            result = get_result_row(session, audio_file_id, model_id)
            # The row can be deleted while the job runs — this lane's window is
            # up to azure_batch_job_timeout_sec (1800s), the widest there is.
            # Same "dropping stale job" outcome as the lookup above, not a crash
            # inside the exception handler.
            if result is not None:
                mark_failed(session, result, str(exc))
        return

    with SessionLocal() as session:
        result = get_result_row(session, audio_file_id, model_id)
        if result is not None:
            # Prefer Azure's own reported job duration; fall back to worker wall-clock.
            processing_ms = model.adapter.processing_ms(raw)
            mark_done(session, result, run.model_dump(by_alias=True, mode="json"), processing_ms=processing_ms)
