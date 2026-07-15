"""Audio upload endpoint.

`POST /upload` streams the audio straight into each requested lane's own
store (MinIO for local models, Azure Blob only when `azure-batch` is
requested), creates one `queued` `EvaluationResult` row per model, and
enqueues one RQ job per model. It returns immediately; the browser polls
`GET /evaluations/{audioFileId}` for real status and results.
"""

import hashlib
import io
import logging
import os
import shutil
import subprocess
import tempfile
import wave

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from apps.background_worker.lanes import split_by_lane
from apps.background_worker.models import REGISTRY
from apps.background_worker.queue_app import queue
from apps.background_worker.worker import run_model
from apps.backend_api.dependencies import get_current_user, get_db
from packages.database.models import AudioFile, EvaluationResult, User
from packages.shared_contracts.schemas import DiarizationModelRun, QueuedModel, UploadAck
from packages.storage import azure_blob, s3_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["upload"])


def _wav_duration_file(path: str) -> float | None:
    """Read the WAV header from a file on disk — no full read into memory.

    Some encoders (e.g. streamed/live-recorded WAV) write a placeholder or
    otherwise inaccurate `data` chunk size, which `wave.getnframes()` trusts
    blindly and can report a duration many times longer than the real audio.
    The frame count is capped by what could actually fit in the file's bytes
    to guard against that.
    """
    with open(path, "rb") as fh:
        try:
            with wave.open(fh, "rb") as wav:
                n_channels = wav.getnchannels()
                sampwidth = wav.getsampwidth()
                framerate = wav.getframerate()
                header_frames = wav.getnframes()
                data_start = fh.tell()  # wave.open() leaves the position at the start of the PCM payload
        except (wave.Error, EOFError):
            return None
    if not framerate or not n_channels or not sampwidth:
        return None
    bytes_per_frame = n_channels * sampwidth
    max_frames_in_file = max(0, os.path.getsize(path) - data_start) // bytes_per_frame
    n_frames = min(header_frames, max_frames_in_file) if header_frames else max_frames_in_file
    return n_frames / framerate if n_frames else None


def _is_canonical_wav(content: bytes) -> bool:
    """Fast path: already 16-bit PCM, 16 kHz, mono — the canonical form we
    store — so skip ffmpeg."""
    buffer = io.BytesIO(content)
    try:
        with wave.open(buffer, "rb") as wav:
            return (
                wav.getsampwidth() == 2
                and wav.getcomptype() == "NONE"
                and wav.getframerate() == 16000
                and wav.getnchannels() == 1
            )
    except (wave.Error, EOFError):
        return False


def _transcode_to_wav_file(content: bytes) -> str | None:
    """Decode any ffmpeg-supported format (MP3, M4A, FLAC, ...) to a 16 kHz mono
    16-bit PCM WAV on disk, returning the temp file's path (the caller deletes it).

    Returns None if ffmpeg is missing or can't decode the bytes (corrupt / non-audio).

    Downmixed to 16 kHz mono: every diarizer resamples to this internally, so model
    output is unchanged, and it keeps a multi-hour file to a few hundred MB on disk
    instead of multiple GB held in memory.

    ffmpeg must write to a seekable file, not a pipe: on a non-seekable pipe it can't
    backfill the RIFF/data chunk sizes and emits a placeholder max size, which the
    frontend waveform parser and the stdlib-`wave` runners would misread. Input is
    still streamed via stdin.
    """
    if shutil.which("ffmpeg") is None:
        return None
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", "pipe:0", "-vn", "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", path],
            input=content, capture_output=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        os.unlink(path)
        return None
    if proc.returncode != 0:
        os.unlink(path)
        return None
    return path


def _content_hash_file(path: str) -> str:
    """Content-address the audio bytes so re-uploading the same file (e.g. while
    repeatedly testing against Azure) reuses the blob already staged there
    instead of uploading it again. Streams the file in chunks so a multi-hour WAV
    never lands in memory just to be hashed."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:32]


@router.post("", response_model=UploadAck, response_model_by_alias=True)
async def upload_audio(
    file: UploadFile,
    models: str = Query(description="Comma-separated model ids to run"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> UploadAck:
    # Validate the requested models first — a bad request must never pay for a
    # (possibly multi-hour) transcode.
    requested_ids = [m.strip() for m in models.split(",") if m.strip()]
    local_ids, azure_ids = split_by_lane(requested_ids)
    valid_ids = local_ids + azure_ids
    unknown_ids = [m for m in requested_ids if m not in valid_ids]
    if unknown_ids:
        raise HTTPException(status_code=422, detail=f"Unknown model id(s): {', '.join(unknown_ids)}")
    if not valid_ids:
        raise HTTPException(status_code=422, detail="No valid model ids requested")
    if azure_ids and not azure_blob.storage_configured():
        raise HTTPException(
            status_code=422,
            detail=(
                "azure-batch needs Azure Blob configured (AZURE_STORAGE_ACCOUNT_NAME + "
                "AZURE_STORAGE_ACCOUNT_KEY in .env)."
            ),
        )

    content = await file.read()
    # Canonicalize to a 16 kHz mono 16-bit PCM WAV on disk so every downstream consumer
    # sees one shape and a multi-hour file never lands wholesale in memory. The blocking
    # ffmpeg call runs off the event loop so it can't freeze the API.
    if _is_canonical_wav(content):
        fd, canonical_path = tempfile.mkstemp(suffix=".wav")
        with os.fdopen(fd, "wb") as fh:
            fh.write(content)
    else:
        canonical_path = await run_in_threadpool(_transcode_to_wav_file, content)
        if canonical_path is None:
            raise HTTPException(status_code=415, detail="Unsupported or unreadable audio file")

    try:
        duration_sec = _wav_duration_file(canonical_path)
        if duration_sec is None:
            raise HTTPException(status_code=415, detail="Could not read audio duration")

        # The stored object is always canonical .wav regardless of the uploaded filename
        # (which is kept only for display).
        audio_file = AudioFile(owner_id=current_user.id, filename=file.filename or "audio.wav", duration_sec=duration_sec)
        db.add(audio_file)
        db.commit()

        # Two independent writes, one per selected lane — never copied lane-to-lane.
        # Each streams the canonical file from disk; nothing re-buffers it in memory.
        if local_ids:
            key = f"audio/{audio_file.id}.wav"
            with open(canonical_path, "rb") as fh:
                s3_client.put_stream(fh, key)
            audio_file.s3_key = key
        if azure_ids:
            # Keyed by content hash, not audio_file.id: uploading the same file
            # again (e.g. repeated Azure test runs) reuses the existing blob
            # instead of re-uploading it.
            blob_key = f"uploads/{_content_hash_file(canonical_path)}.wav"
            if not azure_blob.blob_exists(blob_key):
                with open(canonical_path, "rb") as fh:
                    azure_blob.put_stream(fh, blob_key)
            audio_file.blob_key = blob_key
            audio_file.blob_url = azure_blob.blob_url(blob_key)
        db.commit()
    finally:
        if os.path.exists(canonical_path):
            os.unlink(canonical_path)

    queued: list[QueuedModel] = []
    for model_id in valid_ids:
        result = EvaluationResult(audio_file_id=audio_file.id, model_id=model_id, status="queued")
        db.add(result)
        queued.append(QueuedModel(id=model_id, status="queued"))
    db.commit()

    for model_id in valid_ids:
        queue.enqueue(run_model, audio_file.id, model_id)

    return UploadAck(audio_file_id=audio_file.id, models=queued)


@router.post("/url", response_model=DiarizationModelRun, response_model_by_alias=True)
def diarize_url(
    url: str = Query(description="http(s) audio URL that Azure can fetch"),
    model: str = Query(default="azure-batch", description="Model id to run (URL-based models only)"),
) -> DiarizationModelRun:
    """Synchronous smoke test for a URL-based model against a real Azure
    resource. Not used by the browser UI, which always goes through
    `POST /upload` + polling; this is a manual verification shortcut and
    does not touch the database or queue.
    """
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=422, detail="url must be an http(s) URL")
    model_entry = REGISTRY.get(model)
    if model_entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown model id {model!r}")
    try:
        return model_entry.process(url)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc) or f"{model} is not implemented") from exc
    except Exception as exc:
        logger.exception("diarize_url failed for model %r", model)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
