"""Evaluation endpoints — the frontend's data source.

Everything here is built from the database; nothing is fabricated. Segments
only appear once a model's `EvaluationResult.payload` is written by the
worker; until then, the model's real `status` (queued/running/failed) is
all the client gets.
"""

import logging
from collections.abc import Generator

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from apps.background_worker.lanes import lane_for
from apps.background_worker.models import REGISTRY
from apps.background_worker.pipelines.azure_pipeline import _fixed_key
from apps.background_worker.queue_app import queue
from apps.background_worker.worker import run_model
from apps.backend_api.dependencies import get_current_user, get_db
from packages.database.models import AudioFile, EvaluationResult, User
from packages.shared_contracts.schemas import DiarizationEvaluation, DiarizationModelRun, UploadTimingUpdate
from packages.storage import azure_blob, s3_client

router = APIRouter(prefix="/evaluations", tags=["evaluations"])
logger = logging.getLogger(__name__)

# A single mid-transfer read hiccup shouldn't truncate playback for a large
# file. Cap retries so a genuinely dead store still surfaces an error instead
# of looping forever.
_MAX_STREAM_RETRIES = 5


def _get_audio_file(db: Session, audio_file_id: int, current_user: User) -> AudioFile:
    audio_file = db.query(AudioFile).filter_by(id=audio_file_id, owner_id=current_user.id).one_or_none()
    if audio_file is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return audio_file


def _model_run_from_result(result: EvaluationResult) -> DiarizationModelRun:
    if result.payload:
        run = DiarizationModelRun.model_validate(result.payload)
    else:
        entry = REGISTRY.get(result.model_id)
        run = DiarizationModelRun(
            id=result.model_id,
            name=entry.adapter.name if entry else result.model_id,
            short=entry.adapter.short if entry else result.model_id,
            description=entry.adapter.description if entry else "",
        )
    return run.model_copy(
        update={
            "status": result.status,
            "error": result.error,
            "started_at": result.started_at,
            "finished_at": result.finished_at,
            "processing_ms": result.processing_ms,
        }
    )


def _build_evaluation(db: Session, audio_file: AudioFile) -> DiarizationEvaluation:
    results = db.query(EvaluationResult).filter_by(audio_file_id=audio_file.id).order_by(EvaluationResult.id).all()
    return DiarizationEvaluation(
        audio_file_id=audio_file.id,
        duration_sec=audio_file.duration_sec,
        upload_ms=audio_file.upload_ms,
        models=[_model_run_from_result(r) for r in results],
    )


@router.get("/{audio_file_id}", response_model=DiarizationEvaluation, response_model_by_alias=True)
def get_evaluation(
    audio_file_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DiarizationEvaluation:
    audio_file = _get_audio_file(db, audio_file_id, current_user)
    return _build_evaluation(db, audio_file)


@router.patch("/{audio_file_id}", response_model=DiarizationEvaluation, response_model_by_alias=True)
def update_upload_timing(
    audio_file_id: int,
    body: UploadTimingUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DiarizationEvaluation:
    """Persist the client-perceived upload time, measured once by the browser."""
    audio_file = _get_audio_file(db, audio_file_id, current_user)
    audio_file.upload_ms = body.upload_ms
    db.commit()
    return _build_evaluation(db, audio_file)


@router.post("/{audio_file_id}/models/{model_id}/retry", response_model=DiarizationEvaluation, response_model_by_alias=True)
def retry_model(
    audio_file_id: int,
    model_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DiarizationEvaluation:
    """Run or re-run one model against the audio already in storage, leaving
    every other model's run untouched.

    Handles a model that never ran on this recording (uploaded before the
    model existed, or toggled on afterward): if there's no row yet, one is
    created. An existing row is reset in place rather than duplicated -- the
    pipelines look a run up by (audio_file_id, model_id) with `.one_or_none()`
    and nothing in the schema enforces uniqueness, so a second row would break
    every subsequent job for that model. Either way the row ends up identical
    to one `upload.py` just created, which is what the worker expects.
    """
    audio_file = _get_audio_file(db, audio_file_id, current_user)
    if model_id not in REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model {model_id!r}")

    result = db.query(EvaluationResult).filter_by(audio_file_id=audio_file.id, model_id=model_id).one_or_none()
    if result is None:
        # New run for a model that never ran here. Don't enqueue a job the
        # pipeline can't feed: each lane reads the audio from its own store,
        # and an old upload only lives in one of them. (An existing row already
        # ran once, so its storage was valid -- no need to re-check on reset.)
        lane = lane_for(model_id)
        if lane == "local" and not audio_file.s3_key:
            raise HTTPException(status_code=400, detail=f"Model {model_id!r} needs a local-lane upload; this recording has none")
        if lane == "azure" and not (audio_file.blob_key or audio_file.blob_url):
            raise HTTPException(status_code=400, detail=f"Model {model_id!r} needs an Azure-lane upload; this recording has none")
        result = EvaluationResult(audio_file_id=audio_file.id, model_id=model_id, status="queued")
        db.add(result)
    else:
        if result.status in ("queued", "running"):
            raise HTTPException(status_code=409, detail=f"Model {model_id!r} is already {result.status}")
        result.status = "queued"
        result.error = None
        result.payload = None
        result.loading_started_at = None
        result.started_at = None
        result.finished_at = None
        result.processing_ms = None
    db.commit()

    queue.enqueue(run_model, audio_file.id, model_id)
    logger.info("Queued model %r for audio_file_id=%s", model_id, audio_file.id)
    return _build_evaluation(db, audio_file)


@router.delete("/{audio_file_id}", status_code=204)
def delete_evaluation(
    audio_file_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Response:
    """Delete a recording and every trace of it: its EvaluationResult rows, the
    AudioFile row, and the audio object(s) in whichever lane's store owns them.

    Nothing cascades in the schema, so the child rows are removed explicitly
    before the parent. Storage cleanup runs after the DB commit and is
    best-effort: once the rows are gone the recording is gone from the user's
    perspective, so a missing or unreachable object is logged, not surfaced as
    a failure. In-flight jobs for a just-deleted recording are tolerated by the
    pipelines (they drop a job whose row has vanished), so a queued/running
    model doesn't block deletion.
    """
    audio_file = _get_audio_file(db, audio_file_id, current_user)
    s3_key = audio_file.s3_key
    blob_key = audio_file.blob_key

    # The Azure lane is content-addressed and deduplicated: two recordings with
    # identical audio share one blob_key. Only delete the blob if no other
    # recording still points at it. The local s3_key is keyed by audio_file.id,
    # so it's never shared.
    blob_shared = bool(blob_key) and (
        db.query(AudioFile.id).filter(AudioFile.blob_key == blob_key, AudioFile.id != audio_file.id).first() is not None
    )

    db.query(EvaluationResult).filter_by(audio_file_id=audio_file.id).delete()
    db.delete(audio_file)
    db.commit()

    if s3_key:
        try:
            s3_client.delete_object(s3_key)
        except Exception:
            logger.warning("Failed to delete S3 object %s for deleted audio_file_id=%s", s3_key, audio_file_id, exc_info=True)
    if blob_key and not blob_shared:
        for key in (blob_key, _fixed_key(blob_key)):
            try:
                azure_blob.delete_blob(key)
            except Exception:
                logger.warning("Failed to delete blob %s for deleted audio_file_id=%s", key, audio_file_id, exc_info=True)

    logger.info("Deleted audio_file_id=%s", audio_file_id)
    return Response(status_code=204)


def _iter_s3_object(key: str, start: int = 0, length: int | None = None, chunk_size: int = 65536) -> Generator[bytes, None, None]:
    """Stream an S3/MinIO object (or a `start`/`length`-bounded slice of one
    for an HTTP Range request), resuming from the last delivered byte on a
    mid-transfer read failure instead of aborting the whole response."""
    delivered = 0
    attempts = 0
    stream = s3_client.open_stream(key, start=start)
    while length is None or delivered < length:
        want = chunk_size if length is None else min(chunk_size, length - delivered)
        try:
            chunk = stream.read(want)
        except Exception:
            attempts += 1
            if attempts > _MAX_STREAM_RETRIES:
                raise
            logger.warning(
                "S3 stream read failed for %s at offset %d (attempt %d/%d); resuming",
                key,
                start + delivered,
                attempts,
                _MAX_STREAM_RETRIES,
            )
            stream = s3_client.open_stream(key, start=start + delivered)
            continue
        if not chunk:
            break
        delivered += len(chunk)
        yield chunk


def _iter_blob(blob_key: str, start: int = 0, length: int | None = None) -> Generator[bytes, None, None]:
    """Stream an Azure blob (or a `start`/`length`-bounded slice of one for
    an HTTP Range request), resuming from the last delivered byte on a
    mid-transfer read failure instead of aborting the whole response."""
    delivered = 0
    attempts = 0
    chunks = iter(azure_blob.open_stream(blob_key, start=start).chunks())
    while length is None or delivered < length:
        try:
            chunk = next(chunks)
        except StopIteration:
            break
        except Exception:
            attempts += 1
            if attempts > _MAX_STREAM_RETRIES:
                raise
            logger.warning(
                "Blob stream read failed for %s at offset %d (attempt %d/%d); resuming",
                blob_key,
                start + delivered,
                attempts,
                _MAX_STREAM_RETRIES,
            )
            chunks = iter(azure_blob.open_stream(blob_key, start=start + delivered).chunks())
            continue
        if length is not None and delivered + len(chunk) > length:
            chunk = chunk[: length - delivered]
        delivered += len(chunk)
        yield chunk


def _parse_range(range_header: str | None, total: int) -> tuple[int, int] | None:
    """Parse a single `Range: bytes=start-end` header into inclusive byte
    offsets. Returns None (fall back to a full 200 response) when there's no
    header, it's malformed, or it can't be satisfied — this proxy doesn't
    need full RFC 7233 compliance (e.g. multi-range, 416 responses), just
    enough for browsers to seek into a long audio file."""
    if not range_header or not range_header.startswith("bytes="):
        return None
    spec = range_header[len("bytes=") :].split(",")[0].strip()
    start_str, _, end_str = spec.partition("-")
    try:
        if start_str == "":
            start = max(0, total - int(end_str))  # suffix range: bytes=-500 -> last 500 bytes
            end = total - 1
        else:
            start = int(start_str)
            end = int(end_str) if end_str else total - 1
    except ValueError:
        return None
    end = min(end, total - 1)
    if start < 0 or start > end:
        return None
    return start, end


@router.get("/{audio_file_id}/audio")
def stream_audio(
    audio_file_id: int,
    range_header: str | None = Header(default=None, alias="Range"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Stream the audio through the API from whichever lane's store owns
    it — one origin for playback and client-side waveform decoding, no
    CORS setup, and no lane's storage client is ever exposed to the browser.

    Honors an HTTP `Range` request so the browser can seek into a long file
    without re-downloading everything before the seek point, and so it never
    has to buffer the entire file just to play it.

    Closes the DB session as soon as the lookup is done, rather than leaving
    it open for `Depends(get_db)`'s usual request-scoped lifetime: a browser
    aborting an in-flight Range request (routine during timeline seeking)
    would otherwise hold the session open until FastAPI finishes sending the
    full StreamingResponse, leaking a pooled DB connection."""
    audio_file = _get_audio_file(db, audio_file_id, current_user)
    s3_key = audio_file.s3_key
    blob_key = audio_file.blob_key
    db.close()

    if s3_key:
        total = s3_client.head_object(s3_key)

        def _stream(start: int, length: int | None) -> Generator[bytes, None, None]:
            yield from _iter_s3_object(s3_key, start=start, length=length)
    elif blob_key:
        total = azure_blob.blob_size(blob_key)

        def _stream(start: int, length: int | None) -> Generator[bytes, None, None]:
            yield from _iter_blob(blob_key, start=start, length=length)
    else:
        raise HTTPException(status_code=404, detail="Audio not available for this evaluation")

    byte_range = _parse_range(range_header, total)
    if byte_range is None:
        headers = {"Accept-Ranges": "bytes", "Content-Length": str(total)}
        return StreamingResponse(_stream(0, None), media_type="audio/wav", headers=headers)

    start, end = byte_range
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Range": f"bytes {start}-{end}/{total}",
        "Content-Length": str(end - start + 1),
    }
    return StreamingResponse(_stream(start, end - start + 1), status_code=206, media_type="audio/wav", headers=headers)
