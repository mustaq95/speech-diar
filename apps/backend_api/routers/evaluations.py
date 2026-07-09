"""Evaluation endpoints — the frontend's data source.

Everything here is built from the database; nothing is fabricated. Segments
only appear once a model's `EvaluationResult.payload` is written by the
worker; until then, the model's real `status` (queued/running/failed) is
all the client gets.
"""

from collections.abc import Generator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from apps.background_worker.models import REGISTRY
from apps.backend_api.dependencies import get_current_user, get_db
from packages.database.models import AudioFile, EvaluationResult, User
from packages.shared_contracts.schemas import DiarizationEvaluation, DiarizationModelRun, UploadTimingUpdate
from packages.storage import azure_blob, s3_client

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


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


def _iter_stream(stream, chunk_size: int = 65536) -> Generator[bytes, None, None]:
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        yield chunk


@router.get("/{audio_file_id}/audio")
def stream_audio(
    audio_file_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Stream the audio through the API from whichever lane's store owns
    it — one origin for playback and client-side waveform decoding, no
    CORS setup, and no lane's storage client is ever exposed to the browser."""
    audio_file = _get_audio_file(db, audio_file_id, current_user)
    if audio_file.s3_key:
        stream = s3_client.open_stream(audio_file.s3_key)
        return StreamingResponse(_iter_stream(stream), media_type="audio/wav")
    if audio_file.blob_key:
        downloader = azure_blob.open_stream(audio_file.blob_key)
        return StreamingResponse(downloader.chunks(), media_type="audio/wav")
    raise HTTPException(status_code=404, detail="Audio not available for this evaluation")
