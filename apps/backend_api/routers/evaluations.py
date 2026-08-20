"""Evaluation endpoints — the frontend's data source.

Everything here is built from the database; nothing is fabricated. Segments
only appear once a model's `EvaluationResult.payload` is written by the
worker; until then, the model's real `status` (queued/running/failed) is
all the client gets.
"""

import logging
from collections.abc import Generator

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from apps.background_worker.lanes import lane_for
from apps.background_worker.models import REGISTRY
from apps.background_worker.pipelines.azure_pipeline import _fixed_key
from apps.background_worker.queue_app import queue
from apps.background_worker.transcription import (
    resolve_asr_ids,
    transport_for,
    ALIGNER_NAME,
    DEFAULT_TRANSCRIPTION_MODE,
    asr_id_for_mode,
    engine_for,
)
from apps.background_worker.transcription.pipeline import get_transcript_row, run_asr
from apps.background_worker.worker import run_model
from apps.backend_api.dependencies import get_current_user, get_db
from packages.config.settings import get_settings
from packages.database.models import (
    AudioFile,
    EvaluationResult,
    TranscriptReference,
    TranscriptResult,
    User,
)
from packages.shared_contracts.schemas import (
    DiarizationEvaluation,
    DiarizationModelRun,
    ModelRawOutput,
    ReferenceSource,
    TranscriptionMode,
    TranscriptMetrics,
    TranscriptRawOutput,
    TranscriptReference as TranscriptReferenceContract,
    TranscriptRun,
    TranscriptWord,
    UploadTimingUpdate,
)
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
        result.raw_output = None
        result.loading_started_at = None
        result.started_at = None
        result.finished_at = None
        result.processing_ms = None
    db.commit()

    queue.enqueue(run_model, audio_file.id, model_id)
    logger.info("Queued model %r for audio_file_id=%s", model_id, audio_file.id)
    return _build_evaluation(db, audio_file)


@router.get("/{audio_file_id}/models/{model_id}/raw", response_model=ModelRawOutput, response_model_by_alias=True)
def get_model_raw_output(
    audio_file_id: int,
    model_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ModelRawOutput:
    """One model's native output on this recording, beside the adapted contract.

    What the engine actually emitted, verbatim: the transcript text, per-word
    timings, confidences and non-speech events that its adapter drops on the way
    to the unified contract. Served as an opaque blob — nothing here parses it,
    and the adapter remains the only code that understands a given shape.

    A separate endpoint from `GET /evaluations/{id}` for the same reason the
    transcript is one: that response is polled every pollIntervalMs while models
    are in flight, and azure-batch's raw JSON alone runs to megabytes.

    `rawOutput: null` with a 200 means the run exists but stored no raw output —
    it predates this column, or it failed. A 404 means no run at all.
    """
    audio_file = _get_audio_file(db, audio_file_id, current_user)
    if model_id not in REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model {model_id!r}")

    result = db.query(EvaluationResult).filter_by(audio_file_id=audio_file.id, model_id=model_id).one_or_none()
    if result is None:
        raise HTTPException(status_code=404, detail=f"Model {model_id!r} has not run on this recording")

    # Touching `raw_output` is what loads the deferred column; this route is the
    # only place that pays for it.
    return ModelRawOutput(
        model_id=model_id,
        status=result.status,
        raw_output=result.raw_output,
        run=_model_run_from_result(result),
    )


def _transcript_run(row: TranscriptResult) -> TranscriptRun:
    """Row -> contract. Display names and mode come from the engine that ACTUALLY
    ran (`row.asr_id`), never from the mode a new run would use — a later run in
    the other mode must not relabel a transcript the other engine produced."""
    engine = engine_for(row.asr_id)
    return TranscriptRun(
        audio_file_id=row.audio_file_id,
        status=row.status,
        stage=row.stage,
        asr_id=row.asr_id,
        # An id with no engine entry can only come from a row written by an
        # older/other build. Report it as offline-unknown rather than guessing
        # that audio left the host.
        mode=engine.mode if engine else "offline",
        asr_name=engine.name if engine else row.asr_id,
        aligner_name=ALIGNER_NAME,
        text=row.text or "",
        words=[TranscriptWord.model_validate(word) for word in (row.words or [])],
        asr_ms=row.asr_ms,
        align_ms=row.align_ms,
        error=row.error,
        source=row.source or "batch",
        # From the engine's registry entry, not the row: the transport is a fact
        # about the engine, and a row written before this column existed should
        # still be labelled correctly rather than showing blank.
        transport=row.transport or transport_for(row.asr_id),
        chunk_interval_sec=row.chunk_interval_sec,
        chunk_count=row.chunk_count,
        first_latency_ms=row.first_latency_ms,
        avg_latency_ms=row.avg_latency_ms,
        metrics=_transcript_metrics(row),
    )


def _reference_contract(row: TranscriptReference) -> TranscriptReferenceContract:
    """Row -> contract. `word_count` is MEASURED from the stored text, never
    carried over from whatever a request claimed it would be."""
    return TranscriptReferenceContract(
        audio_file_id=row.audio_file_id,
        source=row.source,
        text=row.text,
        word_count=len(row.text.split()),
        params=row.params,
    )


def _transcript_metrics(row: TranscriptResult) -> TranscriptMetrics | None:
    """Scores for this run, or None when it has not been scored.

    None means "no reference, or not finished" — never zero. A 0.0 WER is a
    perfect transcript and must not be how "unscored" renders.
    """
    if row.wer is None:
        return None
    return TranscriptMetrics(
        wer=row.wer,
        cer=row.cer or 0.0,
        wer_raw=row.wer_raw if row.wer_raw is not None else row.wer,
        cer_raw=row.cer_raw if row.cer_raw is not None else (row.cer or 0.0),
        ref_word_count=row.ref_word_count or 0,
        hyp_word_count=row.hyp_word_count or 0,
        sub_count=row.sub_count or 0,
        del_count=row.del_count or 0,
        ins_count=row.ins_count or 0,
        # Deliberately passed through as-is, including None: a real-time bound
        # transport has no measurable RTF, and 0.0 would read as "infinitely
        # fast" instead of "not applicable".
        rtf=row.rtf,
    )


@router.get("/{audio_file_id}/transcript", response_model=list[TranscriptRun], response_model_by_alias=True)
def get_transcripts(
    audio_file_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[TranscriptRun]:
    """Every live-speech transcript for one recording — at most one per engine.

    A separate endpoint from `GET /evaluations/{id}` on purpose: that response
    is polled every pollIntervalMs while models are in flight, and a long
    recording's word list is hundreds of KB that would ride along on every
    poll forever.

    A list, not one object, because the online and offline runs coexist and
    comparing them is the point. An empty list means no transcript was ever
    started for this recording (e.g. it predates the feature, or no ASR engine
    is configured) — the client renders that as an offer to run one, not as an
    error, which is why this is an empty collection rather than a 404.
    """
    audio_file = _get_audio_file(db, audio_file_id, current_user)
    rows = (
        db.query(TranscriptResult)
        .filter_by(audio_file_id=audio_file.id)
        .order_by(TranscriptResult.asr_id)
        .all()
    )
    return [_transcript_run(row) for row in rows]


def _queue_transcript(
    db: Session, audio_file: AudioFile, asr_id: str, settings
) -> TranscriptResult:
    """Reset (or create) one engine's row for this recording and enqueue its job.

    Keyed on (recording, engine) throughout, so queueing one engine never
    touches another's text, timings or scores — that side-by-side is the whole
    comparison. Raises the same 409/422 the single-engine route always did.
    """
    engine = engine_for(asr_id)
    if engine is None or not engine.configured(settings):
        raise HTTPException(
            status_code=422, detail=f"{asr_id!r} is not configured on this host"
        )

    row = db.query(TranscriptResult).filter_by(audio_file_id=audio_file.id, asr_id=asr_id).one_or_none()
    if row is None:
        row = TranscriptResult(audio_file_id=audio_file.id, asr_id=asr_id, status="queued")
        db.add(row)
    else:
        if row.status in ("queued", "running"):
            raise HTTPException(
                status_code=409, detail=f"{engine.name} transcript is already {row.status}"
            )
        row.status = "queued"
        row.stage = None
        row.error = None
        row.text = None
        row.words = None
        row.raw_output = None
        row.asr_ms = None
        row.align_ms = None
        row.asr_started_at = None
        row.align_started_at = None
        # A re-run invalidates the previous run's measurements and scores too.
        # Leaving stale numbers beside fresh text would show a WER computed
        # against a transcript that no longer exists.
        row.chunk_count = None
        row.chunk_interval_sec = None
        row.chunk_latencies_ms = None
        row.first_latency_ms = None
        row.avg_latency_ms = None
        row.rtf = None
        row.wer = None
        row.cer = None
        row.wer_raw = None
        row.cer_raw = None
        row.ref_word_count = None
        row.hyp_word_count = None
        row.sub_count = None
        row.del_count = None
        row.ins_count = None
        row.alignment = None
    row.source = "batch"
    row.transport = transport_for(asr_id)
    return row


@router.post("/{audio_file_id}/transcripts", response_model=list[TranscriptRun], response_model_by_alias=True)
def start_transcripts(
    audio_file_id: int,
    asr_ids: list[str] = Body(..., embed=True, alias="asrIds"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[TranscriptRun]:
    """Run several engines over stored audio — the transcript-evaluation fan-out.

    Plural sibling of `POST /{id}/transcript`, rather than that route learning to
    return either an object or a list depending on its body. One endpoint, one
    response shape: the singular route stays exactly what the Live Speech panel
    already calls, and this one is what the comparison surface calls.

    One job per engine, mirroring the platform's one-job-per-model rule, so a
    slow engine never gates a fast one and each row reaches done|failed on its
    own.

    All-or-nothing on validation: an unknown or unconfigured engine is rejected
    before ANY job is queued, so a typo cannot leave half the comparison running
    against a scorecard that will never fill.
    """
    audio_file = _get_audio_file(db, audio_file_id, current_user)
    if not audio_file.s3_key:
        raise HTTPException(
            status_code=400, detail="This recording has no local-lane audio, so it cannot be transcribed"
        )
    try:
        resolved = resolve_asr_ids(asr_ids)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"Unknown ASR engine {exc.args[0]!r}") from exc

    settings = get_settings()
    rows = [_queue_transcript(db, audio_file, asr_id, settings) for asr_id in resolved]
    db.commit()
    for asr_id in resolved:
        queue.enqueue(run_asr, audio_file.id, asr_id)
    logger.info("Queued transcripts (%s) for audio_file_id=%s", ",".join(resolved), audio_file.id)
    return [_transcript_run(row) for row in rows]


@router.post("/{audio_file_id}/transcript", response_model=TranscriptRun, response_model_by_alias=True)
def start_transcript(
    audio_file_id: int,
    mode: TranscriptionMode = Body(DEFAULT_TRANSCRIPTION_MODE, embed=True),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TranscriptRun:
    """Run or re-run the transcript for audio already in storage.

    Covers three cases with one path: a recording uploaded before this feature
    existed, a failed run worth retrying, and transcribing in the other mode.
    The mode is chosen by the caller (the Live Speech toggle); its engine is
    resolved HERE, at enqueue time, and stamped on the row.

    Everything below is keyed on (recording, engine), never on the recording
    alone. Running the other mode ADDS a row, leaving the first engine's text,
    words and timings intact — that side-by-side is what the panel compares.
    Only a re-run of the SAME engine resets a row, and the in-flight 409 guards
    only that engine, so offline can start while online is still streaming.
    Mirrors `retry_model` above, which is per model for the same reason.
    """
    audio_file = _get_audio_file(db, audio_file_id, current_user)
    if not audio_file.s3_key:
        raise HTTPException(
            status_code=400, detail="This recording has no local-lane audio, so it cannot be transcribed"
        )

    settings = get_settings()
    asr_id = asr_id_for_mode(mode)
    engine = engine_for(asr_id)
    if engine is None or not engine.configured(settings):
        # Kept as its own check so the message still names the MODE the caller
        # asked for; the shared helper only knows the engine id it was handed.
        raise HTTPException(
            status_code=422,
            detail=f"{mode} mode selects {asr_id!r}, which is not configured on this host",
        )

    row = _queue_transcript(db, audio_file, asr_id, settings)
    db.commit()

    queue.enqueue(run_asr, audio_file.id, asr_id)
    logger.info("Queued transcript (%s) for audio_file_id=%s", asr_id, audio_file.id)
    return _transcript_run(row)


@router.get("/{audio_file_id}/reference", response_model=TranscriptReferenceContract,
            response_model_by_alias=True)
def get_reference(
    audio_file_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TranscriptReferenceContract:
    """The ground truth this recording's engines are scored against.

    404 when there is none: without a reference there is no WER to report, and a
    fabricated empty reference would score every engine at 100% error.
    """
    audio_file = _get_audio_file(db, audio_file_id, current_user)
    row = db.query(TranscriptReference).filter_by(audio_file_id=audio_file.id).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="This recording has no reference transcript")
    return _reference_contract(row)


@router.put("/{audio_file_id}/reference", response_model=TranscriptReferenceContract,
            response_model_by_alias=True)
def put_reference(
    audio_file_id: int,
    text: str = Body(..., embed=True),
    source: ReferenceSource = Body("pasted", embed=True),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TranscriptReferenceContract:
    """Set or replace the reference for a recording that already exists.

    The read-aloud flow supplies its reference at finalize; this is the other
    entry point — scoring audio that was already here, against a transcript
    supplied by hand.

    Replacing the reference deliberately does NOT rescore existing runs here.
    Scores are computed where the run finishes, so silently recomputing them from
    a request would put two different code paths in charge of the same number.
    The affected rows' scores are cleared instead, and the surface offers a
    re-run — an empty score is honest, a stale one is not.
    """
    audio_file = _get_audio_file(db, audio_file_id, current_user)
    cleaned = text.strip()
    if not cleaned:
        raise HTTPException(status_code=422, detail="A reference transcript cannot be empty")

    row = db.query(TranscriptReference).filter_by(audio_file_id=audio_file.id).one_or_none()
    if row is None:
        row = TranscriptReference(audio_file_id=audio_file.id, source=source, text=cleaned)
        db.add(row)
    else:
        row.source = source
        row.text = cleaned
        row.params = None

    for result in db.query(TranscriptResult).filter_by(audio_file_id=audio_file.id).all():
        result.wer = None
        result.cer = None
        result.wer_raw = None
        result.cer_raw = None
        result.ref_word_count = None
        result.hyp_word_count = None
        result.sub_count = None
        result.del_count = None
        result.ins_count = None
        result.alignment = None
    db.commit()
    logger.info("Reference (%s, %d words) set for audio_file_id=%s",
                source, len(cleaned.split()), audio_file.id)
    return _reference_contract(row)


@router.get(
    "/{audio_file_id}/transcript/{asr_id}/raw",
    response_model=TranscriptRawOutput,
    response_model_by_alias=True,
)
def get_transcript_raw_output(
    audio_file_id: int,
    asr_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TranscriptRawOutput:
    """One ASR engine's native output for this recording, beside the transcript.

    `get_model_raw_output`'s counterpart for the transcription subsystem, with
    the same 200-with-null vs 404 distinction. The ASR engine's output only: the
    aligner's is not persisted, because `run.words` already carries its per-word
    timings.

    Keyed on the engine, not the recording — a recording holds one transcript per
    engine and the online/offline comparison is the point, so there is no single
    "the" raw output to return.
    """
    audio_file = _get_audio_file(db, audio_file_id, current_user)
    if engine_for(asr_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown ASR engine {asr_id!r}")

    row = get_transcript_row(db, audio_file.id, asr_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Engine {asr_id!r} has not transcribed this recording")

    return TranscriptRawOutput(
        asr_id=asr_id,
        status=row.status,
        raw_output=row.raw_output,
        run=_transcript_run(row),
    )


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
    db.query(TranscriptResult).filter_by(audio_file_id=audio_file.id).delete()
    # Before db.delete(audio_file): this row holds a FK to it, so leaving it
    # would abort the delete on Postgres (SQLite in tests does not enforce FKs
    # by default and would have let the orphan through).
    db.query(TranscriptReference).filter_by(audio_file_id=audio_file.id).delete()
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
