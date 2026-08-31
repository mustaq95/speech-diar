"""RQ jobs for the live-speech transcript: ASR, then forced alignment.

Two jobs rather than one, chained (`run_asr` enqueues `run_align`), because
that buys three things for very little code:

  * the transcript TEXT reaches the UI as soon as ASR finishes, while word
    alignment is still running;
  * each stage is timed on its own (`asr_ms`, `align_ms`) — which is what the
    panel exists to show, and could not be separated honestly inside one job;
  * an alignment failure leaves a good ASR result intact instead of erasing it.

Chaining by having the first job enqueue the second is the pattern already used
in `../pipelines/local_pipeline.py`.

Both jobs take an engine id as their SECOND argument. That is not decoration:
`../supervisor/state.py` reads `job.args[1]` as the model id when deriving
which models a live worker is busy on. Nothing here is supervisor-managed
today, but keeping the codebase's one job-shape convention means putting an
engine under the residency cap later is a registry entry, not a debugging
session.

`asr_id` travels as a job argument rather than being re-resolved inside the
job: the mode chosen at enqueue time is what runs, and a later run in the other
mode must never silently change what an already-queued job executes. It is also
how each job finds its row — a recording holds one `TranscriptResult` per
engine, so both stages key on (audio_file_id, asr_id). For alignment that means
a THIRD argument, because `job.args[1]` is spoken for by the convention above.

DB sessions are short-lived and never held across an engine call — the rule
`../pipelines/local_pipeline.py` documents at its top. It matters more here:
an online ASR session on a long recording runs for ~16 minutes, far past the
60 s `idle_in_transaction_session_timeout` backstop in
`packages/database/session.py`.
"""

import logging
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from packages.config.settings import get_settings
from packages.database.models import TranscriptResult
from packages.database.session import SessionLocal
from packages.storage.s3_client import download_to

from . import ALIGNER_NAME, engine_for, transport_for
from .scoring import score_if_reference_exists
from .ctc_aligner import adapter as aligner_adapter
from .ctc_aligner import runner as aligner_runner

logger = logging.getLogger(__name__)

#: The alignment stage's own id, used only as `job.args[1]` for job-shape
#: consistency (see the module docstring). It is not a selectable engine.
ALIGNER_ID = "ctc-aligner"


def _score_and_log(session: Session, row: TranscriptResult, audio_file_id: int, asr_id: str) -> None:
    """Score the row if the recording has a reference, never failing the run.

    A scoring failure must not lose a transcript that took real time to produce:
    the text is the expensive part and is already on the row. The error rates are
    left null, which the UI shows as unscored rather than as zero.
    """
    from packages.database.models import AudioFile

    try:
        audio_file = session.get(AudioFile, audio_file_id)
        score_if_reference_exists(session, row, audio_file.duration_sec if audio_file else None)
    except Exception:  # noqa: BLE001 - the transcript is worth more than the score
        logger.exception("Scoring %s for audio_file_id=%s failed", asr_id, audio_file_id)


def get_transcript_row(session: Session, audio_file_id: int, asr_id: str) -> TranscriptResult | None:
    """The row THIS engine owns for this recording.

    Keyed on both: a recording holds one row per engine, so looking up by
    audio_file_id alone would pick an arbitrary one of them and let an online
    job overwrite an offline transcript.

    None when the row is gone — e.g. the recording was deleted while the job
    was queued. Callers log and return rather than crashing the job.
    """
    return (
        session.query(TranscriptResult)
        .filter_by(audio_file_id=audio_file_id, asr_id=asr_id)
        .one_or_none()
    )


def _mark_running(session: Session, row: TranscriptResult, stage: str) -> None:
    row.status = "running"
    row.stage = stage
    stamp = datetime.now(timezone.utc)
    if stage == "asr":
        row.asr_started_at = stamp
    else:
        row.align_started_at = stamp
    session.commit()


def _mark_failed(session: Session, row: TranscriptResult, error: str) -> None:
    row.status = "failed"
    row.error = error[:2048]
    session.commit()


def _locate_audio(session: Session, row: TranscriptResult) -> tuple[str, str] | None:
    """(s3_key, filename suffix) for the recording, or None if it has no
    local-lane audio. The transcript lane always reads from MinIO: an
    Azure-only recording has no bytes here to transcribe."""
    audio_file = row.audio_file
    if not audio_file.s3_key:
        _mark_failed(session, row, "Audio was not stored in MinIO, so it cannot be transcribed")
        return None
    return audio_file.s3_key, Path(audio_file.filename).suffix or ".wav"


def run_asr(audio_file_id: int, asr_id: str) -> None:
    """Stage 1: transcribe the recording with the engine chosen at enqueue time."""
    engine = engine_for(asr_id)
    with SessionLocal() as session:
        row = get_transcript_row(session, audio_file_id, asr_id)
        if row is None:
            logger.warning(
                "No TranscriptResult row for audio_file_id=%s asr=%s — dropping stale job",
                audio_file_id, asr_id,
            )
            return
        if engine is None:
            _mark_failed(session, row, f"Unknown ASR engine {asr_id!r}")
            return
        located = _locate_audio(session, row)
        if located is None:
            return
        s3_key, suffix = located
        _mark_running(session, row, "asr")

    started = time.monotonic()
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix) as scratch:
            download_to(s3_key, Path(scratch.name))
            # Two statements, not one: the engine's native output is persisted
            # verbatim beside the text its adapter reduces it to.
            raw = engine.run(scratch.name)
            text = engine.adapt(raw)
    except Exception as exc:
        logger.exception("ASR (%s) failed on audio_file_id=%s", asr_id, audio_file_id)
        with SessionLocal() as session:
            row = get_transcript_row(session, audio_file_id, asr_id)
            if row is not None:
                _mark_failed(session, row, str(exc))
        return
    asr_ms = int((time.monotonic() - started) * 1000)

    with SessionLocal() as session:
        row = get_transcript_row(session, audio_file_id, asr_id)
        if row is None:
            return
        row.text = text
        row.asr_ms = asr_ms
        # Stamped here rather than at enqueue so a row written by an older build
        # still gets labelled; the scorecard needs it to say what each figure means.
        row.transport = transport_for(asr_id)
        # What this BATCH run actually cut the recording at, recorded rather than
        # left null. A null here made the panel fall back to the LIVE chunk
        # slider's value, so a stored batch run displayed whatever interval the
        # operator's live control happened to be showing -- correct only while the
        # two settings coincide, which by default they do (both 3s). Move the
        # slider and the label started describing a cut that never happened.
        #
        # Only for a chunked transport: a "file" run made one whole-file call and
        # has no interval, so its columns stay null and the UI prints the reason.
        if row.transport == "chunks":
            row.chunk_interval_sec = get_settings().batch_segment_seconds
            if engine.batch_segments is not None:
                row.chunk_count = engine.batch_segments(raw)
        # Scored at the end of ASR, not after alignment: WER is a comparison of
        # TEXT, so it needs nothing the aligner produces, and computing it here
        # means a recording with a reference shows its error rates as soon as the
        # words exist instead of waiting on word timings it does not use.
        _score_and_log(session, row, audio_file_id, asr_id)
        # Above the empty-transcript branch below, so both commit paths keep it:
        # an empty transcript is the case where the native output is most worth
        # having, since it is what explains why the engine found nothing.
        row.raw_output = raw
        if not text:
            # Silence, or speech the engine found nothing in. A legitimate
            # result, not a failure — and there is nothing to align, so the run
            # completes here rather than queueing a no-op second stage. It is
            # still scored: an engine that returned nothing has a 100% error rate
            # against a non-empty reference, and that is a real result to show.
            row.status = "done"
            row.stage = None
            session.commit()
            logger.info("ASR (%s) returned an empty transcript for audio_file_id=%s", asr_id, audio_file_id)
            return
        session.commit()

    # Imported here, not at module scope: apps.backend_api imports this module
    # to enqueue jobs, and queue_app opens a Redis connection at import time.
    from apps.background_worker.queue_app import queue

    queue.enqueue(run_align, audio_file_id, ALIGNER_ID, asr_id)
    logger.info("ASR (%s) done in %dms for audio_file_id=%s; queued alignment", asr_id, asr_ms, audio_file_id)


def run_align(audio_file_id: int, aligner_id: str, asr_id: str) -> None:
    """Stage 2: give every word a start/end against the same audio.

    `aligner_id` exists only to keep `job.args[1]` populated (see the module
    docstring); there is one aligner and it is not selectable. `asr_id` is the
    real argument: it names which of the recording's per-engine rows this
    alignment belongs to, so aligning the offline transcript cannot write word
    timings onto the online one.
    """
    with SessionLocal() as session:
        row = get_transcript_row(session, audio_file_id, asr_id)
        if row is None:
            logger.warning(
                "No TranscriptResult row for audio_file_id=%s asr=%s — dropping stale job",
                audio_file_id, asr_id,
            )
            return
        text = row.text or ""
        located = _locate_audio(session, row)
        if located is None:
            return
        s3_key, suffix = located
        _mark_running(session, row, "align")

    started = time.monotonic()
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix) as scratch:
            download_to(s3_key, Path(scratch.name))
            words = aligner_adapter.adapt(aligner_runner.run(scratch.name, text))
    except Exception as exc:
        logger.exception("Alignment failed on audio_file_id=%s (asr=%s)", audio_file_id, asr_id)
        with SessionLocal() as session:
            row = get_transcript_row(session, audio_file_id, asr_id)
            if row is not None:
                # The ASR text and its timing stay on the row: a failed
                # alignment must not erase a transcript that was produced
                # successfully. The panel shows the text plus this error.
                _mark_failed(session, row, str(exc))
        return
    align_ms = int((time.monotonic() - started) * 1000)

    with SessionLocal() as session:
        row = get_transcript_row(session, audio_file_id, asr_id)
        if row is None:
            return
        row.words = [word.model_dump(by_alias=True, mode="json") for word in words]
        row.align_ms = align_ms
        row.status = "done"
        row.stage = None
        row.error = None
        session.commit()
    logger.info(
        "Alignment done in %dms for audio_file_id=%s (%d words, aligner=%s)",
        align_ms, audio_file_id, len(words), ALIGNER_NAME,
    )
