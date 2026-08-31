"""Scoring one transcript against a recording's reference.

Lives here, in the transcription subsystem, because BOTH callers need it and
neither may import the other: the API finalizes live sessions, and the worker
finishes stored-audio runs. One implementation means a live number and a batch
number are produced by the same arithmetic, so any difference between them is a
difference in the engines rather than in the code that measured them.

Scores are written when a run finishes and then read. Nothing recomputes them on
the way out — a request handler that recalculated would put two code paths in
charge of one number, and they would eventually disagree.
"""

import logging

from sqlalchemy.orm import Session

from packages.database.models import TranscriptReference, TranscriptResult
from packages.metrics import wer as wer_metrics

logger = logging.getLogger(__name__)


def reference_text_for(session: Session, audio_file_id: int) -> str | None:
    """The recording's reference, or None when it has none.

    None means "cannot be scored", which is a real state: a recording with no
    reference shows measured timings and no error rates, rather than a WER
    computed against nothing.
    """
    row = (
        session.query(TranscriptReference)
        .filter_by(audio_file_id=audio_file_id)
        .one_or_none()
    )
    return row.text if row else None


def score_row(
    row: TranscriptResult, reference_text: str, audio_duration_sec: float | None = None
) -> None:
    """Compute and store one row's scores, in place.

    Stores normalized AND raw rates: the UI's normalization toggle is then a read,
    and normalization's own effect stays visible instead of being invisible
    preprocessing applied once and forgotten.
    """
    hypothesis = row.text or ""
    normalized = wer_metrics.score(reference_text, hypothesis, normalized=True)
    raw = wer_metrics.score(reference_text, hypothesis, normalized=False)

    row.wer = normalized.wer
    row.cer = normalized.cer
    row.wer_raw = raw.wer
    row.cer_raw = raw.cer
    row.ref_word_count = normalized.ref_words
    row.hyp_word_count = normalized.hyp_words
    row.sub_count = normalized.sub
    row.del_count = normalized.delete
    row.ins_count = normalized.ins
    row.alignment = [
        {"op": step.op, "ref": step.ref, "hyp": step.hyp, "hypIndex": step.hyp_index}
        for step in normalized.alignment
    ]
    row.rtf = _real_time_factor(row, audio_duration_sec)


def _real_time_factor(row: TranscriptResult, audio_duration_sec: float | None) -> float | None:
    """Processing time over audio duration, or None where it cannot be measured.

    None for a streaming transport, always. A real-time protocol consumes audio at
    1x by definition, so the quotient would be ~1.0 no matter how fast the model
    is and would say nothing about it. NULL is what makes the UI print
    "real-time bound" instead of a number that looks like a measurement.

    Every OTHER transport gets one. This tested `transport != "chunks"` while
    "chunks" and "stream" were the only two, which reads the same as excluding
    the stream and is not: adding "file" silently dropped an RTF that is not only
    measurable but is the single most meaningful speed figure that engine has
    (one call, whole recording, no waiting on a speaker). It rendered as "—",
    which is the blank this codebase is not allowed to print for something it
    did measure.
    """
    if row.transport == "stream" or not audio_duration_sec:
        return None
    # Prefer the summed per-chunk latencies (what was actually spent on inference)
    # over the run's wall clock, which on a live capture includes the time spent
    # waiting for someone to finish speaking. A file transport has no per-chunk
    # latencies, so it falls through to asr_ms -- which for it is the whole
    # measurement, not a fallback: one call, timed end to end.
    total_ms = sum(row.chunk_latencies_ms or []) or row.asr_ms
    if not total_ms:
        return None
    return (total_ms / 1000.0) / audio_duration_sec


def score_if_reference_exists(
    session: Session, row: TranscriptResult, audio_duration_sec: float | None = None
) -> bool:
    """Score `row` when its recording has a reference. Returns whether it did."""
    reference = reference_text_for(session, row.audio_file_id)
    if not reference:
        return False
    score_row(row, reference, audio_duration_sec)
    logger.info(
        "Scored %s for audio_file_id=%s: WER %.3f (%d ref words)",
        row.asr_id, row.audio_file_id, row.wer or 0.0, row.ref_word_count or 0,
    )
    return True
