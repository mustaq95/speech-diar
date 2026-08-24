"""GET /recordings — the recordings list, scoped to one evaluation surface.

The first collection route in this API, and it exists to replace a pattern rather
than to add one. The frontend used to assemble its Projects list from localStorage
and then issue one `GET /evaluations/{id}` PER ROW to refresh the counts, every time
the page was opened. That is N requests to render one page, and it meant the list
showed whatever a particular browser happened to remember rather than what exists.

Diarization and transcript recordings are listed separately because they are
different artifacts: one is scored by comparing speaker timelines, the other by
comparing text to a reference. `surface` is a stored column, so the split is a fact
about the row rather than a guess from its filename.

Counts come from aggregate subqueries. Nothing here touches `EvaluationResult.payload`
or `TranscriptResult.raw_output` — both are `deferred=True` and hold a whole model's
output, which a list has no use for.
"""

import logging

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from apps.backend_api.dependencies import get_current_user, get_db
from packages.database.models import (
    AudioFile,
    EvaluationResult,
    TranscriptReference,
    TranscriptResult,
    TtsResult,
    User,
)
from packages.shared_contracts.schemas import RecordingSurface, RecordingSummary

logger = logging.getLogger(__name__)
router = APIRouter(tags=["recordings"])


@router.get("/recordings", response_model=list[RecordingSummary], response_model_by_alias=True)
def list_recordings(
    surface: RecordingSurface = Query(
        "diarization", description="Which evaluation surface's recordings to list"
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[RecordingSummary]:
    """Every recording this user owns on one surface, newest first.

    Unpaginated, deliberately: nothing in this API pages, the working set is tens of
    recordings, and a limit nobody needs is worse than none. If this ever has to
    scroll, `created_at DESC` is already the right sort key to page on.

    An empty list is a real answer — a surface with no recordings yet — not a 404.
    """
    # Correlated aggregates rather than joins: a join over evaluation_results would
    # multiply the audio_files row by its model count and need a GROUP BY over every
    # selected column.
    model_count = (
        select(func.count(EvaluationResult.id))
        .where(EvaluationResult.audio_file_id == AudioFile.id)
        .correlate(AudioFile)
        .scalar_subquery()
    )
    done_count = (
        select(func.count(EvaluationResult.id))
        .where(EvaluationResult.audio_file_id == AudioFile.id, EvaluationResult.status == "done")
        .correlate(AudioFile)
        .scalar_subquery()
    )
    failed_count = (
        select(func.count(EvaluationResult.id))
        .where(EvaluationResult.audio_file_id == AudioFile.id, EvaluationResult.status == "failed")
        .correlate(AudioFile)
        .scalar_subquery()
    )
    engine_count = (
        select(func.count(TranscriptResult.id))
        .where(TranscriptResult.audio_file_id == AudioFile.id)
        .correlate(AudioFile)
        .scalar_subquery()
    )
    tts_count = (
        select(func.count(TtsResult.id))
        .where(TtsResult.audio_file_id == AudioFile.id)
        .correlate(AudioFile)
        .scalar_subquery()
    )
    # MIN over a nullable column: NULL for an unscored run, so this is the best score
    # among the runs that HAVE one, and None when none do.
    best_wer = (
        select(func.min(TranscriptResult.wer))
        .where(TranscriptResult.audio_file_id == AudioFile.id)
        .correlate(AudioFile)
        .scalar_subquery()
    )
    has_reference = (
        select(func.count(TranscriptReference.id))
        .where(TranscriptReference.audio_file_id == AudioFile.id)
        .correlate(AudioFile)
        .scalar_subquery()
    )

    rows = db.execute(
        select(
            AudioFile.id,
            AudioFile.filename,
            AudioFile.duration_sec,
            AudioFile.created_at,
            AudioFile.surface,
            # Whether a recording exists. A transcript row is created when its script
            # is generated, before anything is recorded, and until then it has no
            # stored object. Selected rather than derived from duration, which is 0
            # for such a row because nothing was measured.
            AudioFile.s3_key,
            AudioFile.blob_key,
            model_count.label("model_count"),
            done_count.label("done_count"),
            failed_count.label("failed_count"),
            engine_count.label("engine_count"),
            best_wer.label("best_wer"),
            has_reference.label("has_reference"),
            tts_count.label("tts_count"),
        )
        .where(AudioFile.owner_id == current_user.id, AudioFile.surface == surface)
        .order_by(AudioFile.created_at.desc(), AudioFile.id.desc())
    ).all()

    # Speaker counts live inside each model's payload, so they cannot come from an
    # aggregate. Fetched in ONE query for the whole page rather than per row — the
    # thing this route exists to stop doing.
    speaker_counts = _speaker_counts(db, [row.id for row in rows]) if surface == "diarization" else {}

    return [
        RecordingSummary(
            audio_file_id=row.id,
            surface=row.surface or "diarization",
            filename=row.filename,
            duration_sec=row.duration_sec,
            created_at=row.created_at.isoformat() if row.created_at else "",
            has_audio=bool(row.s3_key or row.blob_key),
            model_count=row.model_count,
            speaker_count=speaker_counts.get(row.id, 0),
            done_count=row.done_count,
            failed_count=row.failed_count,
            engine_count=row.engine_count,
            # Scored means "there is a reference AND at least one run was scored
            # against it" — a reference alone does not make the numbers exist.
            scored=bool(row.has_reference) and row.best_wer is not None,
            best_wer=row.best_wer,
            tts_count=row.tts_count,
        )
        for row in rows
    ]


def _speaker_counts(db: Session, audio_file_ids: list[int]) -> dict[int, int]:
    """Highest speaker count any model found, per recording.

    `num_spk` is a key inside the JSON payload, so it cannot be aggregated in SQL
    without a JSON path expression that would tie this route to Postgres. One query
    for the page, loading only the two columns needed, then the max in Python.
    """
    if not audio_file_ids:
        return {}
    rows = db.execute(
        select(EvaluationResult.audio_file_id, EvaluationResult.payload).where(
            EvaluationResult.audio_file_id.in_(audio_file_ids),
            EvaluationResult.payload.is_not(None),
        )
    ).all()
    counts: dict[int, int] = {}
    for audio_file_id, payload in rows:
        num_spk = (payload or {}).get("numSpk") or 0
        if num_spk > counts.get(audio_file_id, 0):
            counts[audio_file_id] = num_spk
    return counts
