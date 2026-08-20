"""Database engine, session factory, and dev bootstrap.

Dev stage uses `create_all`; when the schema changes, drop/recreate the
local database (no migrations tool this iteration — see the plan's
"Design decisions" section).
"""

import logging

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from packages.config.settings import get_settings
from packages.database.models import Base, ModelContainerState, User

logger = logging.getLogger(__name__)

DEV_USER_EMAIL = "dev@example.com"

engine = create_engine(
    get_settings().database_url,
    future=True,
    # The flip side of the timeout below: once Postgres kills a connection the
    # pool still hands it out, and the failure surfaces later as an
    # InternalError on whatever statement happens to run next (it surfaced as a
    # commit that had nothing to do with the session that actually leaked).
    # Costs one trivial round-trip per checkout.
    pool_pre_ping=True,
    # Backstop against a leaked/abandoned session (e.g. an aborted streaming
    # response) permanently starving the pool: Postgres kills it after 60s
    # of holding an open transaction instead of holding it forever.
    connect_args={"options": "-c idle_in_transaction_session_timeout=60000"},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """Create tables if missing and seed startup rows.

    Called once at both API startup and the supervisor daemon's startup
    (apps/background_worker/supervisor/daemon.py) — idempotent, safe to call
    from either or both. There is no real authentication yet — every
    request is attributed to this one seeded user (see dependencies.py).
    """
    Base.metadata.create_all(bind=engine)
    _ensure_added_columns()
    _backfill_recording_surface()
    with SessionLocal() as session:
        _seed_dev_user(session)
        _seed_model_container_state(session)


#: Columns added to already-deployed tables, as (table, column, type).
#:
#: `create_all` only ever CREATEs; it never ALTERs a table that already exists,
#: and this repo has no migration tool. So every column added after a table is
#: deployed has to be listed here, or the first read of that table on an older
#: database fails with UndefinedColumn.
#:
#: Append-only. Removing a line does not drop the column, it just stops
#: creating it on databases that never had it.
_ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("evaluation_results", "raw_output", "JSON"),
    # Which surface produced a recording. The DEFAULT is right for every row that
    # existed before the transcript surface did, but NOT for read-aloud rows created
    # between that surface shipping and this column existing -- see
    # _backfill_recording_surface below.
    ("audio_files", "surface", "VARCHAR(16) DEFAULT 'diarization'"),
    ("transcript_results", "raw_output", "JSON"),
    # The transcript-evaluation surface: how a transcript was produced, and how
    # it scored against the reference.
    ("transcript_results", "source", "VARCHAR(8) DEFAULT 'batch'"),
    ("transcript_results", "transport", "VARCHAR(16)"),
    ("transcript_results", "chunk_interval_sec", "DOUBLE PRECISION"),
    ("transcript_results", "chunk_count", "INTEGER"),
    ("transcript_results", "first_latency_ms", "INTEGER"),
    ("transcript_results", "avg_latency_ms", "INTEGER"),
    ("transcript_results", "chunk_latencies_ms", "JSON"),
    ("transcript_results", "rtf", "DOUBLE PRECISION"),
    ("transcript_results", "wer", "DOUBLE PRECISION"),
    ("transcript_results", "cer", "DOUBLE PRECISION"),
    ("transcript_results", "wer_raw", "DOUBLE PRECISION"),
    ("transcript_results", "cer_raw", "DOUBLE PRECISION"),
    ("transcript_results", "ref_word_count", "INTEGER"),
    ("transcript_results", "hyp_word_count", "INTEGER"),
    ("transcript_results", "sub_count", "INTEGER"),
    ("transcript_results", "del_count", "INTEGER"),
    ("transcript_results", "ins_count", "INTEGER"),
    ("transcript_results", "alignment", "JSON"),
)


#: Read-aloud recordings that predate `audio_files.surface`, which the column's
#: DEFAULT would file as diarization. Identified by what they have rather than by
#: filename: no diarization results, but transcripts and/or a reference.
#:
#: Verified unambiguous against real data before shipping -- no row had both
#: diarization results and a reference -- so nothing legitimate is reclassified.
#: Idempotent: the `surface = 'diarization'` guard means a second run is a no-op,
#: and it only ever writes a label, never audio, transcripts or scores.
_BACKFILL_RECORDING_SURFACE = """
UPDATE audio_files a SET surface = 'transcript'
 WHERE a.surface = 'diarization'
   AND NOT EXISTS (SELECT 1 FROM evaluation_results e WHERE e.audio_file_id = a.id)
   AND (EXISTS (SELECT 1 FROM transcript_results t WHERE t.audio_file_id = a.id)
     OR EXISTS (SELECT 1 FROM transcript_references r WHERE r.audio_file_id = a.id))
"""


def _backfill_recording_surface() -> None:
    """Label pre-existing read-aloud recordings, which the column default gets wrong.

    Postgres only, for the same reason as `_ensure_added_columns`: on SQLite the
    tables were just built from the current models and hold no legacy rows.
    """
    if engine.dialect.name != "postgresql":
        return
    with engine.begin() as conn:
        result = conn.execute(text(_BACKFILL_RECORDING_SURFACE))
        if result.rowcount:
            logger.info(
                "Labelled %d pre-existing recording(s) as transcript recordings", result.rowcount
            )


def _ensure_added_columns() -> None:
    """Add every column in `_ADDED_COLUMNS` that isn't there yet.

    Postgres only: on SQLite (the test harness) `create_all` built the tables
    from the current models a moment ago, so the columns are already there --
    and SQLite has no ADD COLUMN IF NOT EXISTS to be idempotent with.
    """
    if engine.dialect.name != "postgresql":
        return
    with engine.begin() as conn:
        for table, column, column_type in _ADDED_COLUMNS:
            conn.execute(
                text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {column_type}")
            )


def _seed_dev_user(session: Session) -> None:
    existing = session.query(User).filter_by(email=DEV_USER_EMAIL).one_or_none()
    if existing is None:
        session.add(User(email=DEV_USER_EMAIL))
        session.commit()


def _seed_model_container_state(session: Session) -> None:
    """Insert a default (`unloaded`) row for every GPU-supervisor-managed
    model that doesn't already have one — see
    apps/background_worker/supervisor/registry.py for the managed set."""
    from apps.background_worker.supervisor.registry import all_managed_model_ids

    existing_ids = {row.model_id for row in session.query(ModelContainerState.model_id).all()}
    for model_id in all_managed_model_ids():
        if model_id not in existing_ids:
            session.add(ModelContainerState(model_id=model_id))
    session.commit()
