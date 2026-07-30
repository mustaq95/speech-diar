"""Database engine, session factory, and dev bootstrap.

Dev stage uses `create_all`; when the schema changes, drop/recreate the
local database (no migrations tool this iteration — see the plan's
"Design decisions" section).
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from packages.config.settings import get_settings
from packages.database.models import Base, ModelContainerState, User

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
    with SessionLocal() as session:
        _seed_dev_user(session)
        _seed_model_container_state(session)


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
