"""Database engine, session factory, and dev bootstrap.

Dev stage uses `create_all`; when the schema changes, drop/recreate the
local database (no migrations tool this iteration — see the plan's
"Design decisions" section).
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from packages.config.settings import get_settings
from packages.database.models import Base, User

DEV_USER_EMAIL = "dev@example.com"

engine = create_engine(
    get_settings().database_url,
    future=True,
    # Backstop against a leaked/abandoned session (e.g. an aborted streaming
    # response) permanently starving the pool: Postgres kills it after 60s
    # of holding an open transaction instead of holding it forever.
    connect_args={"options": "-c idle_in_transaction_session_timeout=60000"},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """Create tables if missing and seed the single dev user.

    Called once at API startup. There is no real authentication yet — every
    request is attributed to this one seeded user (see dependencies.py).
    """
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as session:
        _seed_dev_user(session)


def _seed_dev_user(session: Session) -> None:
    existing = session.query(User).filter_by(email=DEV_USER_EMAIL).one_or_none()
    if existing is None:
        session.add(User(email=DEV_USER_EMAIL))
        session.commit()
