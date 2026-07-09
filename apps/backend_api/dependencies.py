"""FastAPI dependency injection: database sessions and auth."""

from collections.abc import Iterator

from fastapi import Depends
from sqlalchemy.orm import Session

from packages.database.models import User
from packages.database.session import DEV_USER_EMAIL, SessionLocal


def get_db() -> Iterator[Session]:
    """Yield a database session per request."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def get_current_user(db: Session = Depends(get_db)) -> User:
    """Resolve the authenticated user.

    Single seeded dev user for this iteration (no real auth yet); the
    User table already carries the shape real auth will need later, so
    swapping this out won't change any caller.
    """
    return db.query(User).filter_by(email=DEV_USER_EMAIL).one()
