"""Shared pytest fixtures.

Every test gets its own isolated in-memory SQLite database (never the real
Postgres) and a fake-Redis-backed RQ queue (never a real Redis) — nothing
here touches dev/prod infrastructure. `client` also never triggers the
app's real `startup` event (that would call `init_db()` against the real
DB), so tests are safe to run with no Docker services up at all.
"""

from __future__ import annotations

import io
import os
import wave
from collections.abc import Iterator

import fakeredis
import pytest
from fastapi.testclient import TestClient
from rq import Queue
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from apps.backend_api.dependencies import get_db
from apps.backend_api.main import app
from packages.database.models import Base, User
from packages.database.session import DEV_USER_EMAIL


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip everything marked `@pytest.mark.live` unless RUN_LIVE_TESTS=1 —
    these hit a real external service (Azure) and cost real time/quota."""
    if os.environ.get("RUN_LIVE_TESTS") == "1":
        return
    skip_live = pytest.mark.skip(reason="live test skipped (set RUN_LIVE_TESTS=1 to run)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture()
def db_session_factory() -> Iterator[sessionmaker[Session]]:
    """A fresh SQLite schema per test, with the seeded dev user already present.

    `expire_on_commit=False`: unlike Postgres (TIMESTAMPTZ), SQLite has no
    real timezone-aware column type, so a post-commit reload would silently
    hand back naive datetimes and break the pipelines' aware-datetime math.
    Keeping committed objects' in-memory values avoids that SQLite-only
    artifact without touching production code (which always runs on Postgres).
    """
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    test_session_local = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True, expire_on_commit=False)
    with test_session_local() as session:
        session.add(User(email=DEV_USER_EMAIL))
        session.commit()
    yield test_session_local
    engine.dispose()


@pytest.fixture()
def db_session(db_session_factory: sessionmaker[Session]) -> Iterator[Session]:
    """A single session onto the isolated test database, for direct DB assertions."""
    session = db_session_factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def fake_queue() -> Queue:
    """An RQ queue backed by fakeredis. `enqueue()` records the job but nothing
    runs it — no worker process is started by these tests."""
    return Queue("diarization", connection=fakeredis.FakeStrictRedis())


@pytest.fixture()
def client(db_session_factory: sessionmaker[Session], fake_queue: Queue, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A TestClient wired to the isolated DB + fake queue.

    Deliberately NOT used as `with TestClient(app) as c:` — that would fire
    the app's `startup` event (`init_db()`), which talks to the real,
    configured Postgres. Route handlers work fine without it since every DB
    access here goes through the overridden `get_db` dependency.
    """

    def override_get_db() -> Iterator[Session]:
        session = db_session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr("apps.backend_api.routers.upload.queue", fake_queue)
    monkeypatch.setattr("apps.backend_api.routers.evaluations.queue", fake_queue)
    test_client = TestClient(app)
    try:
        yield test_client
    finally:
        app.dependency_overrides.clear()


def make_wav_bytes(duration_sec: float = 1.0, framerate: int = 16000) -> bytes:
    """A tiny, valid, silent mono WAV file — real header, real (silent) frames."""
    n_frames = int(duration_sec * framerate)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(framerate)
        wav.writeframes(b"\x00\x00" * n_frames)
    return buffer.getvalue()
