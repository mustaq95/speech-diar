"""FastAPI application entry point.

Run locally (from repo root):  uv run uvicorn apps.backend_api.main:app --reload --port 8010
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.backend_api.routers import evaluations, models, upload
from packages.config.logging import configure_logging
from packages.config.settings import get_settings
from packages.database.session import init_db

configure_logging()
settings = get_settings()

app = FastAPI(
    title="Speaker Diarization Evaluation Platform",
    version="0.1.0",
)


@app.on_event("startup")
def _startup() -> None:
    init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(evaluations.router)
app.include_router(models.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
def frontend_config() -> dict[str, int]:
    """Runtime settings the frontend can pick up without a rebuild.

    The API's own address (VITE_API_BASE_URL) is the one value that must be
    known before the frontend can call this endpoint at all, so it stays a
    build-time setting; everything else here is .env-only, both to set and
    to change later.
    """
    return {"pollIntervalMs": settings.frontend_poll_interval_ms}
