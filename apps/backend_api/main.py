"""FastAPI application entry point.

Run locally (from repo root):  uv run uvicorn apps.backend_api.main:app --reload --port 8010
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.background_worker.transcription import (
    ASR_ENGINES,
    DEFAULT_TRANSCRIPTION_MODE,
    MODE_TO_ASR_ID,
)
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
def frontend_config() -> dict[str, object]:
    """Runtime settings the frontend can pick up without a rebuild.

    The API's own address (VITE_API_BASE_URL) is the one value that must be
    known before the frontend can call this endpoint at all, so it stays a
    build-time setting; everything else here is .env-only, both to set and
    to change later.

    The transcription block reports, per mode, the engine's display name and
    whether it is configured on this host. The panel uses it to render the
    online/offline toggle and disable a side whose engine has no
    credentials/endpoint, instead of offering a run that is certain to fail.
    `defaultTranscriptionMode` is the mode an upload auto-transcribes in and the
    one the toggle starts on; it is not a fallback order, and the other mode
    never runs unless asked for.

    Which mode PRODUCED a given transcript is a different question, answered on
    each TranscriptResult row and never rewritten by config.

    Read from the process-wide `settings` captured at import, so changing an
    engine's credentials in .env takes effect on restart — the same contract as
    every other setting here.
    """
    return {
        "pollIntervalMs": settings.frontend_poll_interval_ms,
        "defaultTranscriptionMode": DEFAULT_TRANSCRIPTION_MODE,
        "transcriptionModes": {
            mode: {
                "asrName": ASR_ENGINES[asr_id].name,
                "configured": ASR_ENGINES[asr_id].configured(settings),
            }
            for mode, asr_id in MODE_TO_ASR_ID.items()
        },
    }
