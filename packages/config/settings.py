"""Single source of truth for all runtime configuration.

Every service (API, worker, tests) imports `get_settings()` instead of
reading `os.environ` directly. Values are loaded from `.env` at the repo
root; nothing here is a substitute for setting a real value in `.env` —
these are dev-friendly defaults only, always overridable.

Moving to a new environment (staging, production, DGX Spark) means editing
`.env` only; no code changes.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Database (Postgres) ---
    database_url: str = "postgresql+psycopg://diarization:diarization@localhost:5432/diarization"

    # --- Queue (Redis / RQ) ---
    redis_url: str = "redis://localhost:6379/0"
    worker_concurrency: int = 1

    # --- Primary storage lane: MinIO / S3 (local models) ---
    s3_endpoint_url: str | None = "http://localhost:9000"
    s3_bucket: str = "diarization-audio"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    # --- Azure Speech (real-time + batch diarization) ---
    azure_speech_key: str | None = None
    azure_speech_region: str | None = None
    azure_speech_endpoint: str | None = None
    azure_realtime_transcribe_timeout_sec: int = 600
    azure_batch_poll_interval_sec: int = 10
    azure_batch_job_timeout_sec: int = 1800
    azure_batch_time_to_live: str = "PT4H"
    azure_batch_max_speakers: int = 8
    locale: str = "en-US"

    # --- Azure lane storage: Blob (only touched when the Azure model is on) ---
    azure_storage_account_name: str | None = None
    azure_storage_account_key: str | None = None
    azure_storage_container_name: str = "diarization-audio"
    azure_storage_sas_read_expiry_hours: int = 48

    # --- API ---
    api_port: int = 8010
    cors_allow_origins: str = "http://localhost:5173"

    # --- Frontend (served via GET /config; no rebuild needed for this value) ---
    frontend_poll_interval_ms: int = 1500

    # --- Local diarization models (DGX Spark: cuda; dev machine: cpu) ---
    diarization_device: str = "cpu"

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
