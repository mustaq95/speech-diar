# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

A speaker-diarization evaluation platform: upload audio, fan it out to multiple diarization engines, compare their raw output on a shared timeline. FastAPI API + RQ worker pool + React/Vite frontend, all `uv`-managed. `README.md` is the long-form reference; this file is the operator's cheat sheet.

## Commands

```bash
uv sync                       # Python env (pins 3.12, installs api+worker+dev; NOT the heavy models)
uv sync --extra models        # additionally installs torch + pyannote.audio + whisperx (CUDA hosts only)
docker compose up -d          # local infra: postgres, redis, minio

uv run honcho start           # run API + worker pool together (Procfile); Ctrl+C stops both
uv run uvicorn apps.backend_api.main:app --reload --port 8010   # API alone (default port 8010, not 8000)
./scripts/run_workers.sh      # worker pool alone: WORKER_CONCURRENCY SimpleWorker processes

uv run pytest                 # full backend suite; no Docker services needed (in-memory SQLite + fakeredis)
uv run pytest tests/test_lanes.py::test_split_by_lane   # single test
RUN_LIVE_TESTS=1 uv run pytest -m live                  # the one test that hits real Azure (needs .env creds)

cd apps/frontend && npm install && npm run dev          # Vite dev server on :5173, expects API on 127.0.0.1:8010
npm run build                 # tsc -b && vite build   (lint: npx oxlint, config in .oxlintrc.json)
```

There is no lint/format step for the Python side.

## Architecture — the rules that constrain every change

**Raw model output never crosses a service boundary.** Every engine emits a different native shape. Each model has a `runner.py` (executes the engine, returns native output untouched) and an `adapter.py` (the *only* code allowed to understand that shape; translates it to `DiarizationModelRun`). API, DB, and frontend speak only the unified contract. Same split exists on the frontend (`apps/frontend/src/adapters/`).

**Segments are never merged or cleaned up.** This is a KPI evaluation tool — the UI shows exactly what the model produced, including sub-second gaps. One segment per turn the engine reported. Never coalesce.

**Nothing is fabricated.** No synthetic progress, waveform, or timing. Status/timing come from the DB; the waveform is decoded from the real uploaded audio.

**Two storage lanes, chosen per model, never mixed** (`apps/background_worker/lanes.py` `LANE_MAP` is the single source of truth):
- `local` lane → MinIO/S3, run by `pipelines/local_pipeline.py` — used by `pyannote`, `whisperx`, and `azure` (real-time, needs a local file to stream).
- `azure` lane → Azure Blob, run by `pipelines/azure_pipeline.py` — used **only** by `azure-batch` (URL/SAS-based). No model belongs to both; no code copies bytes between stores.

**One RQ job per model.** `upload.py` enqueues one job per requested model; `worker.py:run_model` just dispatches by lane. Each model's `EvaluationResult` row tracks its own `queued → running → done|failed` independently, so a slow model never blocks a fast one. Concurrency comes from N separate `rq.SimpleWorker` OS processes (SimpleWorker never forks — required on macOS), not from forking.

**The browser never talks to MinIO/Azure directly.** `GET /evaluations/{id}/audio` proxies the stream through the API (one origin, no CORS setup).

## Things that bite

- **Config is centralized.** Every service imports `get_settings()` from `packages/config/settings.py` — never read `os.environ` directly. All settings load from `.env` at repo root; moving environments (staging/prod/DGX) is `.env`-only, no code changes. Set `DIARIZATION_DEVICE=cuda` on GPU hosts.
- **The contract lives in three files that must stay in sync:** `packages/shared_contracts/schemas.py` (Python), `packages/shared_contracts/types.ts`, and the frontend working copy `apps/frontend/src/types/diarization.ts`. Change one → change all three. Wire format is camelCase.
- **`pyannote` and `whisperx` are stubs** — `available = false`, their `run()` raises `NotImplementedError`. `GET /models` uses the flag so the UI lists-but-disables them rather than hiding or faking output.
- **No real auth.** Every request is attributed to a seeded dev user (`dev@example.com`, created on API startup) via `get_current_user` in `apps/backend_api/dependencies.py`.
- **Tests never touch real infra.** `conftest.py` gives each test a fresh in-memory SQLite DB + fakeredis queue, and deliberately does *not* fire the app's `startup` event (which would call `init_db()` against real Postgres). The SQLite session uses `expire_on_commit=False` to dodge a naive-vs-aware datetime artifact — production always runs on Postgres.

## Test harness

Backend-only (no frontend unit tests). Everything is defined in `tests/conftest.py` and runs with **no Docker services up** — the harness stands in for real infra:

- **`db_session_factory` / `db_session`** — a fresh in-memory SQLite schema per test, with the seeded dev user already inserted. Never the real Postgres. Uses `expire_on_commit=False` to avoid SQLite handing back naive datetimes (production runs on Postgres/`TIMESTAMPTZ`).
- **`fake_queue`** — an RQ `Queue` backed by `fakeredis`. `enqueue()` records the job but nothing runs it; no worker process starts.
- **`client`** — a `TestClient` with `get_db` overridden to the SQLite session and `upload.queue` monkeypatched to `fake_queue`. Deliberately **not** used as `with TestClient(app)` — that would fire the app's `startup` event (`init_db()` against real Postgres). Route handlers work without startup because all DB access goes through the overridden dependency.
- **`make_wav_bytes(duration_sec, framerate)`** — builds a tiny valid silent mono WAV (real header, real frames) for upload/duration tests.
- **Live tests** — `pytest_collection_modifyitems` skips everything marked `@pytest.mark.live` unless `RUN_LIVE_TESTS=1`; those hit the real Azure batch endpoint and cost time/quota. The marker is declared in `pyproject.toml` under `[tool.pytest.ini_options]`.

Test files map to the layers above: `test_adapters.py`, `test_contracts.py`, `test_lanes.py`, `test_worker_dispatch.py` (unit); `test_api_*.py`, `test_pipelines.py` (integration); `test_live_azure_batch.py` (live).

## Adding a diarization model

1. `apps/background_worker/models/<name>/runner.py` — run the engine, return native output.
2. `apps/background_worker/models/<name>/adapter.py` — translate to `DiarizationModelRun` (one segment per turn, never merge).
3. Register in `apps/background_worker/models/__init__.py` (one line in `REGISTRY`).
4. Add to `LANE_MAP` in `apps/background_worker/lanes.py` (`local` or `azure`).

Nothing in the API, database, or frontend changes.
