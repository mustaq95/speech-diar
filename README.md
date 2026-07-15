# Speaker Diarization Evaluation Platform

Upload audio, run it through multiple speaker-diarization engines, and compare
their raw output side by side on a shared timeline.

**Core architectural rules:**

- Every engine emits a different native output shape. Raw output never
  crosses a service boundary — each engine has an *adapter* that translates it
  into the unified contract in `packages/shared_contracts`; everything else
  (API, database, frontend) speaks only that contract.
- Segments are never merged or cleaned up. What the UI shows is exactly what
  the model produced — this is a KPI evaluation tool, so a model's raw
  turn-taking behavior (including sub-second gaps) has to stay visible.
- Nothing is fabricated. There is no synthetic progress bar, no synthetic
  waveform, no canned timing. Status, timing, and the waveform are all
  computed from real data — a queued/running/done/failed model status from
  the database, and a waveform decoded from the real uploaded audio.

## Architecture

Two independent storage lanes, chosen per model, never mixed:

```
                              ┌─────────────────────┐
                     ┌───────▶│  MinIO (local lane)  │◀───────┐
                     │        └─────────────────────┘        │
Browser ──POST /upload──▶ FastAPI API                    RQ worker
                     │        ┌─────────────────────┐        │
                     └───────▶│ Azure Blob (azure    │◀───────┘
                              │ lane — only when      │
                              │ azure-batch is used)  │
                              └─────────────────────┘

FastAPI API ──▶ Postgres (AudioFile + one EvaluationResult row per model)
FastAPI API ──▶ Redis/RQ  (one job enqueued per model, independent status)
Browser ──poll──▶ GET /evaluations/{id} ──▶ Postgres
Browser ──▶ GET /evaluations/{id}/audio ──▶ API streams from whichever lane owns the file
```

- **Local lane (MinIO)** — the primary lane, used by every diarization model
  except `azure-batch`, including the real-time `azure` model (which just
  needs a local file to stream from).
- **Azure lane (Blob)** — used **only** by `azure-batch`, and only when it's
  requested. Uploading the same file's content twice reuses the already
  staged blob (content-hash-keyed) instead of re-uploading it.
- One RQ job **per model**, so a slow model never blocks a fast one, and each
  model's `EvaluationResult` row tracks its own `queued → running →
  done|failed` status, timing, and error independently.
- The browser never talks to MinIO or Azure Blob directly — `GET
  /evaluations/{id}/audio` proxies the stream through the API, so playback
  and the real decoded waveform both come from one origin (no CORS setup).

## Repository layout

```
├── apps/
│   ├── frontend/                 # React + Vite UI (TypeScript)
│   │   └── src/
│   │       ├── components/       # Visual UI elements (contract-agnostic)
│   │       ├── adapters/         # Frontend adapter layer (raw -> contract)
│   │       └── types/            # Working copy of the unified contract
│   │
│   ├── backend_api/              # FastAPI web server
│   │   ├── main.py               # Application entry point
│   │   ├── routers/              # /upload, /evaluations, /models
│   │   └── dependencies.py       # DB session + current-user injection
│   │
│   └── background_worker/        # RQ task + pluggable models
│       ├── worker.py             # run_model(audio_file_id, model_id) — dispatches by lane
│       ├── lanes.py               # The one place that maps model id -> storage lane
│       ├── pipelines/
│       │   ├── local_pipeline.py  # MinIO lane: download -> run -> write result
│       │   └── azure_pipeline.py  # Azure lane: SAS URL -> run -> write result
│       └── models/                # One folder per engine
│           ├── base_model.py      # Abstract Runner + Adapter every model follows
│           ├── azure_speech/      # Real-time ConversationTranscriber (local file)
│           ├── azure_batch/       # Batch v3.2 transcription + diarization (URL-based)
│           ├── pyannote/          # In-process: pyannote community-1
│           ├── nim_sortformer_str/ # NVIDIA Parakeet + Sortformer NIM (streaming, gRPC)
│           ├── nim_sortformer_ofl/ # NVIDIA Parakeet + Sortformer NIM (offline, gRPC)
│           ├── nemo_clustering/   # NeMo VAD + TitaNet + spectral clustering (HTTP)
│           ├── speaker3d_clustering/ # 3D-Speaker CAM++ clustering (HTTP)
│           ├── diarizen/          # DiariZen WavLM + Conformer EEND (HTTP)
│           └── vibevoice/         # VibeVoice-ASR 8B, joint ASR+diarization (HTTP)
│
├── packages/
│   ├── config/
│   │   ├── settings.py            # Single source of truth for ALL config (.env)
│   │   └── logging.py             # Shared logging setup for API + worker
│   ├── database/                  # SQLAlchemy models + session/init
│   ├── storage/                   # s3_client.py (MinIO) + azure_blob.py (Azure Blob) — separate, never cross-called
│   └── shared_contracts/          # SOURCE OF TRUTH for the wire format
│       ├── schemas.py             # Pydantic models (Python side)
│       └── types.ts               # Same contract (TypeScript side)
│
├── tests/                          # pytest suite (backend only — see Testing)
├── docker-compose.yml              # Local infra: postgres, redis, minio
├── pyproject.toml                  # uv-managed deps (api/worker groups + optional `models` extra + dev)
└── .env.example                    # Every runtime setting, dev-friendly defaults
```

## Setup

Requires [`uv`](https://docs.astral.sh/uv/), Docker, and `ffmpeg` on the
`PATH`. Uploads in any format other than 16-bit PCM WAV (MP3, M4A, FLAC, …)
are transcoded to WAV at ingest via the `ffmpeg` CLI; install it with
`apt install ffmpeg` (or your platform's equivalent).

```bash
# 1. Python env (pins Python 3.12, installs api+worker+dev deps from uv.lock)
uv sync

# 2. Local infrastructure: Postgres, Redis, MinIO
docker compose up -d

# 3. Config — copy and fill in Azure credentials (everything else has a
#    working default that matches docker-compose.yml)
cp .env.example .env
```

`uv sync --extra models` additionally installs the heavy, CUDA-oriented local
engines (`torch`, `pyannote.audio`) — skip this on a machine without a GPU.
The containerized models (`deploy/`) are unaffected either way: they run
their own GPU inference out of process and the worker only speaks HTTP to
them.

## Running

**One command** (starts the API and the worker pool together, via the
[`Procfile`](Procfile) and [`honcho`](https://github.com/nickstenning/honcho),
a small process manager installed as a dev dependency):

```bash
uv run honcho start
```

Ctrl+C stops both. Logs are prefixed `web |` / `worker |`. Under the hood
this is just running the two commands below as siblings — honcho doesn't
know anything about the API or the queue, it only starts/stops processes and
forwards signals.

> **These are host processes, not containers.** `docker ps` shows postgres,
> redis, minio, and the model inference containers — it does **not** show the
> API, workers, or GPU supervisor, which honcho runs directly on the host.
> "All containers green" does not mean the API is up; check with
> `ss -ltnp | grep 8010`.
>
> **honcho runs them as one group: if any one exits, it terminates all the
> others**, including the API. So a crash in a worker or the supervisor takes
> the whole stack down and uploads start failing with `ECONNREFUSED` on
> `:8010`. On this DGX box the most likely trigger is a **GPU out-of-memory**
> in a model container (kernel log: `NVRM ... NV_ERR_NO_MEMORY`) — tune
> `MAX_RESIDENT_MODELS` / `IDLE_UNLOAD_TIMEOUT_SEC` in `.env` if it recurs.
> Restart cleanly with `uv run honcho start`. To make a model crash unable to
> take the API down, run the API in its own process
> (`uv run uvicorn apps.backend_api.main:app --port 8010`) separately from
> `honcho start` (workers + supervisor).

Or run them separately in two terminals, which is also exactly what you'd do
in production (see [Deploying elsewhere](#deploying-elsewhere-eg-dgx-spark)):

**Backend API** (port 8010 by default):

```bash
uv run uvicorn apps.backend_api.main:app --reload --port 8010
```

Creates tables and seeds the single dev user on startup (see [Auth](#auth)).

**Background worker(s)** (consume jobs from the `diarization` RQ queue —
one job per model, so multiple models for the same upload run at the same
time as long as more than one worker process is listening):

```bash
./scripts/run_workers.sh
```

This starts `WORKER_CONCURRENCY` (see `.env`, default `3`) independent
`rq worker --worker-class rq.SimpleWorker diarization` processes. Each is a
separate OS process — `SimpleWorker` itself never forks (required on
macOS) — and Redis hands each queued job to exactly one of them, so N
processes means N models processing concurrently. Raise
`WORKER_CONCURRENCY` on DGX/production hosts to match how many models you
expect to run at once.

Running a single `uv run rq worker --worker-class rq.SimpleWorker
diarization` still works for quick debugging, but only one job runs at a
time.

Every time it starts, `run_workers.sh` clears the `diarization` queue
(`rq empty`, so a restart always begins from a clean slate) and cleans up any
worker processes it left running from a previous ungraceful stop — it tracks
this itself in a local `.worker_pool.pids` file scoped to this checkout, so
it never touches another process on the machine. **If you're running this on
a shared machine (e.g. DGX Spark) alongside other users**, give each
person/instance their own `REDIS_URL` (own port or DB index) in their own
`.env` — the self-heal and `rq empty` are only isolated per Redis instance,
not per user, so sharing one Redis + queue name means sharing that queue
(intended for one production deployment serving many end-users, not for
several people's independent dev instances).

**Frontend**:

```bash
cd apps/frontend
npm install
npm run dev        # http://localhost:5173 (expects the API on 127.0.0.1:8010)
```

Drop a WAV file to diarize it for real through whichever models are enabled
in Settings. The "Synthetic demo" link on the empty dashboard loads a canned,
clearly-labeled comparison that never touches the backend.

### Typical flow

1. `POST /upload?models=azure` — streams the WAV into MinIO, creates an
   `AudioFile` row + one `queued` `EvaluationResult` row per model, enqueues
   one RQ job per model, and returns immediately (`UploadAck`). Mix in
   `azure-batch` too and it's staged into Azure Blob instead, in the same call.
2. Frontend polls `GET /evaluations/{audioFileId}` until every model is
   `done` or `failed`.
3. `GET /evaluations/{audioFileId}/audio` streams the audio back for playback
   and waveform decoding.

### Azure models

| id | engine | input | storage lane |
|---|---|---|---|
| `azure` | Speech SDK `ConversationTranscriber` (real-time) | local file | MinIO (primary lane) |
| `azure-batch` | REST v3.2 batch transcription + diarization | http(s) URL | Azure Blob (azure lane) |

- `azure` needs only `AZURE_SPEECH_KEY` + `AZURE_SPEECH_REGION` (or
  `AZURE_SPEECH_ENDPOINT`) — no blob storage involved.
- `azure-batch` additionally needs `AZURE_STORAGE_ACCOUNT_NAME` +
  `AZURE_STORAGE_ACCOUNT_KEY` so the API can stage the upload to Blob and
  hand Azure a SAS URL to fetch it from.
- Manual smoke test against a URL Azure can already reach (skips the
  DB/queue entirely):
  ```bash
  curl -X POST "http://127.0.0.1:8010/upload/url?models=azure-batch&url=<public-audio-url>"
  ```

## Testing

Backend-only (no frontend unit tests): unit tests for adapters/contract/lane
routing, integration tests for the API and worker pipelines via an isolated
in-memory SQLite database + a fake-Redis-backed queue — no Docker services
required.

```bash
uv run pytest              # everything except the live Azure test
```

One test hits the real Azure batch endpoint against a public two-speaker
sample and is skipped unless explicitly opted in (needs Azure credentials in
`.env`):

```bash
RUN_LIVE_TESTS=1 uv run pytest -m live
```

## Auth

There's no real authentication yet. Every request is attributed to a single
seeded dev user (`dev@example.com`, created on API startup) via
`get_current_user` in `apps/backend_api/dependencies.py`. The `User` table
already has the shape real auth will need later, so swapping this out won't
change any caller.

## Adding a new diarization model

1. `apps/background_worker/models/<name>/runner.py` — execute the engine,
   return its native output untouched.
2. `apps/background_worker/models/<name>/adapter.py` — translate that native
   output into `DiarizationModelRun` (one segment per turn the engine
   produced — never merge).
3. Register it in `apps/background_worker/models/__init__.py` (one line).
4. Add it to `apps/background_worker/lanes.py`'s `LANE_MAP` (`"local"` if it
   reads a local file via MinIO, `"azure"` if it's URL-based via Blob).

Nothing in the API, database, or frontend changes.

## Contract sync

`packages/shared_contracts/schemas.py`, `packages/shared_contracts/types.ts`,
and the frontend's working copy `apps/frontend/src/types/diarization.ts`
describe the same camelCase wire format. If the contract changes, update all
three.

## Deploying elsewhere (e.g. DGX Spark)

Moving to a new machine — staging, production, or a DGX Spark box for the
GPU-backed local models — is `.env`-only, no code changes:

```bash
git clone <repo> && cd <repo>
uv sync --extra models        # pulls CUDA torch + pyannote too
cp .env.example .env          # set DATABASE_URL/REDIS_URL/S3_*/AZURE_* for that environment
docker compose up -d          # or point at already-managed Postgres/Redis/S3
```

Set `DIARIZATION_DEVICE=cuda` in `.env` so the local models use the GPU.
