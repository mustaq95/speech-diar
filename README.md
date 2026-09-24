# Speech evaluation platform

Compare speech engines on a shared ground truth. The app has two top-level surfaces (toggle in the header):

| Surface | What it measures | Ground truth |
|---------|------------------|--------------|
| **Diarization** | Who spoke when, across multiple diarization engines | None (descriptive comparison on a shared timeline) |
| **Transcript** | Speech-to-text, text-to-speech, and voice cloning | Script text (WER/CER, synthesis timing, clone registration) |

FastAPI API, RQ worker pool, optional GPU container supervisor, and a React/Vite frontend. Python dependencies are managed with [`uv`](https://docs.astral.sh/uv/).

**Core rules (all surfaces):**

- Raw model output never crosses a service boundary interpreted. Each engine has a `runner.py` (native output) and an `adapter.py` (unified contract only).
- Segments and timings are never merged, smoothed, or fabricated. Status and waveforms come from real data.
- Config is centralized in `packages/config/settings.py` from repo-root `.env` only.

Deeper operator notes live in [`CLAUDE.md`](CLAUDE.md).

---

## Prerequisites

| Requirement | Used for |
|-------------|----------|
| [uv](https://docs.astral.sh/uv/) | Python 3.12 env, API, workers, tests |
| Docker + Compose | Postgres, Redis, MinIO; optional GPU ASR/diarization containers |
| `ffmpeg` on `PATH` | Transcode uploads to 16 kHz mono WAV at ingest |
| Node.js 20+ | Frontend dev server and production build |
| NVIDIA GPU (optional) | Local diarization (`uv sync --extra models`), Cohere Transcribe in Compose, and `deploy/` inference containers |

---

## Quick start (local dev)

```bash
git clone https://github.com/mustaq95/speech-diar.git
cd speech-diar

# Python: API + worker + dev tools (no torch/pyannote yet)
uv sync

# Infra: Postgres, Redis, MinIO (+ Cohere Transcribe on GPU hosts — see below)
docker compose up -d

# Config: copy template, then fill credentials for the engines you care about
cp .env.example .env

# API + worker pool + GPU supervisor + local recording API stub
uv run honcho start
```

In another terminal:

```bash
cd apps/frontend
npm install
npm run dev    # http://localhost:5173 — proxies API to http://127.0.0.1:8010
```

Open the UI, pick **Diarization** or **Transcript** in the header, and use **Settings** to enable engines that are configured in your `.env`.

---

## Setup in detail

### 1. Python environment

```bash
uv sync                  # api + worker + dev (matches uv.lock)
uv sync --extra models   # adds torch + pyannote for in-process local diarization (CUDA hosts)
```

On GPU boxes used for diarization or the CTC aligner, set `DIARIZATION_DEVICE=cuda` in `.env`.

### 2. Infrastructure (`docker-compose.yml`)

| Service | Port (host) | Purpose |
|---------|-------------|---------|
| Postgres | 5432 | `AudioFile`, results, transcript references |
| Redis | 6379 | RQ queue (`diarization` jobs) |
| MinIO | 9010 (S3 API), 9099 (console) | Primary object storage for uploads and clips |
| cohere-transcribe | 9025 | Arabic STT container (included via `deploy/cohere-transcribe/`) |

**GPU hosts:** the included Cohere service requests `gpus: all`. If you have no GPU, start only the core stack:

```bash
docker compose up -d postgres redis minio
```

MinIO uses host port **9010** (not 9000) so it does not clash with Parakeet NIM on some DGX layouts. Defaults match `.env.example` (`minioadmin` / bucket `diarization-audio`).

### 3. Configuration (`.env`)

Copy `.env.example` to `.env`. Every runtime setting is documented there. Engines you do not configure show as **not configured** in the UI instead of failing the whole app.

**Diarization (pick any subset):**

| Group | Keys (summary) | Enables |
|-------|------------------|---------|
| Azure real-time | `AZURE_SPEECH_KEY`, region or endpoint | `azure` |
| Azure batch | above + `AZURE_STORAGE_*` | `azure-batch` |
| Hugging Face | `HF_TOKEN` | Gated pyannote / wespeaker downloads |
| NIM Parakeet | `NIM_*` URLs and timeouts | Sortformer streaming/offline |
| Custom HTTP diarizers | `NEMO_*`, `SPEAKER3D_*`, `DIARIZEN_*`, `VIBEVOICE_*`, `MOSS_*` | Matching `deploy/*` containers |

**Transcript surface — script generation:**

| Group | Keys | Enables |
|-------|------|---------|
| LLM | `LLM_*` | `POST /transcript/script` |

**Transcript — STT (batch and/or live):**

| Group | Keys | Engine |
|-------|------|--------|
| LiteLLM gateway | `LITELLM_*`, `STT_*` | Inception-STT (chunked HTTP) |
| TryHamsa | `HAMSA_STT_*` | Streaming WebSocket live path |
| Cohere | `COHERE_TRANSCRIBE_*` | Local container (offline live mode) |
| Speechmatics | `SPEECHMATICS_*` | Batch + realtime |
| ElevenLabs | `ELEVENLABS_*` | Batch + realtime |
| ADEO proxies | `ADEO_QWEN_*`, `ADEO_WHISPER_*` | Per-model HTTP proxies |

**Transcript — TTS and clone:**

| Group | Keys | Engine |
|-------|------|--------|
| Hamsa pod | `HAMSA_TTS_*` | `hamsa-tts` (streaming) |
| Hamsa gateway | `HAMSA_TTS_NEW_*` | `hamsa-tts-new` (single response) |
| Inception | `INCEPTION_TTS_*` | `inception-tts` |
| Voice clone | `VOICE_CLONE_*` | Clone sub-mode (gateway upload) |

**Shared / ops:**

- `DATABASE_URL`, `REDIS_URL`, `S3_*`, `WORKER_CONCURRENCY`, `QUEUE_JOB_TIMEOUT_SEC`
- `RECORDING_API_TOKEN` + `tools/recording_api` on `:8215` (honcho `recording` process) for S3-backed “Load Blob” dev flow
- GPU supervisor: `MAX_RESIDENT_MODELS`, `IDLE_UNLOAD_TIMEOUT_SEC`, per-model cold-start timeouts in `.env.example`

Never commit `.env`. Rotate any key that was ever pasted into a chat or log.

### 4. GPU model containers (`deploy/`)

Heavy inference runs **out of process**. Each model folder typically has:

- `docker-compose*.yml` — ports, GPUs, volumes
- `<model>.env.example` — image tags and runtime options
- `<model>_up.sh` — pull/start and print the URL to put in `.env`

Examples:

```bash
./deploy/nemo-clustering/nemo_clustering_up.sh
./deploy/diarizen/diarizen_up.sh
./deploy/cohere-transcribe/cohere_transcribe_up.sh   # also wired into root docker compose
```

Register managed models in the worker supervisor (`apps/background_worker/supervisor/registry.py`) so cold start, health checks, and residency caps apply. `uv run pytest tests/test_model_registration.py` catches registry/lane drift.

With `uv run honcho start`, the **supervisor** process starts/stops GPU containers on demand; the API and RQ workers stay on the host.

---

## Running

### All-in-one (development)

```bash
uv run honcho start
```

Starts (see [`Procfile`](Procfile)):

| Process | Command | Port |
|---------|---------|------|
| `web` | FastAPI | 8010 |
| `worker` | `./scripts/run_workers.sh` | — |
| `supervisor` | GPU container lifecycle | — |
| `recording` | Local recording API stub | 8215 |

Ctrl+C stops the group. **Honcho ties processes together:** if the worker or supervisor exits, the API goes down too. For a stable API during worker debugging, run uvicorn alone:

```bash
uv run uvicorn apps.backend_api.main:app --reload --port 8010
```

### Workers only

```bash
./scripts/run_workers.sh
```

Starts `WORKER_CONCURRENCY` separate `rq.SimpleWorker` processes (default 3). On shared Redis, use a **per-developer** `REDIS_URL` (different DB index or instance) so `rq empty` on restart does not clear someone else's queue.

### Frontend

```bash
cd apps/frontend
npm install
npm run dev      # :5173
npm run build    # production static build; lint: npx oxlint
```

Runtime poll interval and similar UI knobs come from `GET /config` (driven by `.env`), not a frontend rebuild.

### Typical diarization flow

1. `POST /upload?models=...` — stores audio (MinIO and/or Azure Blob per model lane), one RQ job per model.
2. Poll `GET /evaluations/{id}` until each model is `done` or `failed`.
3. `GET /evaluations/{id}/audio` — proxied stream for playback (browser never talks to MinIO directly).

### Transcript surface (short)

1. Generate or paste a script (`POST /transcript/script` or `POST /transcript/reference`).
2. **STT:** open a live session or run batch engines against recorded audio; finalize persists scores (WER/CER/MER).
3. **TTS:** synthesize per engine/voice; clips stored in MinIO with measured RTF and delivery mode.
4. **Clone:** upload reference audio via the Clone sub-mode; cloned voices appear on `hamsa-tts-new` when the vendor registers them.

Live STT feeds each engine in its native transport (WebSocket vs fixed-interval chunks). See [`CLAUDE.md`](CLAUDE.md) for measured constants you should not “tune away” (chunk size, gateway behaviour, Hamsa headers, etc.).

---

## Test media (`tests/samples/`)

Small fixtures (`3-two-speakers-en.wav`, `katiesteve.wav`, mp3 samples) are in Git.

Large Arabic evaluation clips are **gitignored** (GitHub file size limits):

- `tests/samples/youtube_ar_32min_8spk.wav` (~327 MB source)
- `tests/samples/youtube_ar_32min_8spk.16k.wav` (~59 MB, 16 kHz mono)

Place them locally if you run live tests that need the long clip. The suite still passes without them; `@pytest.mark.live` tests skip unless opted in.

---

## Testing

Backend only (in-memory SQLite + fakeredis, no Docker required for the default suite):

```bash
uv run pytest
uv run pytest tests/test_lanes.py::test_split_by_lane   # single test
RUN_LIVE_TESTS=1 uv run pytest -m live                  # real cloud/GPU endpoints; needs .env
```

Contract drift is enforced: `tests/test_contract_sync.py` cross-checks Python schemas, `packages/shared_contracts/types.ts`, and `apps/frontend/src/types/diarization.ts`.

---

## Auth

No real authentication yet. All requests attach to seeded dev user `dev@example.com` (`apps/backend_api/dependencies.py`).

---

## Repository layout (abbreviated)

```
apps/
  backend_api/          FastAPI routers (upload, evaluations, transcript, tts, voice clone, …)
  background_worker/    RQ worker, lanes, pipelines, models/, transcription/, tts/, voice_clone/
  frontend/             React UI, adapters/, report HTML builders
packages/
  config/settings.py    Single source of truth for env
  database/             SQLAlchemy models + Postgres init shim
  shared_contracts/     Wire format (Python + TypeScript)
  metrics/              Transcript WER/CER scoring
  script_gen/           LLM script generation
deploy/                 GPU / vLLM containers (one folder per engine)
tests/                  pytest suite
docker-compose.yml      Postgres, Redis, MinIO, cohere-transcribe include
```

---

## Adding an engine

**Diarization:** `models/<name>/runner.py` + `adapter.py`, register in `models/__init__.py`, add to `lanes.py` `LANE_MAP`. Containerized models also need supervisor registry entries and `deploy/<name>/`.

**ASR / TTS:** same runner/adapter split under `transcription/` or `tts/`, register in the package `__init__.py`, settings in `.env.example`.

Update all three contract files when the wire format changes.

---

## Deploying elsewhere (e.g. DGX Spark)

1. Clone, `uv sync` (add `--extra models` on CUDA workers).
2. Point `DATABASE_URL`, `REDIS_URL`, `S3_*` at managed services or local Compose.
3. `docker compose up -d` (or subset without GPU services).
4. Start `./deploy/*/_up.sh` for each diarization/STT container you need; mirror URLs in `.env`.
5. `uv run honcho start` or split API/workers/supervisor across systemd/process manager.
6. Build and serve frontend (`npm run build`) behind your reverse proxy; set `CORS_ALLOW_ORIGINS`.

Set `DIARIZATION_DEVICE=cuda` on GPU worker hosts.

---

## Architecture (diarization lanes)

Two storage lanes, chosen per model, never mixed:

```
Browser ──POST /upload──▶ FastAPI ──▶ Postgres + Redis (one RQ job per model)
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
        MinIO (local lane)              Azure Blob (azure lane)
              │                               │
              └─────────── RQ workers ────────┘
```

- **Local lane:** almost all models, including real-time Azure (streams from a local file).
- **Azure lane:** `azure-batch` only (URL/SAS based).

One job per model; each `EvaluationResult` row tracks its own status and timing.

---

## License / contributions

Use the issue workflow in `docs/agents/` if you extend the platform. For day-to-day commands and pitfalls, prefer [`CLAUDE.md`](CLAUDE.md) as the operator cheat sheet.
