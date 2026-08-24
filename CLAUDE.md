# gstack

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## Project
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

A speech evaluation platform with **two surfaces**, chosen by the toggle in the top bar centre:

- **Diarization** (the original): upload audio, fan it out to multiple diarization engines, compare their raw output on a shared timeline. No ground truth, so the comparison is descriptive.
- **Transcript**: generate a script, read it aloud, and score multiple STT engines against it live. There IS ground truth, so this one produces real error rates.

FastAPI API + RQ worker pool + React/Vite frontend, all `uv`-managed. `README.md` is the long-form reference (diarization only so far); this file is the operator's cheat sheet.

### Commands

```bash
uv sync                       # Python env (pins 3.12, installs api+worker+dev; NOT the heavy models)
uv sync --extra models        # additionally installs torch + pyannote.audio (CUDA hosts only)
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

The transcript surface needs three credential groups in `.env` or it degrades honestly rather than failing: `LLM_*` (script generation), `LITELLM_*` + `STT_*` (Inception-STT), `HAMSA_*` (TryHamsa). An engine with no credentials reports "not configured" instead of failing every run.

When running Python directly (not via `uv run`), `source .venv/bin/activate` first. This venv has no `pip` module installed — use `uv pip install <pkg>` instead of bare `pip install`.

### Architecture — the rules that constrain every change

**Raw model output never crosses a service boundary interpreted.** Every engine emits a different native shape. Each model has a `runner.py` (executes the engine, returns native output untouched) and an `adapter.py` (the *only* code allowed to understand that shape; translates it to `DiarizationModelRun`). API, DB, and frontend speak only the unified contract. Same split exists on the frontend (`apps/frontend/src/adapters/`).

The one exception is deliberate and narrow: native output is persisted verbatim (`EvaluationResult.raw_output`, `TranscriptResult.raw_output`) and served by `GET /evaluations/{id}/models/{model_id}/raw` and `GET /evaluations/{id}/transcript/{asr_id}/raw`, because the adapters necessarily drop real data (transcript text, per-word timings, confidences, non-speech events) that an evaluation tool should not lose. It travels as an **opaque blob**: nothing outside the owning adapter parses it, both columns are `deferred=True` so the polled routes never load them, and no UI renders them.

**Segments are never merged or cleaned up.** This is a KPI evaluation tool — the UI shows exactly what the model produced, including sub-second gaps. One segment per turn the engine reported. Never coalesce.

**Nothing is fabricated.** No synthetic progress, waveform, or timing. Status/timing come from the DB; the waveform is decoded from the real uploaded audio.

**Two storage lanes, chosen per model, never mixed** (`apps/background_worker/lanes.py` `LANE_MAP` is the single source of truth):
- `local` lane → MinIO/S3, run by `pipelines/local_pipeline.py` — used by every model except `azure-batch`, including `azure` (real-time, needs a local file to stream).
- `azure` lane → Azure Blob, run by `pipelines/azure_pipeline.py` — used **only** by `azure-batch` (URL/SAS-based). No model belongs to both; no code copies bytes between stores.

**One RQ job per model.** `upload.py` enqueues one job per requested model; `worker.py:run_model` just dispatches by lane. Each model's `EvaluationResult` row tracks its own `queued → running → done|failed` independently, so a slow model never blocks a fast one. Concurrency comes from N separate `rq.SimpleWorker` OS processes (SimpleWorker never forks — required on macOS), not from forking.

**The two surfaces never share a recording list.** `AudioFile.surface` (`"diarization" | "transcript"`, indexed) is the single source of truth, set at write time by `store_recording(surface=...)`. `GET /recordings?surface=...` is the only listing route; Projects fetches it and shows one surface at a time. Nothing derives the surface by guessing from a filename or from the absence of `EvaluationResult` rows, which is what the code did before the column existed.

**Nav and the toggle mean different things.** Nav (Dashboard / Upload / Projects / Settings) is what you are doing; the toggle is which surface you are doing it on. Both are explicit state in `App.tsx` (`nav`, `studioMode`, `listSurface`, `transcriptTab`). Never infer one from the other: an earlier version inferred the active tab from "is a recording open", and clicking Dashboard highlighted Upload.

**The browser never talks to MinIO/Azure directly.** `GET /evaluations/{id}/audio` proxies the stream through the API (one origin, no CORS setup).

### Things that bite

- **Config is centralized.** Every service imports `get_settings()` from `packages/config/settings.py` — never read `os.environ` directly. All settings load from `.env` at repo root; moving environments (staging/prod/DGX) is `.env`-only, no code changes. Set `DIARIZATION_DEVICE=cuda` on GPU hosts.
- **The contract lives in three files that must stay in sync:** `packages/shared_contracts/schemas.py` (Python), `packages/shared_contracts/types.ts`, and the frontend working copy `apps/frontend/src/types/diarization.ts`. Change one → change all three. Wire format is camelCase.
- **Every registered model is implemented.** The `available = false` flag still exists on `ModelRunner` for engines that land registered but not yet runnable (`run()` raising `NotImplementedError`); `GET /models` uses it so the UI lists-but-disables them rather than hiding or faking output. Nothing currently sets it.
- **No real auth.** Every request is attributed to a seeded dev user (`dev@example.com`, created on API startup) via `get_current_user` in `apps/backend_api/dependencies.py`.
- **`BATCH_SEGMENT_SECONDS` must stay small (default 3).** Measured: the Inception gateway drops content **non-monotonically** as the posted segment grows. At 25s it discarded roughly 60% of every transcript while still returning 200 OK. It is not a timeout and it does not error; you only see it by diffing word counts. Do not raise this to "reduce overhead".
- **`LLM_TEMPERATURE` is 1.0 on purpose, and lowering it makes the Arabic/English mix WORSE.** Counterintuitive and measured, so it is called out at the setting too. `packages/script_gen` measures the split it actually got and retries up to `SCRIPT_MIX_MAX_ATTEMPTS`; it does not trust the instruction.
- **Blocking I/O in an `async def` route stalls the whole API.** A bare `httpx.post` inside `transcribe_chunk` took `/health` from 4ms to 1066ms and made Stop look hung. Anything blocking in an `async def` handler goes through `run_in_threadpool` (with `partial()` for kwargs). The HTTP client is a pooled `httpx.Client` with keepalive; a per-call client added 73ms of TLS handshake to every chunk.
- **`vite.config.ts` proxy entries need `ws: true`** for the live relay. Without it the socket neither opens nor errors, so any code awaiting open-or-error hangs forever with no message. There is now a deadline (`LIVE_SOCKET_OPEN_TIMEOUT_SEC`) so this can never silently hang again.
- **Live runs must persist `raw_output` too.** The batch pipeline always did; the live path did not, and each engine's native response (Inception's `audio_duration`/`usage`/`word_timestamps`, hamsa's per-segment frames with their own language detection) was discarded the moment the adapter reduced it to a string. It is written in four places now: `live_session.append_part` keeps the payload, `live_session.transcript` returns it in index order, `_persist_session` writes it, and both feeders pass it in. `TranscriptResult.words` (per-word CTC timings) stays batch-only, because the aligner is a GPU model in a second queued job and running it per live session would cost minutes in a flow whose whole point is immediacy.
- **There is no migration tool.** Schema changes are an append-only `_ADDED_COLUMNS` tuple in `packages/database/session.py`, applied as `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` at `init_db()` and Postgres-only. Append, never edit or reorder. `test_added_columns_shim_matches_the_models` (in `tests/test_model_registration.py`) resolves the expected types from `Base.metadata.tables`, so a new column with no shim entry fails the suite.
- **The contract is machine-enforced now.** `tests/test_contract_sync.py` cross-checks `packages/shared_contracts/schemas.py`, `packages/shared_contracts/types.ts` and `apps/frontend/src/types/diarization.ts`. Editing one and forgetting the other two fails the suite instead of failing in the browser.
- **Tests never touch real infra.** `conftest.py` gives each test a fresh in-memory SQLite DB + fakeredis queue, and deliberately does *not* fire the app's `startup` event (which would call `init_db()` against real Postgres). The SQLite session uses `expire_on_commit=False` to dodge a naive-vs-aware datetime artifact — production always runs on Postgres.

### Adding a diarization model

1. `apps/background_worker/models/<name>/runner.py` — run the engine, return native output.
2. `apps/background_worker/models/<name>/adapter.py` — translate to `DiarizationModelRun` (one segment per turn, never merge).
3. Register in `apps/background_worker/models/__init__.py` (one line in `REGISTRY`).
4. Add to `LANE_MAP` in `apps/background_worker/lanes.py` (`local` or `azure`).

An in-process model (pyannote-style, weights loaded in the worker) stops here. A **containerized GPU model** also needs:

5. An entry in `apps/background_worker/supervisor/registry.py` `_registry()` — without it the model is silently treated as in-process: no container start, no health check, no GPU residency cap. `container_name` and the `health_url` port must match the `deploy/<model>/` compose file exactly.
6. Its settings fields (endpoint URL, timeout, cold-start timeout) in `packages/config/settings.py` plus `.env.example` entries — the supervisor registry reads them at import time and crashes with `AttributeError` if missing.
7. The `deploy/<model>/` folder (see "Deploying a GPU model" below).
8. Check `QUEUE_JOB_TIMEOUT_SEC` still exceeds the new model's cold-start + inference sum, or RQ kills the job mid-run.

`tests/test_model_registration.py` cross-checks REGISTRY, LANE_MAP, and the supervisor registry, so `uv run pytest` catches a forgotten step 3–5. The frontend needs no changes: it renders whatever `GET /models` returns.

### The transcript surface

Lives in `apps/background_worker/transcription/` (engines), `packages/metrics/` (scoring), `packages/script_gen/` (the read-aloud script), `apps/backend_api/routers/transcript.py` (the live flow) and `apps/frontend/src/components/TranscriptStudio.tsx` (the UI).

**One flow, three steps.** `POST /transcript/script` generates a script from an LLM **and saves it**: one `AudioFile` row (surface `transcript`, no `s3_key`, `duration_sec` 0) plus its `TranscriptReference`, so the script is in Projects before anything is recorded and regenerating never loses one. `POST /transcript/session` opens a session (per-engine state in Redis, TTL `LIVE_SESSION_TTL_SEC`). Then, while the operator reads, each engine is fed **in its own native mode** and nothing is done to make the two match:

| Engine | Transport | Fed by |
|---|---|---|
| `hamsa` (TryHamsa) | `stream` | `WS /transcript/live/{sid}/{asrId}`, a relay to its socket. Server-side VAD, 100ms / 3200-byte frames. |
| `inception-stt` | `chunks` | `POST /transcript/chunk`, one POST per fixed interval. |

`POST /transcript/session/{sid}/finalize` stores the audio, writes the reference and one `TranscriptResult` per engine, and scores everything.

**A recording row can exist before its audio does, and `has_audio` is the signal.** One row per generated script means a row whose `s3_key` is null. `duration_sec` is 0 there because nothing was measured, not because the audio is zero-length, so `RecordingSummary.has_audio` is what the UI keys on and no route prints that 0 as a duration. Passing `audioFileId` to finalize attaches the recording to that row (`attach_audio`); omitting it creates a row as before, which is the pasted-reference path. **Every row on this surface is named `recording_<ts>` from creation**, generated or pasted, audio or not: `has_audio` already carries whether audio exists, so the filename does not need to, and the old `script_<ts>` -> `recording_<ts>` rename at finalize made one artifact look like two. Rows created before that keep their `script_` name -- renaming them would change a name someone may have referenced, and `tests/test_api_transcript_live.py` pins that they are left alone. Finalize refuses with 409 rather than recording over audio that already exists, because the stored transcripts and scores were measured from the previous take.

**The reference write at finalize is an UPSERT, and that is load-bearing.** A generated script already wrote its `TranscriptReference`, and `audio_file_id` is unique on that table. A blind `db.add` there is the same bug that once 500'd every finalize; `tests/test_api_transcript_live.py` fails on it.

**Raw to raw. No cut-point heuristics.** Chunks are cut at a fixed interval and nothing else. There is deliberately no silence detection, no lookback, no noise floor: any such variable changes what the chunked engine sees and the comparison stops being raw to raw. Do not add one back, and do not "improve" the boundary. A word split across a chunk boundary counts against that engine, which is a real property of that pipeline and is stated in the UI and the report rather than smoothed away.

**Every rate is labelled with the transport that produced it**, because these are measurements of two *pipelines*, not two models in the abstract. RTF is blank for a streaming engine (a real-time protocol consumes audio at 1x by definition, so there is no factor to report and none is invented).

**Scoring** is `packages/metrics/`: hand-written Levenshtein DP with a backtrace, no `jiwer`. Every run stores both the normalized and the raw rate (`wer`/`wer_raw`, `cer`/`cer_raw`); normalization is Arabic-aware (tashkeel, tatweel, alef/hamza/ta-marbuta/ya/kaf folds, Arabic-Indic digits), so the gap between the two columns is orthographic variation rather than misrecognition.

**Reference word counts have two different values and only one is reportable.** `TranscriptReference.wordCount` is a whitespace split of the script; `TranscriptResult.ref_word_count` is the tokenized count that every WER actually divides by, and it runs slightly higher. Report the second. Printing both under one label gives a reader two different totals for the same thing.

**Two reports, one per surface**, both in `apps/frontend/src/report.ts`, both single-file HTML on a light palette (a report gets printed and emailed, so it does not inherit the app's dark theme). `computeReport`/`buildReportHtml` is the diarization one (descriptive). `computeTranscriptReport`/`buildTranscriptAggregateReportHtml` is the transcript one: summary cards, by engine, **by language of script**, by recording. `buildTranscriptReportHtml` is the single-recording variant. Aggregate rates are **total errors over total reference words**, never the mean of per-recording rates, or one short clip would swing the headline as hard as a long one.

**The by-language table groups on the REQUESTED mix**, and also shows the Arabic share actually measured per recording, because the generator follows a mix instruction loosely and the two genuinely differ.

### The TTS sub-mode

The Transcript surface has two sub-modes, chosen by the STT/TTS toggle in the page head: STT measures speech in (read a script aloud, score engines against it), TTS measures speech out (hand the identical script to both TTS engines and compare the audio they return). Lives in `apps/background_worker/tts/` (engines), `apps/backend_api/routers/tts.py` (the routes) and `apps/frontend/src/components/TtsStudio.tsx` (the UI).

**TTS is a sub-mode, not a third surface.** A TTS run hangs off the *same* `AudioFile`/`TranscriptReference` row `POST /transcript/script` already creates, with one `tts_results` row per engine. `RecordingSurface` stays two-valued and `StudioMode`/`listSurface` stay two-valued: an earlier draft added `surface="tts"` and it was wrong, because `AudioFile.surface` is single-valued and a third value makes the most interesting artifact impossible (one script both read aloud by a human *and* synthesized, against the same ground truth). Projects tells them apart by `ttsCount`, the way it already uses `has_audio`.

**`delivery` is not `transport`.** STT engines are labelled `stream|chunks`; TTS engines are labelled `delivery` = `stream|single`. Different vocabulary on purpose, because neither describes the other direction. Every figure carries it: Hamsa streams its body so its time-to-first-audio is a real head start, while Inception buffers the whole response so its first-audio lands at ≈ its total. Both are measured the same way; showing them side by side *without* the delivery label invites the reader to conclude Inception is 100x slower to first audio when what they are seeing is response buffering.

**One engine per request.** `POST /evaluations/{id}/tts/{ttsId}` runs exactly one engine and the browser fires both in parallel, so a Hamsa 502 can never take Inception's clip with it. Same reasoning as `start_transcripts`.

**The write is get-or-reset, never a blind `db.add`.** Re-synthesizing is the natural second action in this UI, and a blind add trips `uq_tts_results_audio_file_tts` and 500s: the identical shape to the `finalize_session` incident. `test_api_tts.py` fails if it is reintroduced.

**Not queued through RQ** — but not because a queue would pollute the measurement. `perf_counter` inside the runner is queue-immune. The real reason is that the flow is interactive and short, and queuing would cost a poll route, a poll loop and a retry path for a call that takes seconds. The route does close its request session before the engine call, though, because synthesis can outlast the 60s `idle_in_transaction_session_timeout`.

**Measured constants, from probing the real endpoints. Do not "clean these up":**
- **Hamsa TTS's `Content-Type: text/event-stream` header is a lie.** The body is raw PCM16 binary in 3200-byte chunks, not SSE. Parse it as bytes.
- **Hamsa's sample rate is nowhere in the response.** `HAMSA_TTS_SAMPLE_RATE` is a *declared* value, not a measurement, and every derived figure (duration, RTF) inherits it. The UI renders Hamsa's rate with "(assumed)" attached for exactly this reason. `test_live_tts.py` is the only thing that can catch it being wrong.
- **Hamsa needs BOTH `X-API-Key` and `Authorization: Bearer`.** Omitting the bearer is a 401.
- **Inception-TTS honours `response_format: "wav"`** (verified real RIFF/WAVE, pcm_s16le, mono, 24000 Hz) but its `Content-Type` is **always** `audio/mpeg` regardless of the container it actually returns. Never trust that header; the adapter sniffs magic bytes and treats a mismatch against the requested format as an error rather than storing mislabelled bytes.

**Voice lists are ONE comma-separated setting per engine, entry 0 is the default** (`HAMSA_TTS_SPEAKER`, `INCEPTION_TTS_VOICE`). There was briefly a default field beside a separate list field, and putting the list on the singular field posted `"Ruba,Sandra"` as one speaker name: the pod accepts that with a 200 and then kills the stream, which reads as a network fault rather than a config typo. `_split_csv` strips whitespace and empty entries, and `tests/test_tts_registration.py` asserts the default never contains a comma. Neither gateway has a list-voices endpoint, so the dropdown is exactly what `.env` holds and nothing is discovered at runtime.

**Generating a script SWITCHES the studio onto the row it just created.** `handleScriptSaved` in `App.tsx` refreshes the list and then `setCurrent`s the new row, because generating a script creates a recording: left on the previous one, the next action (record, or synthesize) would attach to a recording the operator is no longer looking at, and the new entry would sit in Projects with nothing pointing at it. Nav and the tab deliberately do NOT change — only which recording is open. `refreshRecordings` returns its mapped rows for this, since reading `projects` back would see the previous render's value.

That switch is also what makes the script survive the STT/TTS toggle: both sub-modes read `audioFileId` from `current` and fetch the same `TranscriptReference`, so the text and the settings it was generated with (`params.minutes`, `languageMix`, `hardCases`) appear identically on either side. Each side's explicit control picks are cleared after a generate so the controls fall through to those saved params — otherwise the two sides keep private picks and disagree about the settings behind one shared script. `.script-paste` carries the same 15px/1.8 type as `.script-text` for the same reason: a size change across the toggle read as a different document.

**Both sub-modes use ONE script control**, the editable `.script-paste` textarea. STT used to render a read-only justified `<p class="script-text">` whenever a script existed, which blocked editing on that side and, because justification plus zero padding breaks lines differently from a padded textarea, made the identical script occupy a visibly different shape across the toggle. Matching the font alone did not fix it -- the control had to be the same control. STT's box is `readOnly` only once the recording exists (`viewingSaved`): the scores beside it were measured against exactly that text, and editing there would leave them describing something else.

**A script can be brought, not just generated.** `POST /transcript/reference` writes the same audio-less `AudioFile` + `TranscriptReference` row `generate_script` does, with `source="pasted"` and no LLM call — it exists because TTS synthesis reads its text from a STORED reference, and until then the only way to get one was to generate it. The TTS script box is always a textarea: Synthesize saves the draft first (create when there is no row, `PUT /reference` when the text changed) so typing and pressing Synthesize just works. A row created that way must be held in component state — the clip URL, the download link and the waveform decode all key on the recording id, and all three silently skip themselves when it is null.

**One stored clip per (recording, engine, VOICE).** `uq_tts_results_audio_file_tts_voice` is the key, the S3 key carries a slugified voice segment (`tts/{id}/{tts_id}/{voice}.{ext}`), and `voice` is NOT NULL because Postgres treats NULLs as distinct inside a UNIQUE and would let duplicates through. Re-running the SAME voice still resets in place; a different voice adds a row. Widening this was the first constraint change in the repo: `_ADDED_COLUMNS` only emits ADD COLUMN, so `_migrate_tts_results_per_voice()` in `session.py` does the swap, **deletes the old voice-less clips**, and is gated on the old constraint still existing so it can never fire twice. Do not relax that guard.

**The read routes take an optional `?voice=`.** Omitted is served only while unambiguous (one clip for that engine); with several the route 400s naming them rather than returning an arbitrary take. That is what keeps callers written before per-voice storage working.

**A clip belongs to ONE voice, and the card must never present it as another's.** There is one `tts_results` row per (recording, engine), so selecting a different voice does not produce a second clip. Two bugs came from ignoring that: the voice picker defaulted to `engine.defaultVoice` instead of the stored `run.voice`, so reopening a project synthesized with Sandra labelled it "Ruba"; and changing the selection left the previous take's player and all four measurement tiles on screen under the new voice's name. Per-voice storage removes the mismatch case entirely: the card looks up `runs[ttsId::voice]`, so an unsynthesized voice simply has no row and reads "Not synthesized yet." The picker opens on the newest stored clip's voice (the list route orders newest-first within an engine, which is behaviour the UI depends on) and dots the voices that already have one. `test_api_tts.py` asserts the row records the voice that actually ran.

**The TTS card's colour is keyed on `delivery`, set once as a `--engine` CSS variable per panel** (stream=green, single=blue, extending the existing `.engine-dot` convention). Dot, play button, waveform, download fill and the first tile all read it. The waveform tints up to the real playhead (`onTimeUpdate`) and seeks on click; bars are decoded from the stored PCM, so both the shape and the split are measured, never animated. `VoiceSelect`'s subtitle carries `synthesisParams` — engine-level settings true of every voice — because there is no per-voice metadata anywhere to show.

**`probe_audio` must use `.get()` on every ffprobe field.** ffprobe OMITS a key it has no value for rather than emitting null, and MP3 reports no `bits_per_raw_sample` at all. An unguarded `KeyError` there escaped to the outer `suppress` and discarded the sample rate and duration too, so every mp3 clip stored null everything — while WAV kept working through the `wave` fallback, which is exactly why the suite stayed green.

**Clips are stored verbatim, never canonicalized.** `upload.py` transcodes to 16 kHz mono because downstream diarizers need one shape; nothing downstream of a TTS clip needs a shape, and downsampling would delete the fidelity difference between the engines and hand the operator a download that is not what the engine sent. The audio route's `media_type` therefore comes from the row's `audio_format` — `stream_audio` hardcodes `audio/wav` and copying that would serve an mp3 mislabelled.

**`tts_results.raw_output` is metadata only** (status code, headers) and is **not** `deferred=True`: nothing polls this table and headers are not megabytes. The audio bytes are the other half of the native output and live in object storage at `s3_key`, because base64ing minutes of audio into a JSON column would be an unreadable multi-MB duplicate of the object store.

**What is deliberately not measured**, because it would be invented: chars-skipped (neither API reports what it declined to synthesize), any naturalness/MOS score (no reference audio, no MOS model — the honest artifact is A/B playback plus the download), and round-trip WER through an STT engine (measurable, but it measures the TTS+STT pair, not the TTS).

### Adding a TTS engine

1. `apps/background_worker/tts/<name>/runner.py` and `adapter.py`, same rule as everywhere: the runner returns native output, the adapter is the only code that understands it.
2. One `TtsEngine(...)` entry in `TTS_ENGINES` (`tts/__init__.py`) plus `TTS_DELIVERY` and, if it should be compared, `COMPARISON_TTS_IDS`. `tests/test_tts_registration.py` cross-checks all three.
3. Its settings in `packages/config/settings.py` plus `.env.example`, including a comma-separated voices list (neither gateway exposes a list-voices endpoint, so the dropdown is whatever `.env` holds).
4. A `configured` predicate, so an unconfigured host shows "not configured" instead of failing every run.

### Adding an ASR engine

1. `apps/background_worker/transcription/<name>/runner.py` and `adapter.py`, same rule as diarization: the runner returns native output, the adapter is the only code that understands it.
2. One `AsrEngine(...)` entry in `ASR_ENGINES` (`transcription/__init__.py`), including a `configured` predicate so an unconfigured host shows "not run" instead of failing every upload.
3. Its settings in `packages/config/settings.py` plus `.env.example`.
4. If it should be selectable by the Live Speech mode toggle, `MODE_TO_ASR_ID`. The transcript surface selects by `asr_id`, not by mode.
5. Live runs must set `raw_output`. See "Things that bite".

### Deploying a GPU model
GPU inference runs outside the API/worker, in `deploy/<model>/`.
The app talks to it over a network endpoint (gRPC, HTTP, etc.), not by loading weights in-process.
1. `deploy/<model>/docker-compose.yml` (or equivalent) — container/runtime config, `gpus: all`, host ports, volumes.
2. `deploy/<model>/<model>.env.example` — committed template (ports, image/tag, runtime options).
3. `deploy/<model>/<model>.env` — local secrets (gitignored).
4. `deploy/<model>/<model>_up.sh` — pull/start, health check, print endpoint URL.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Harness

Backend-only (no frontend unit tests). Everything is defined in `tests/conftest.py` and runs with **no Docker services up** — the harness stands in for real infra:

- **`db_session_factory` / `db_session`** — a fresh in-memory SQLite schema per test, with the seeded dev user already inserted. Never the real Postgres. Uses `expire_on_commit=False` to avoid SQLite handing back naive datetimes (production runs on Postgres/`TIMESTAMPTZ`).
- **`fake_queue`** — an RQ `Queue` backed by `fakeredis`. `enqueue()` records the job but nothing runs it; no worker process starts.
- **`client`** — a `TestClient` with `get_db` overridden to the SQLite session and `upload.queue` monkeypatched to `fake_queue`. Deliberately **not** used as `with TestClient(app)` — that would fire the app's `startup` event (`init_db()` against real Postgres). Route handlers work without startup because all DB access goes through the overridden dependency.
- **`make_wav_bytes(duration_sec, framerate)`** — builds a tiny valid silent mono WAV (real header, real frames) for upload/duration tests.
- **Live tests** — `pytest_collection_modifyitems` skips everything marked `@pytest.mark.live` unless `RUN_LIVE_TESTS=1`; those hit the real Azure batch endpoint and cost time/quota. The marker is declared in `pyproject.toml` under `[tool.pytest.ini_options]`.

Test files map to the layers above. Diarization: `test_adapters.py`, `test_contracts.py`, `test_lanes.py`, `test_worker_dispatch.py` (unit); `test_api_*.py`, `test_pipelines.py` (integration); `test_live_azure_batch.py`, `test_live_hamsa.py` (live). Transcript: `test_transcript_adapters.py`, `test_transcript_metrics.py`, `test_transcript_pipeline.py`, `test_inception_runner.py` (unit); `test_api_transcript.py`, `test_api_transcript_live.py`, `test_api_recordings.py` (integration). Cross-cutting: `test_contract_sync.py`, `test_model_registration.py`.

A green suite is not proof a route works. `finalize_session` 500'd on **every** call (a duplicate `db.add` batched into one multi-VALUES INSERT against a unique index) while the whole suite stayed green, because nothing exercised the route end to end. `test_api_transcript_live.py` now does, and 6 of its 11 tests fail if that bug is reintroduced. When adding a route, add the test that would have caught the bug you just fixed, and verify it fails without the fix.

## 6. Loops & Autonomy

Sections 4 and 5 define the loop and how to verify it. These rules govern running it without me approving every turn.

- "Done" is defined in section 5. If a task has no programmatic check, say so before starting rather than looping with no stop condition.
- Prefer the built-in autonomy commands over asking me to type "continue":
  - `/goal <verifiable end state>` for "keep working until it's correct" (test-fix-retest cycles, lint-clean, migrations with a clear stop condition).
  - `/loop <interval, or until: condition>` for polling or repeating a check (watch CI, re-run a suite).
  - `/batch <one mechanical change>` for a repetitive edit spread across many files. Requires git; it spawns parallel worktree agents and opens a PR per agent, so it also multiplies token usage.
- The `/goal` completion check is read from your output by a separate evaluator model, not from the filesystem. State conditions you can demonstrate in the transcript (show the passing test run), not silent file assertions.
- Always work on a git branch so changes can be reverted. Never start an autonomous loop without an iteration cap.
- Loops are for code with programmatic verification. Do not loop on judgment-heavy work, design decisions, or long compute jobs (training runs, quantization sweeps). Those are a script or my call, not a loop.
- If you are still stuck when the cap is reached, stop. Document what is blocking progress, what you tried, and suggested next steps. Do not thrash.

## Text

Rules for human-readable text you produce (PR descriptions, comments, docstrings, commit messages, docs):

- No em-dashes or other long dashes. Use commas, periods, or parentheses instead.
- Cut filler and hedging: "um", "basically", "essentially", "it's worth noting", "of course".
- Vary sentence length. Do not pad a short, correct statement into a long fuzzy one, and do not chain choppy fragments either.
- Avoid the usual LLM tells: no "it's not just X, it's Y", no "delve", no overwrought openers.
- Reread what you wrote before you finish. Delete anything that does not earn its place.

Use the `/browse` skill from gstack for all web browsing. Never use `mcp__claude-in-chrome__*` tools.

Available gstack skills:

- `/office-hours`
- `/plan-ceo-review`
- `/plan-eng-review`
- `/plan-design-review`
- `/design-consultation`
- `/design-shotgun`
- `/design-html`
- `/review`
- `/ship`
- `/land-and-deploy`
- `/canary`
- `/benchmark`
- `/browse`
- `/connect-chrome`
- `/qa`
- `/qa-only`
- `/design-review`
- `/setup-browser-cookies`
- `/setup-deploy`
- `/setup-gbrain`
- `/retro`
- `/investigate`
- `/document-release`
- `/document-generate`
- `/codex`
- `/cso`
- `/autoplan`
- `/plan-devex-review`
- `/devex-review`
- `/careful`
- `/freeze`
- `/guard`
- `/unfreeze`
- `/gstack-upgrade`
- `/learn`