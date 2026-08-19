/**
 * Source of truth for every data structure that crosses a service boundary —
 * TypeScript side. Mirrors `schemas.py` (Pydantic) field for field; the wire
 * format is camelCase JSON.
 *
 * The frontend's working copy lives at `apps/frontend/src/types/diarization.ts`
 * so the app stays self-contained for Vite. If you change the contract, update
 * BOTH files (and `schemas.py`). Wiring a single shared import (tsconfig path
 * alias + Vite `server.fs.allow`) is a possible later step.
 */

/** Identifier for a diarization model run. Open-ended: backends register new ones. */
export type ModelId = string;

/** The real, true state of one model's run — never fabricated. */
export type ModelStatus = "queued" | "running" | "done" | "failed";

/** A model's real-time GPU-residency lifecycle state, platform-wide (not tied to one evaluation). */
export type ModelLifecycleState = "unloaded" | "starting" | "ready" | "in_use" | "stopping" | "unhealthy";

/** Which half of the transcript pipeline a run is in — two RQ jobs, each timed on its own. */
export type TranscriptStage = "asr" | "align";

/** Where the ASR ran. "online" streamed the audio to a remote endpoint; "offline" kept it
 * on the host. Derived from the engine that actually ran, not from the current setting. */
export type TranscriptionMode = "online" | "offline";

/** One contiguous stretch of speech attributed to a single speaker. */
export interface DiarizationSegment {
  /** Zero-based speaker index, stable within one model run. */
  spk: number;
  /** Start time in seconds from the beginning of the audio. */
  s: number;
  /** End time in seconds (exclusive), always >= s. */
  e: number;
}

/** The complete diarization output of one model over one audio file. */
export interface DiarizationModelRun {
  id: ModelId;
  /** Full display name, e.g. "PyAnnote Audio 3.1". */
  name: string;
  /** Short label used in pills and compact rows, e.g. "PyAnnote". */
  short: string;
  /** One-line description shown in settings and model cards. */
  description: string;
  /** Speech segments, sorted by start time. */
  segs: DiarizationSegment[];
  /** Number of distinct speakers the model detected (max spk + 1). */
  numSpk: number;
  /** queued|running|done|failed — real status, not a fabricated percentage. */
  status?: ModelStatus;
  /** Failure reason when status === "failed". */
  error?: string;
  /** When the GPU supervisor granted a slot and began waiting for the model's container
   * to become healthy. Undefined for models that never needed a cold start. The interval
   * [loadingStartedAt, startedAt) is cold-start wait, not inference. */
  loadingStartedAt?: string;
  /** When inference actually began (container confirmed healthy). */
  startedAt?: string;
  /** When the worker finished running this model. */
  finishedAt?: string;
  /** Wall-clock run time in ms (excludes queue wait); Azure's own reported duration for azure-batch when available. */
  processingMs?: number;
}

/** Everything the UI needs to render one evaluation session. */
export interface DiarizationEvaluation {
  /** Primary key of the AudioFile this evaluation belongs to. */
  audioFileId: number;
  /** Audio duration in seconds. */
  durationSec: number;
  /** Client-perceived upload time (browser start -> API response). */
  uploadMs?: number;
  /** One entry per model that produced output for this audio. */
  models: DiarizationModelRun[];
}

/** One model's run on one recording, both representations side by side.
 *
 * An inspection surface, served only by `GET /evaluations/{id}/models/{modelId}/raw`
 * and deliberately absent from the polled evaluation response: raw output is the one
 * thing here that can run to megabytes. `rawOutput` is untyped because its shape is
 * whatever the engine emits — nothing outside that engine's own adapter may parse it. */
export interface ModelRawOutput {
  /** The model this run belongs to. */
  modelId: string;
  /** queued|running|done|failed. */
  status?: ModelStatus;
  /** The engine's native output, verbatim. Null when the run predates raw-output
   * persistence or did not reach completion — only a done run stores one. */
  rawOutput?: Record<string, unknown> | unknown[] | null;
  /** The adapted contract built from that raw output. */
  run?: DiarizationModelRun;
}

/** One word of the transcript, with the timing the aligner gave it. `s`/`e` are
 * undefined when the aligner could not place the word (characters outside the CTC
 * vocabulary); the word is still carried, unplaced, never dropped or guessed. */
export interface TranscriptWord {
  /** The word exactly as the ASR emitted it, for display. */
  w: string;
  /** Start time in seconds, or undefined if unaligned. */
  s?: number;
  /** End time in seconds, or undefined if unaligned. */
  e?: number;
  /** The aligner's score: a mean log-probability, so <= 0 and higher is better.
   * NOT a 0..1 confidence — do not render it as a percentage. */
  score?: number;
}

/** One engine's live-speech transcript of one audio file: ASR text plus word-level
 * timings, with each stage's real measured cost. Separate from DiarizationModelRun
 * on purpose — this is transcription, and the diarization contract has no text.
 *
 * A recording can hold one run per engine, so `GET /evaluations/{id}/transcript`
 * returns an ARRAY: the online and offline transcripts coexist and are compared. */
export interface TranscriptRun {
  audioFileId: number;
  /** queued|running|done|failed across both stages. */
  status: ModelStatus;
  /** Stage currently running, or the stage that failed; undefined once done. */
  stage?: TranscriptStage;
  /** Engine that actually ran: "hamsa" | "cohere-transcribe". */
  asrId: string;
  /** Where the ASR ran, derived from asrId — NOT from the current setting. */
  mode: TranscriptionMode;
  /** Display name of the ASR engine, e.g. "TryHamsa STT". */
  asrName: string;
  /** Display name of the forced aligner. */
  alignerName: string;
  /** Full ASR transcript; present as soon as the ASR stage finishes. */
  text: string;
  /** Empty until the alignment stage finishes. */
  words: TranscriptWord[];
  /** Measured wall-clock time of the ASR stage in ms. */
  asrMs?: number;
  /** Measured wall-clock time of the alignment stage in ms. */
  alignMs?: number;
  /** Failure reason when status === "failed". */
  error?: string;
}

/** ModelRawOutput's counterpart for one ASR engine's transcript.
 *
 * `rawOutput` is the ASR engine's native output only — hamsa's WebSocket frame log
 * or cohere's response JSON. The aligner's native output is not persisted, because
 * `run.words` already carries its per-word timings. */
export interface TranscriptRawOutput {
  /** Engine that actually ran: "hamsa" | "cohere-transcribe". */
  asrId: string;
  /** queued|running|done|failed. */
  status?: ModelStatus;
  /** The ASR engine's native output, verbatim. Null when the transcript predates
   * raw-output persistence or its ASR stage has not finished. */
  rawOutput?: Record<string, unknown> | unknown[] | null;
  /** The adapted transcript built from that raw output. */
  run?: TranscriptRun;
}

/** One model's initial state right after being enqueued. */
export interface QueuedModel {
  id: ModelId;
  status: ModelStatus;
}

/** Immediate response to `POST /upload` — the client then polls `GET /evaluations/{audioFileId}`. */
export interface UploadAck {
  audioFileId: number;
  models: QueuedModel[];
}

/** One entry in the `GET /models` registry — what a model IS, not a run's output. */
export interface ModelMetadata {
  id: ModelId;
  name: string;
  short: string;
  description: string;
  /** False for engines that are pluggable stubs (report failed, never fake segments). */
  available: boolean;
}

/** One entry in `GET /models/status` — a model's real-time GPU-residency lifecycle
 * state, platform-wide (shared across every evaluation, not scoped to one upload).
 * Models with no container to manage (e.g. pyannote, which runs in-process) have no
 * entry — never fabricate a state for them. */
export interface ModelContainerStatus {
  modelId: ModelId;
  state: ModelLifecycleState;
  /** In-flight jobs on this model; > 0 means untouchable by eviction/idle-unload. */
  activeJobCount: number;
  /** Jobs waiting on this model because the residency cap is full. */
  queuedJobCount: number;
  /** Most recent failure reason, if any (e.g. a failed cold start). */
  lastError?: string;
}
