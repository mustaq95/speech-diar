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
  /** When the worker started running this model. */
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
