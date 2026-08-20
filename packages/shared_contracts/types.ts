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
 * on the host. Derived from the engine that actually ran, not from the current setting.
 *
 * No longer what SELECTS an engine — asrId is. Two engines are "online" (hamsa and
 * inception-stt), so a mode cannot identify one. */
export type TranscriptionMode = "online" | "offline";

/** How a transcript was produced. "live" was captured chunk by chunk while someone read a
 * script aloud; "batch" was run over stored audio. Not interchangeable — they are different
 * measurements of the same engine. */
export type TranscriptSource = "live" | "batch";

/** What the audio travelled over for a given engine. "stream" is continuous with engine-side
 * VAD; "chunks" is fixed-interval cuts. A chunked engine carries its boundary cost inside its
 * own error rate, so every figure is labelled with this and the two transports' chunk
 * statistics are NOT comparable to each other. */
export type TranscriptTransport = "stream" | "chunks";

/** Where a reference transcript came from. "script" was generated and read aloud, so the words
 * were known before the audio existed; "pasted" was supplied by hand afterwards. */
export type ReferenceSource = "script" | "pasted";

/** One edit operation aligning hypothesis to reference. */
export type AlignmentOp = "equal" | "sub" | "del" | "ins";

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

/** One step of the reference-to-hypothesis alignment, for the word-level error highlight.
 *
 * From the WER edit-distance backtrace, NOT from any aligner or per-word timing: the boxed
 * words in the UI are substitutions against the reference, which is a text comparison. */
export interface TranscriptAlignmentOp {
  op: AlignmentOp;
  /** Reference word; undefined for an insertion. */
  ref?: string;
  /** Hypothesis word; undefined for a deletion. */
  hyp?: string;
  /** Index into the hypothesis word list, so the UI can mark the rendered word. */
  hypIndex?: number;
}

/** How one engine's transcript scored against the recording's reference.
 *
 * Both normalized and raw rates are carried so the normalization toggle is a read rather than
 * a recompute, and so normalization's own effect is visible.
 *
 * `rtf` is undefined when the transport cannot produce one: a real-time streaming protocol
 * consumes audio at 1x by definition, so a figure for it would be invented. Render
 * "real-time bound" for a missing value, never a number. */
export interface TranscriptMetrics {
  /** Word error rate on normalized text, 0..1+ (S+D+I over reference words). */
  wer: number;
  /** Character error rate on normalized text. */
  cer: number;
  /** Word error rate WITHOUT normalization. */
  werRaw: number;
  /** Character error rate WITHOUT normalization. */
  cerRaw: number;
  refWordCount: number;
  hypWordCount: number;
  subCount: number;
  delCount: number;
  insCount: number;
  /** Processing time / audio duration; undefined when the transport is real-time bound. */
  rtf?: number;
}

/** The ground truth a recording's engines are scored against. One per recording, never one per
 * engine: comparability depends on every engine being scored against the same text. */
export interface TranscriptReference {
  audioFileId: number;
  source: ReferenceSource;
  text: string;
  /** Measured from `text`, never taken from a request. */
  wordCount: number;
  /** For a generated script: the request that produced it, plus the generating model. */
  params?: Record<string, unknown> | null;
}

/** Which evaluation surface produced a recording. Diarization recordings come from an
 * upload; transcript recordings are read-aloud captures. Listed separately because they are
 * different artifacts scored in different ways. */
export type RecordingSurface = "diarization" | "transcript";

/** One row of the recordings list.
 *
 * The counts are computed by aggregate query on the server, not derived here: the list used
 * to be assembled by fetching every recording's full evaluation one request at a time, which
 * is N requests to render one page.
 *
 * Deliberately absent: `payload` and `rawOutput`. Both are deferred columns holding a whole
 * model's output, and a list has no use for either. */
export interface RecordingSummary {
  audioFileId: number;
  surface: RecordingSurface;
  /** Display name; not unique. */
  filename: string;
  durationSec: number;
  /** ISO-8601; the list is ordered by this, newest first. */
  createdAt: string;

  // --- diarization recordings ---
  /** Models that have a result row. */
  modelCount: number;
  /** Highest speaker count any model found. */
  speakerCount: number;
  doneCount: number;
  failedCount: number;

  // --- transcript recordings ---
  /** ASR engines compared. */
  engineCount: number;
  /** Whether a reference exists and runs were scored. */
  scored: boolean;
  /** Lowest WER across engines; undefined when unscored. */
  bestWer?: number;
}

/** What to generate a read-aloud script for. The options are served by GET /config from .env,
 * so the slider stops and chips are not literals here. */
export interface ScriptRequest {
  /** Target read-aloud length in minutes. */
  minutes: number;
  /** One of GET /config transcript.scriptLanguageMixes. */
  languageMix: string;
  /** Subset of GET /config transcript.scriptHardCases. */
  hardCases: string[];
}

/** A generated script, with the request that produced it.
 *
 * `wordCount` is MEASURED from `text`, never echoed from the request: it becomes the WER
 * denominator, so a requested length must never be mistaken for a produced one. */
export interface GeneratedScript {
  text: string;
  /** Measured from `text`. */
  wordCount: number;
  /** The model that actually generated it. */
  generatorModel: string;
  /** The request, plus finish reason and generation time. */
  params: Record<string, unknown>;
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

  // --- transcript evaluation: how this run was produced, and how it scored ---
  /** live (read-aloud capture) or batch (stored audio). */
  source: TranscriptSource;
  /** stream or chunks — label every figure with it; the two are not comparable. */
  transport?: TranscriptTransport;
  /** Cut interval for a chunked transport; undefined for a stream. */
  chunkIntervalSec?: number;
  chunkCount?: number;
  /** Measured latency of the first chunk/frame: the panel's "lag". */
  firstLatencyMs?: number;
  /** Mean measured chunk latency — the comparable headline figure. */
  avgLatencyMs?: number;
  /** Undefined until this recording has a reference and the run has finished. */
  metrics?: TranscriptMetrics;
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

/** Body of `PATCH /evaluations/{audioFileId}` — the client's one-time measured upload time,
 * persisted so it survives reopening the project. */
export interface UploadTimingUpdate {
  uploadMs: number;
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
