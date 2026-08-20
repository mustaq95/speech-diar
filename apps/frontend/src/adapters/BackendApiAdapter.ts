/// <reference types="vite/client" />
import type {
  DiarizationEvaluation,
  GeneratedScript,
  ModelContainerStatus,
  RecordingSummary,
  RecordingSurface,
  ModelMetadata,
  ScriptRequest,
  TranscriptReference,
  TranscriptRun,
  TranscriptionMode,
  UploadAck,
} from "../types/diarization";
import { normalizeModelRun, type DiarizationAdapter } from "./DiarizationAdapter";

/**
 * Adapter for the platform's own backend API (apps/backend_api). The wire
 * format is already the unified contract, so adaptation is re-validation —
 * the same seam every source goes through, keeping the UI honest even if the
 * backend drifts.
 */
export const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? "/api";

export type BackendEvaluationPayload = DiarizationEvaluation;

export const BackendApiAdapter: DiarizationAdapter<BackendEvaluationPayload> = {
  source: "backend-api",
  adapt(raw) {
    return {
      audioFileId: raw.audioFileId,
      durationSec: raw.durationSec,
      uploadMs: raw.uploadMs,
      models: raw.models.map((model) => normalizeModelRun(model)),
    };
  },
};

async function errorDetail(response: Response): Promise<string> {
  const parsed = await response.json().catch(() => null);
  if (parsed && typeof parsed === "object" && "detail" in parsed) {
    const detail = (parsed as { detail: unknown }).detail;
    // FastAPI validation errors (422) return `detail` as an array of objects,
    // not a string — stringify it so callers never surface "[object Object]".
    if (detail != null) return typeof detail === "string" ? detail : JSON.stringify(detail);
  }
  return `Request failed with status ${response.status}`;
}

/** One transcription mode's engine name and whether it is usable on this host. */
export interface ModeAvailability {
  asrName: string;
  /** False when this mode's engine has no credentials/endpoint on this host. */
  configured: boolean;
}

/**
 * Runtime settings the frontend can pick up without a rebuild (see backend
 * `GET /config`).
 *
 * `transcriptionModes` describes each mode's engine and availability so the
 * Live Speech toggle can offer both and disable an unconfigured side. It says
 * nothing about what produced an existing transcript — a transcript records its
 * own engine, and none of this may relabel it. `defaultTranscriptionMode` only
 * seeds the toggle's initial position.
 */
/** One engine in the transcript comparison, as the host reports it.
 *
 * `transport` is not decoration: a streaming engine and a chunked one are not
 * measuring the same thing, so every figure rendered for an engine has to be
 * labelled with it. */
export interface TranscriptEngineInfo {
  asrId: string;
  name: string;
  mode: TranscriptionMode;
  transport: "stream" | "chunks";
  configured: boolean;
}

/** Everything the transcript surface would otherwise hardcode. All of it comes
 * from .env via GET /config, so none of these are literals in the frontend. */
export interface TranscriptConfig {
  engines: TranscriptEngineInfo[];
  recordSampleRate: number;
  /** Samples per microphone callback. Smaller means PCM reaches a streaming
   * engine sooner; the waveform's render rate is independent of it. */
  recordBlockSamples: number;
  chunkIntervalSec: number;
  chunkIntervalMinSec: number;
  chunkIntervalMaxSec: number;
  /** Deadline for a streaming engine's socket to open. A dropped upgrade neither
   * opens nor errors, so without this the recorder would wait forever. */
  socketOpenTimeoutSec: number;
  scriptLengthsMin: number[];
  scriptLanguageMixes: string[];
  scriptHardCases: string[];
  /** Null when no script gateway is configured: the UI then offers only the
   * paste-a-reference path, instead of a Generate button that cannot work. */
  scriptModel: string | null;
}

export interface RuntimeConfig {
  pollIntervalMs: number;
  defaultTranscriptionMode: TranscriptionMode;
  transcriptionModes: Record<TranscriptionMode, ModeAvailability>;
  transcript: TranscriptConfig;
}

export async function fetchRuntimeConfig(): Promise<RuntimeConfig> {
  const response = await fetch(`${API_BASE_URL}/config`);
  if (!response.ok) throw new Error(await errorDetail(response));
  return response.json();
}

/** The honest model registry: real engines, real availability — no fabricated capability. */
export async function fetchModelCatalog(): Promise<ModelMetadata[]> {
  const response = await fetch(`${API_BASE_URL}/models`);
  if (!response.ok) throw new Error(await errorDetail(response));
  return response.json();
}

/**
 * Real-time GPU-residency lifecycle state for every supervisor-managed
 * model, platform-wide (not scoped to one evaluation). Polled continuously
 * by App.tsx on its own loop — unlike the evaluation poll, this doesn't
 * stop just because the currently-loaded evaluation has no in-flight
 * models, since a model's residency can change from a different
 * evaluation's job entirely.
 */
export async function fetchModelStatus(): Promise<ModelContainerStatus[]> {
  const response = await fetch(`${API_BASE_URL}/models/status`);
  if (!response.ok) throw new Error(await errorDetail(response));
  return response.json();
}

/**
 * Upload audio and get an immediate ack (queued models); `onProgress`
 * reports real bytes-sent progress from the browser's own upload transfer —
 * never a simulated percentage.
 */
export function uploadAudio(file: File, modelIds: string[], onProgress?: (pct: number) => void): Promise<UploadAck> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE_URL}/upload?models=${encodeURIComponent(modelIds.join(","))}`);
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) onProgress((event.loaded / event.total) * 100);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText) as UploadAck);
        return;
      }
      const detail = (() => {
        try {
          return JSON.parse(xhr.responseText)?.detail;
        } catch {
          return null;
        }
      })();
      reject(new Error(detail ?? `Upload failed with status ${xhr.status}`));
    };
    xhr.onerror = () => reject(new Error("Upload failed: network error"));
    xhr.send(form);
  });
}

/**
 * Pull a recording from an external stream URL (production ADEO or the local
 * replica) instead of a browser file. The backend does the fetch, so there's
 * no client-side byte progress. `token`, when set, was parsed from a pasted
 * curl and overrides the backend's RECORDING_API_TOKEN.
 */
export async function ingestRecording(url: string, token: string | undefined, modelIds: string[]): Promise<UploadAck> {
  const response = await fetch(`${API_BASE_URL}/upload/blob`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, token, models: modelIds.join(",") }),
  });
  if (!response.ok) throw new Error(await errorDetail(response));
  return response.json();
}

/**
 * Accept either a bare stream URL or a full `curl` command pasted from Postman.
 * Pulls out the first http(s) token as the URL and a `Bearer` token from an
 * `Authorization` header if present. A bare URL yields `{ url }` with no token.
 */
export function parseBlobInput(raw: string): { url: string; token?: string } {
  const trimmed = raw.trim();
  const url = trimmed.match(/https?:\/\/[^\s"']+/)?.[0] ?? "";
  const token = trimmed.match(/[Bb]earer\s+([^\s"']+)/)?.[1];
  return token ? { url, token } : { url };
}

/** Poll target — the real, current state of one evaluation. */
export async function fetchEvaluation(audioFileId: number): Promise<DiarizationEvaluation> {
  const response = await fetch(`${API_BASE_URL}/evaluations/${audioFileId}`);
  if (!response.ok) throw new Error(await errorDetail(response));
  return BackendApiAdapter.adapt(await response.json());
}

/** Persist the client-measured upload time once, right after the upload finishes. */
export async function patchUploadTiming(audioFileId: number, uploadMs: number): Promise<DiarizationEvaluation> {
  const response = await fetch(`${API_BASE_URL}/evaluations/${audioFileId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ uploadMs }),
  });
  if (!response.ok) throw new Error(await errorDetail(response));
  return BackendApiAdapter.adapt(await response.json());
}

/** Requeue one model against the audio already uploaded, leaving the other models' runs alone. */
export async function retryModel(audioFileId: number, modelId: string): Promise<DiarizationEvaluation> {
  const response = await fetch(`${API_BASE_URL}/evaluations/${audioFileId}/models/${encodeURIComponent(modelId)}/retry`, {
    method: "POST",
  });
  if (!response.ok) throw new Error(await errorDetail(response));
  return BackendApiAdapter.adapt(await response.json());
}

/** Delete a recording and all its backend evidence (DB rows + stored audio).
 * Idempotent: a 404 means it's already gone, which is success from the caller's
 * point of view (e.g. a stale localStorage card whose row was already removed). */
export async function deleteEvaluation(audioFileId: number): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/evaluations/${audioFileId}`, { method: "DELETE" });
  if (response.status === 404) return;
  if (!response.ok) throw new Error(await errorDetail(response));
}

/**
 * Every live-speech transcript for one recording — at most one per ASR engine,
 * so an online and an offline run coexist and the panel can switch between
 * them. Empty when none was ever started: a recording predating the feature, or
 * a host with no ASR engine configured. That is a real state the panel renders
 * as an offer to run one, not an error.
 *
 * Deliberately its own request rather than a field on the evaluation: a long
 * recording's word list is hundreds of KB, and the evaluation is polled every
 * pollIntervalMs while models are in flight.
 */
export async function fetchTranscripts(audioFileId: number): Promise<TranscriptRun[]> {
  const response = await fetch(`${API_BASE_URL}/evaluations/${audioFileId}/transcript`);
  if (!response.ok) throw new Error(await errorDetail(response));
  return response.json();
}

/**
 * Run or re-run the transcript for audio already uploaded, in the chosen mode.
 * The backend resolves that mode's engine and keys the row on it, so running
 * the other mode ADDS a transcript rather than replacing the existing one; only
 * re-running the same mode overwrites. Returns just the run it started.
 */
export async function startTranscript(audioFileId: number, mode: TranscriptionMode): Promise<TranscriptRun> {
  const response = await fetch(`${API_BASE_URL}/evaluations/${audioFileId}/transcript`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  });
  if (!response.ok) throw new Error(await errorDetail(response));
  return response.json();
}

/** One origin for playback and waveform decoding — the API streams from whichever lane owns the audio. */
/** Every recording on one surface, newest first.
 *
 * One request for the whole list. It replaces a pattern where the page assembled its
 * list from localStorage and then fetched each recording's full evaluation to refresh
 * the counts — N requests to render one page, showing whatever a particular browser
 * happened to remember rather than what exists. */
export async function fetchRecordings(surface: RecordingSurface): Promise<RecordingSummary[]> {
  const response = await fetch(`${API_BASE_URL}/recordings?surface=${surface}`);
  if (!response.ok) throw new Error(await errorDetail(response));
  return (await response.json()) as RecordingSummary[];
}

/** Generate a script to read aloud, which becomes the scoring reference. */
export async function generateScript(request: ScriptRequest): Promise<GeneratedScript> {
  const response = await fetch(`${API_BASE_URL}/transcript/script`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok) throw new Error(await errorDetail(response));
  return (await response.json()) as GeneratedScript;
}

/** What a freshly opened live session reports back about itself. */
export interface LiveSessionAck {
  sessionId: string;
  asrIds: string[];
  chunkIntervalSec: number;
  sampleRate: number;
  engines: Array<{ asrId: string; name: string; transport: "stream" | "chunks" }>;
}

/** Open a live read-aloud session. The transcripts accumulate server-side, next
 * to the calls that produce them — see the backend router for why. */
export async function openLiveSession(
  asrIds: string[],
  chunkIntervalSec: number,
  referenceText: string,
): Promise<LiveSessionAck> {
  const response = await fetch(`${API_BASE_URL}/transcript/session`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ asrIds, chunkIntervalSec, referenceText }),
  });
  if (!response.ok) throw new Error(await errorDetail(response));
  return (await response.json()) as LiveSessionAck;
}

/** Send one chunk to a request/response engine. The returned latency is the same
 * value the server records, so the panel and the scorecard cannot disagree. */
export async function sendLiveChunk(
  sessionId: string,
  asrId: string,
  chunkIndex: number,
  wav: Blob,
): Promise<{ text: string; latencyMs: number; chunkIndex: number }> {
  const form = new FormData();
  form.append("sessionId", sessionId);
  form.append("asrId", asrId);
  form.append("chunkIndex", String(chunkIndex));
  form.append("file", wav, `chunk${chunkIndex}.wav`);
  const response = await fetch(`${API_BASE_URL}/transcript/chunk`, { method: "POST", body: form });
  if (!response.ok) throw new Error(await errorDetail(response));
  return await response.json();
}

/** WebSocket URL for a streaming engine's relay. Derived from API_BASE_URL so it
 * follows the same origin and dev proxy as every other call. */
export function liveStreamUrl(sessionId: string, asrId: string): string {
  const base = API_BASE_URL.startsWith("http")
    ? API_BASE_URL
    : `${window.location.origin}${API_BASE_URL}`;
  return `${base.replace(/^http/, "ws")}/transcript/live/${sessionId}/${encodeURIComponent(asrId)}`;
}

/** Persist a finished session: the recording, its reference, and each engine's
 * captured transcript with its measured timings and scores. Nothing is re-run. */
export async function finalizeLiveSession(
  sessionId: string,
  recording: Blob,
  referenceText: string,
  referenceSource: "script" | "pasted",
  scriptParams: Record<string, unknown> | null,
): Promise<TranscriptRun[]> {
  const form = new FormData();
  form.append("file", recording, "read-aloud.wav");
  form.append("referenceText", referenceText);
  form.append("referenceSource", referenceSource);
  if (scriptParams) form.append("scriptParams", JSON.stringify(scriptParams));
  const response = await fetch(`${API_BASE_URL}/transcript/session/${sessionId}/finalize`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) throw new Error(await errorDetail(response));
  return (await response.json()) as TranscriptRun[];
}

/** The reference a recording is scored against, or null when it has none. */
export async function fetchReference(audioFileId: number): Promise<TranscriptReference | null> {
  const response = await fetch(`${API_BASE_URL}/evaluations/${audioFileId}/reference`);
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(await errorDetail(response));
  return (await response.json()) as TranscriptReference;
}

/** Set the reference for a recording that already exists. Existing scores are
 * cleared by the backend rather than recomputed, so nothing shows a stale WER. */
export async function putReference(
  audioFileId: number,
  text: string,
  source: "script" | "pasted" = "pasted",
): Promise<TranscriptReference> {
  const response = await fetch(`${API_BASE_URL}/evaluations/${audioFileId}/reference`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, source }),
  });
  if (!response.ok) throw new Error(await errorDetail(response));
  return (await response.json()) as TranscriptReference;
}

/** Run several engines over stored audio — one job per engine, all-or-nothing on
 * validation so a typo cannot leave half a comparison running. */
export async function startTranscripts(
  audioFileId: number,
  asrIds: string[],
): Promise<TranscriptRun[]> {
  const response = await fetch(`${API_BASE_URL}/evaluations/${audioFileId}/transcripts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ asrIds }),
  });
  if (!response.ok) throw new Error(await errorDetail(response));
  return (await response.json()) as TranscriptRun[];
}

export function audioStreamUrl(audioFileId: number): string {
  return `${API_BASE_URL}/evaluations/${audioFileId}/audio`;
}
