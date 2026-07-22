/// <reference types="vite/client" />
import type { DiarizationEvaluation, ModelContainerStatus, ModelMetadata, UploadAck } from "../types/diarization";
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

/** Runtime settings the frontend can pick up without a rebuild (see backend `GET /config`). */
export async function fetchRuntimeConfig(): Promise<{ pollIntervalMs: number }> {
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

/** One origin for playback and waveform decoding — the API streams from whichever lane owns the audio. */
export function audioStreamUrl(audioFileId: number): string {
  return `${API_BASE_URL}/evaluations/${audioFileId}/audio`;
}
