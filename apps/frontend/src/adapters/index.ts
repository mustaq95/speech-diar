import { DEFAULT_EVAL, DEFAULT_PARAMS } from "../data";
import type { ActiveMap, EvalConfig, ModelParams, ModelRun, ParamMap } from "../types";
import type { DiarizationEvaluation, ModelMetadata } from "../types/diarization";

export type { DiarizationAdapter } from "./DiarizationAdapter";
export type { RuntimeConfig } from "./BackendApiAdapter";
export { DEMO_AUDIO_FILE_ID, normalizeModelRun } from "./DiarizationAdapter";
export {
  API_BASE_URL,
  BackendApiAdapter,
  audioStreamUrl,
  deleteEvaluation,
  fetchEvaluation,
  fetchModelCatalog,
  fetchModelStatus,
  fetchRuntimeConfig,
  fetchTranscripts,
  ingestRecording,
  parseBlobInput,
  patchUploadTiming,
  retryModel,
  startTranscript,
  uploadAudio,
} from "./BackendApiAdapter";

// Defaults are derived from whatever models the backend returned, so a new
// model never requires touching the UI's state wiring.

export const FALLBACK_PARAMS: ModelParams = { min: 1, max: 8, thr: 0.65, ovl: true };

export function deriveDefaultActive(evaluation: DiarizationEvaluation): ActiveMap {
  return Object.fromEntries(evaluation.models.map((model) => [model.id, true]));
}

export function deriveDefaultParams(evaluation: DiarizationEvaluation): ParamMap {
  const presets = DEFAULT_PARAMS as Partial<ParamMap>;
  return Object.fromEntries(
    evaluation.models.map((model) => [model.id, presets[model.id] ?? FALLBACK_PARAMS]),
  );
}

export function deriveDefaultEval(evaluation: DiarizationEvaluation): EvalConfig {
  return { ...DEFAULT_EVAL, baseline: evaluation.models[0]?.id ?? DEFAULT_EVAL.baseline };
}

// Before any evaluation exists, the "which models will run next" UI (empty
// dashboard, settings) is driven by the real `GET /models` catalog instead.

/** Represent one catalog entry as a `ModelRun` shell (no segments yet) so it can flow through the same display components as a real run. */
export function modelRunFromMetadata(meta: ModelMetadata): ModelRun {
  return { id: meta.id, name: meta.name, short: meta.short, description: meta.description, segs: [], numSpk: 0 };
}

/** Reconciles a saved Settings choice with the current catalog: an
 * unavailable model (unimplemented, or excluded via server-side
 * `ENABLED_MODELS`) is always forced off regardless of what was saved, and a
 * model with no saved entry (new to the catalog) defaults to `available`. */
export function mergeActiveWithCatalog(saved: ActiveMap, catalog: ModelMetadata[]): ActiveMap {
  return Object.fromEntries(
    catalog.map((meta) => [meta.id, meta.available && (saved[meta.id] ?? meta.available)]),
  );
}
