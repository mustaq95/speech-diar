import type { DiarizationModelRun, DiarizationSegment, ModelId } from "./types/diarization";

export type { DiarizationEvaluation, ModelId, ModelMetadata } from "./types/diarization";

export type Nav = "dashboard" | "upload" | "projects" | "settings";
/** Upload tab sub-state; "loading" also covers opening a saved project from the Dashboard tab. */
export type Workflow = "idle" | "uploading" | "loading" | "processing";
export type Metric = "DER" | "JER" | "WDER";

// The UI-facing data types are aliases of the unified diarization contract:
// components only ever see data that came through an adapter.
export type Segment = DiarizationSegment;
export type ModelRun = DiarizationModelRun;

export interface Project {
  id: number;
  /** Backend AudioFile id — lets reopening a project re-fetch its real evaluation. */
  audioFileId: number;
  name: string;
  date: string;
  duration: string;
  models: number;
  speakers: number;
  fresh?: boolean;
}

export interface EvalConfig {
  reference: string;
  baseline: ModelRun["id"];
  collar: number;
  ignoreOverlap: boolean;
  metric: Metric;
}

export interface ModelParams {
  min: number;
  max: number;
  thr: number;
  ovl: boolean;
}

export type ParamMap = Record<ModelId, ModelParams>;
export type ActiveMap = Record<ModelId, boolean>;
/** Which catalog model ids are implemented, from `GET /models`; unavailable ids are shown but their toggle is disabled. */
export type AvailableMap = Record<ModelId, boolean>;
