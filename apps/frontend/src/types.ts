import type { DiarizationModelRun, DiarizationSegment, ModelId } from "./types/diarization";

export type { DiarizationEvaluation, ModelId, ModelMetadata } from "./types/diarization";

export type Nav = "dashboard" | "transcript" | "upload" | "projects" | "settings";
/** Upload tab sub-state; "loading" also covers opening a saved project from the Dashboard tab. */
export type Workflow = "idle" | "uploading" | "loading" | "processing";
export type Metric = "DER" | "JER" | "WDER";

/** The two evaluation surfaces the center-of-dashboard toggle switches between.
 * Diarization is the timeline studio; Transcript is a full page, because the
 * read-aloud flow starts with no recording loaded and so cannot live inside a
 * studio pane that only exists once one is. */
export type StudioMode = "diarization" | "transcript";

/** Which sub-mode of the Transcript surface is open.
 *
 * NOT a third StudioMode: the surface toggle picks what an AudioFile row IS
 * (`AudioFile.surface` is single-valued and stays two-valued), while this picks
 * what you are doing on the transcript surface. Cloning goes further than TTS
 * did and produces no AudioFile at all — a cloned voice is a speaker name, not
 * a recording. */
export type TranscriptSubMode = "stt" | "tts" | "clone";

// The UI-facing data types are aliases of the unified diarization contract:
// components only ever see data that came through an adapter.
export type Segment = DiarizationSegment;
export type ModelRun = DiarizationModelRun;

/** One row of the recordings list.
 *
 * A thin view over `RecordingSummary` from the backend rather than a stored object:
 * the list is now fetched, so `date`/`duration` are formatted for display here and
 * the counts arrive already computed. `fresh` is the only piece of local state left —
 * it marks recordings created in THIS browser session, which is what the NEW badge
 * always actually meant.
 */
export interface Project {
  /** Backend AudioFile id. The row's identity — there is no separate local id now. */
  audioFileId: number;
  surface: StudioMode;
  name: string;
  date: string;
  /** Formatted for display. `durationSec` is the number the report sums. */
  duration: string;
  durationSec: number;
  /** Diarization: models run and highest speaker count found. */
  models: number;
  speakers: number;
  /** Transcript: engines compared, and the best WER once scored. */
  engines: number;
  scored: boolean;
  bestWer?: number;
  /** Transcript: TTS engines that have synthesized this recording's reference. */
  ttsCount: number;
  /** False for a transcript entry whose script was generated but never read aloud.
   * `duration` is meaningless for such a row: nothing was measured. */
  hasAudio: boolean;
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
