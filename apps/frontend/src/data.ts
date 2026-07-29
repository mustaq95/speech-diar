// UI constants shared across the studio: the speaker color theme and the
// default Settings/eval presets applied before any real evaluation exists.
import type { ActiveMap, EvalConfig, ParamMap } from "./types";

export const SPEAKER_COLORS = [
  "#4C9AFF", "#F5A623", "#2DD4BF", "#F471B5", "#A3E635",
  "#B57BFF", "#FF6B6B", "#22D3EE", "#FACC15", "#34D399",
  "#E879F9", "#FB923C", "#38BDF8", "#C084FC", "#4ADE80",
  "#FB7185", "#818CF8", "#D6A76A", "#F87171", "#94A3B8",
];

export const DEFAULT_ACTIVE: ActiveMap = {
  pyannote: true,
  whisperx: true,
  nemo: true,
  aws: true,
};

export const DEFAULT_EVAL: EvalConfig = {
  reference: "Uploaded RTTM",
  baseline: "pyannote",
  collar: 250,
  ignoreOverlap: true,
  metric: "DER",
};

export const DEFAULT_PARAMS: ParamMap = {
  pyannote: { min: 1, max: 8, thr: 0.7, ovl: true },
  whisperx: { min: 1, max: 8, thr: 0.65, ovl: true },
  nemo: { min: 1, max: 8, thr: 0.72, ovl: false },
  aws: { min: 1, max: 10, thr: 0.6, ovl: true },
};
