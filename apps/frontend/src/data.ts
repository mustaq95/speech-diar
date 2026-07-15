// Baseline mock fixture for the studio. This data defined the unified
// contract in `src/types/diarization.ts` and reaches the UI only through
// `src/adapters/MockDiarizationAdapter.ts` — never import MODELS/DURATION
// into components directly (SPEAKER_COLORS is a pure UI theme constant and
// is exempt).
import type { ActiveMap, EvalConfig, ModelRun, ParamMap } from "./types";

const withSpeakerCount = (model: Omit<ModelRun, "numSpk">): ModelRun => ({
  ...model,
  numSpk: Math.max(...model.segs.map((seg) => seg.spk)) + 1,
});

export const DURATION = 255;

export const SPEAKER_COLORS = [
  "#4C9AFF", "#F5A623", "#2DD4BF", "#F471B5", "#A3E635",
  "#B57BFF", "#FF6B6B", "#22D3EE", "#FACC15", "#34D399",
  "#E879F9", "#FB923C", "#38BDF8", "#C084FC", "#4ADE80",
  "#FB7185", "#818CF8", "#D6A76A", "#F87171", "#94A3B8",
];

export const MODELS: ModelRun[] = [
  withSpeakerCount({
    id: "pyannote",
    name: "PyAnnote Audio 3.1",
    short: "PyAnnote",
    description: "End-to-end neural diarization · v3.1",
    segs: [
      { spk: 0, s: 8, e: 40 },
      { spk: 1, s: 38, e: 72 },
      { spk: 0, s: 72, e: 110 },
      { spk: 2, s: 108, e: 140 },
      { spk: 1, s: 140, e: 185 },
      { spk: 0, s: 185, e: 220 },
      { spk: 2, s: 218, e: 255 },
    ],
  }),
  withSpeakerCount({
    id: "whisperx",
    name: "WhisperX Diarization",
    short: "WhisperX",
    description: "ASR-aligned diarization · 4-speaker capable",
    segs: [
      { spk: 0, s: 5, e: 35 },
      { spk: 1, s: 30, e: 55 },
      { spk: 2, s: 55, e: 95 },
      { spk: 1, s: 90, e: 120 },
      { spk: 3, s: 118, e: 160 },
      { spk: 0, s: 155, e: 190 },
      { spk: 2, s: 190, e: 230 },
      { spk: 3, s: 225, e: 255 },
    ],
  }),
  withSpeakerCount({
    id: "nemo",
    name: "NVIDIA NeMo Diarization",
    short: "NeMo",
    description: "MSDD multi-scale clustering",
    segs: [
      { spk: 0, s: 10, e: 45 },
      { spk: 1, s: 42, e: 58 },
      { spk: 0, s: 58, e: 100 },
      { spk: 2, s: 100, e: 135 },
      { spk: 1, s: 133, e: 175 },
      { spk: 0, s: 175, e: 215 },
      { spk: 2, s: 213, e: 255 },
    ],
  }),
  withSpeakerCount({
    id: "aws",
    name: "AWS Transcribe Diarized",
    short: "AWS",
    description: "Managed cloud transcription + diarization",
    segs: [
      { spk: 0, s: 6, e: 42 },
      { spk: 1, s: 40, e: 80 },
      { spk: 2, s: 80, e: 125 },
      { spk: 0, s: 123, e: 165 },
      { spk: 1, s: 165, e: 210 },
      { spk: 2, s: 208, e: 255 },
    ],
  }),
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
