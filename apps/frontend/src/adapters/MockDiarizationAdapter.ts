import { DURATION, MODELS } from "../data";
import type { DiarizationEvaluation, DiarizationModelRun } from "../types/diarization";
import { DEMO_AUDIO_FILE_ID, normalizeModelRun, type DiarizationAdapter } from "./DiarizationAdapter";

/**
 * Native shape of the bundled mock fixture (`src/data.ts`). The unified
 * contract was derived from this data, so adaptation is mostly re-validation:
 * the adapter exists so the mock source flows through the exact same seam a
 * real backend will, and stays honest if the contract evolves.
 */
export interface MockRawPayload {
  durationSec: number;
  models: DiarizationModelRun[];
}

export const MockDiarizationAdapter: DiarizationAdapter<MockRawPayload> = {
  source: "mock",
  adapt(raw) {
    return {
      audioFileId: DEMO_AUDIO_FILE_ID,
      durationSec: raw.durationSec,
      // Canned output represents an already-completed run.
      models: raw.models.map((model) => normalizeModelRun({ ...model, status: "done" })),
    };
  },
};

/** Convenience loader binding the adapter to the bundled fixture. */
export function loadMockEvaluation(): DiarizationEvaluation {
  return MockDiarizationAdapter.adapt({ durationSec: DURATION, models: MODELS });
}
