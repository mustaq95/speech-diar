import type { DiarizationEvaluation } from "../types/diarization";
import { DEMO_AUDIO_FILE_ID, normalizeModelRun, type DiarizationAdapter } from "./DiarizationAdapter";

/**
 * "Model A" is a hypothetical backend that speaks a completely different
 * dialect from our contract: speaker identities are string tags instead of
 * indices, timestamps are VTT-style "HH:MM:SS.mmm --> HH:MM:SS.mmm" windows
 * instead of numeric seconds, and metadata is nested under `engine`/`audio`.
 * This is the template for wiring up any future real backend format.
 */
export interface ModelARawOutput {
  engine: {
    label: string;
    build: string;
  };
  audio: {
    /** Total length as an "HH:MM:SS.mmm" timecode string. */
    length: string;
  };
  transcript_turns: Array<{
    /** Opaque speaker tag, e.g. "SPK_A" — not a numeric index. */
    speaker_tag: string;
    /** "HH:MM:SS.mmm --> HH:MM:SS.mmm" time window. */
    window: string;
    confidence: number;
  }>;
}

function timecodeToSeconds(timecode: string): number {
  const [h, m, s] = timecode.trim().split(":");
  return Number(h) * 3600 + Number(m) * 60 + Number(s);
}

export const ModelAAdapter: DiarizationAdapter<ModelARawOutput> = {
  source: "model-a",
  adapt(raw) {
    // Speaker tags become zero-based indices in order of first appearance.
    const speakerIndex = new Map<string, number>();
    const indexFor = (tag: string): number => {
      let index = speakerIndex.get(tag);
      if (index === undefined) {
        index = speakerIndex.size;
        speakerIndex.set(tag, index);
      }
      return index;
    };

    const segs = raw.transcript_turns.map((turn) => {
      const [start, end] = turn.window.split("-->").map(timecodeToSeconds);
      return { spk: indexFor(turn.speaker_tag), s: start, e: end };
    });

    return {
      audioFileId: DEMO_AUDIO_FILE_ID,
      durationSec: timecodeToSeconds(raw.audio.length),
      models: [
        normalizeModelRun({
          id: "model-a",
          name: raw.engine.label,
          short: "Model A",
          description: `Experimental engine · build ${raw.engine.build}`,
          segs,
          status: "done",
        }),
      ],
    };
  },
};

/** Sample payload in Model A's native dialect, for exercising the adapter. */
export const SAMPLE_MODEL_A_OUTPUT: ModelARawOutput = {
  engine: { label: "Acme DiarizeNet", build: "2026.06-rc2" },
  audio: { length: "00:04:15.000" },
  transcript_turns: [
    { speaker_tag: "SPK_A", window: "00:00:08.000 --> 00:00:40.000", confidence: 0.94 },
    { speaker_tag: "SPK_B", window: "00:00:38.000 --> 00:01:12.000", confidence: 0.91 },
    { speaker_tag: "SPK_A", window: "00:01:12.000 --> 00:01:50.000", confidence: 0.9 },
    { speaker_tag: "SPK_C", window: "00:01:48.000 --> 00:02:20.000", confidence: 0.88 },
    { speaker_tag: "SPK_B", window: "00:02:20.000 --> 00:03:05.000", confidence: 0.92 },
    { speaker_tag: "SPK_A", window: "00:03:05.000 --> 00:03:40.000", confidence: 0.93 },
    { speaker_tag: "SPK_C", window: "00:03:38.000 --> 00:04:15.000", confidence: 0.89 },
  ],
};

/** Convenience loader binding the adapter to the sample payload. */
export function loadModelAEvaluation(): DiarizationEvaluation {
  return ModelAAdapter.adapt(SAMPLE_MODEL_A_OUTPUT);
}
