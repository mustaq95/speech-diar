import type { DiarizationEvaluation, DiarizationModelRun } from "../types/diarization";

/**
 * Sentinel `audioFileId` for evaluations that did not come from the backend
 * (the synthetic demo sources). Real ids are Postgres serials starting at 1,
 * so 0 is never a real evaluation — callers use this to skip polling,
 * playback, and waveform decoding for demo data.
 */
export const DEMO_AUDIO_FILE_ID = 0;

/**
 * Base adapter contract — the Adapter Pattern seam of the platform.
 *
 * One adapter per backend output format. The adapter is the ONLY place allowed
 * to know a backend's native JSON shape; everything downstream of `adapt()`
 * speaks the unified contract in `src/types/diarization.ts`.
 *
 * `TRaw` is the backend's native payload type, so each adapter is fully typed
 * end to end and swapping the data source never touches the UI.
 */
export interface DiarizationAdapter<TRaw> {
  /** Stable identifier for the source format this adapter understands. */
  readonly source: string;
  /** Translate one native payload into the unified contract. Must be pure. */
  adapt(raw: TRaw): DiarizationEvaluation;
}

/**
 * Shared normalisation every adapter should run its model runs through, so the
 * UI can rely on the contract's invariants (sorted segments, correct numSpk)
 * regardless of how sloppy the source format is.
 */
export function normalizeModelRun(run: Omit<DiarizationModelRun, "numSpk">): DiarizationModelRun {
  const segs = [...run.segs].sort((a, b) => a.s - b.s);
  return {
    ...run,
    segs,
    numSpk: segs.length ? Math.max(...segs.map((seg) => seg.spk)) + 1 : 0,
  };
}
