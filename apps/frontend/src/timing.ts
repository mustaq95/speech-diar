import type { ModelRun } from "./types";

/** Milliseconds since `startedAt` (an ISO timestamp), or null if not started. */
export function elapsedMs(startedAt: string | undefined, now: number): number | null {
  if (!startedAt) return null;
  return Math.max(0, now - new Date(startedAt).getTime());
}

function fmtSeconds(ms: number): string {
  return `${(ms / 1000).toFixed(1)}s`;
}

/**
 * Human label for one model's real status — never a fabricated percentage.
 * `now` is passed in (rather than read via Date.now()) so callers control
 * the re-render cadence for the "running" live-elapsed case.
 */
export function modelTimingLabel(model: ModelRun, durationSec: number, now: number): string {
  switch (model.status) {
    case "queued":
      return "Queued";
    case "running": {
      const ms = elapsedMs(model.startedAt, now);
      return ms == null ? "Running" : `Running · ${fmtSeconds(ms)} elapsed`;
    }
    case "failed": {
      const ms = model.processingMs;
      const reason = model.error ?? "unknown error";
      return ms == null ? `Failed: ${reason}` : `Failed after ${fmtSeconds(ms)}: ${reason}`;
    }
    case "done": {
      if (model.processingMs == null) return "Done";
      const rtf = durationSec > 0 ? model.processingMs / 1000 / durationSec : null;
      return rtf == null
        ? `Done · ${fmtSeconds(model.processingMs)}`
        : `Done · ${fmtSeconds(model.processingMs)} · ${rtf.toFixed(2)}x realtime`;
    }
    default:
      // No status at all (the synthetic demo source): already-complete canned output.
      return "";
  }
}

export function isSettled(model: ModelRun): boolean {
  return model.status === "done" || model.status === "failed" || model.status === undefined;
}

export function isInFlight(model: ModelRun): boolean {
  return model.status === "queued" || model.status === "running";
}
