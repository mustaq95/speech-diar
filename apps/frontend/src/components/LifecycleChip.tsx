import type { ModelLifecycleState } from "../types/diarization";

interface LifecycleChipProps {
  /** A real GPU-residency state, or "in_process" for a model with no
   * container to manage (e.g. pyannote) — not a ModelLifecycleState value,
   * since that enum only covers real, non-fabricated container states. */
  state: ModelLifecycleState | "in_process";
  /** Omit when the model name is already shown elsewhere (e.g. inline next
   * to a timeline row's own <strong> label) — the chip then shows just the
   * state dot + text, not a redundant repeated name. */
  label?: string;
  detail?: string;
}

const STATE_LABEL: Record<ModelLifecycleState, string> = {
  unloaded: "Unloaded",
  starting: "Loading",
  ready: "Ready",
  in_use: "Inference",
  stopping: "Stopping",
  unhealthy: "Unhealthy",
};

// Same semantic palette --good/--warn/--bad already used elsewhere (see
// .status-dot in index.css) -- not a new accent, just applying the
// existing good/warn/bad meanings to lifecycle state. "in_process" gets its
// own neutral treatment (no dot at all) since it isn't a real lifecycle
// state -- pyannote has no container to be loading/ready/unhealthy.
const STATE_CLASS: Record<ModelLifecycleState | "in_process", string> = {
  unloaded: "is-unloaded",
  starting: "is-warn",
  ready: "is-good",
  in_use: "is-good",
  stopping: "is-warn",
  unhealthy: "is-bad",
  in_process: "is-in-process",
};

/**
 * One model's GPU-residency (or in-process) status, rendered identically
 * wherever it appears — the top strip (ModelStatusStrip.tsx) and each
 * timeline row's gutter (Studio.tsx) both use this so the two surfaces
 * share one status vocabulary instead of drifting apart.
 */
export function LifecycleChip({ state, label, detail }: LifecycleChipProps) {
  const stateLabel = state === "in_process" ? "In-process" : STATE_LABEL[state];
  return (
    <span className={`model-status-chip ${STATE_CLASS[state]}`}>
      {state !== "in_process" && <span className="status-dot" aria-hidden="true" />}
      {label && <span className="model-status-name">{label}</span>}
      <span className="model-status-state mono">
        {stateLabel}
        {detail ? ` · ${detail}` : ""}
      </span>
    </span>
  );
}
