import type { ModelContainerStatus } from "../types/diarization";
import type { ModelMetadata } from "../types";
import { LifecycleChip } from "./LifecycleChip";

interface ModelStatusStripProps {
  catalog: ModelMetadata[];
  status: ModelContainerStatus[];
}

/**
 * Always-visible, platform-wide strip showing every GPU-supervisor-managed
 * model's real-time residency state (loading / ready / running inference /
 * unloaded / unhealthy) — deliberately independent of which evaluation, if
 * any, is currently loaded, since a model's residency can change because of
 * a *different* evaluation's job entirely.
 *
 * Models with no container to manage (e.g. pyannote, which runs in-process
 * in the worker — see apps/background_worker/supervisor/registry.py) have
 * no `ModelContainerStatus` entry. Rather than silently omitting them
 * (which reads as a bug/gap, not a deliberate distinction), they get a
 * distinct "In-process" chip — not a fabricated lifecycle state, just an
 * honest label that this model works differently.
 */
export function ModelStatusStrip({ catalog, status }: ModelStatusStripProps) {
  const managedIds = new Set(status.map((entry) => entry.modelId));
  const inProcessModels = catalog.filter((model) => model.available && !managedIds.has(model.id));

  if (status.length === 0 && inProcessModels.length === 0) return null;
  const nameById = new Map(catalog.map((model) => [model.id, model.short || model.name]));

  return (
    <div className="model-status-strip" aria-label="Model GPU status">
      {status.map((entry) => (
        <LifecycleChip
          key={entry.modelId}
          state={entry.state}
          label={nameById.get(entry.modelId) ?? entry.modelId}
          detail={entry.queuedJobCount > 0 ? `${entry.queuedJobCount} waiting` : undefined}
        />
      ))}
      {inProcessModels.map((model) => (
        <LifecycleChip key={model.id} state="in_process" label={model.short || model.name} />
      ))}
    </div>
  );
}
