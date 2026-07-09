import type { ActiveMap, ModelRun } from "../types";

interface ConfigBarProps {
  models: ModelRun[];
  active: ActiveMap;
  onToggle: (id: ModelRun["id"]) => void;
}

export function ConfigBar({ models, active, onToggle }: ConfigBarProps) {
  const shown = models.filter((model) => active[model.id]).length;
  return (
    <div className="config-bar">
      <span className="eyebrow">Models</span>
      <div className="model-pills">
        {models.map((model) => {
          const on = active[model.id];
          return (
            <button key={model.id} type="button" className={`model-pill ${on ? "is-on" : ""}`} onClick={() => onToggle(model.id)}>
              <span className="status-dot" />
              <span>{model.short}</span>
              <b>{on ? "ON" : "OFF"}</b>
            </button>
          );
        })}
      </div>
      <span className="model-count">
        {shown} of {models.length} shown
      </span>
    </div>
  );
}
