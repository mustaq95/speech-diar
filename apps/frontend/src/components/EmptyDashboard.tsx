import { useRef } from "react";
import type { ActiveMap, AvailableMap, ModelRun, Project } from "../types";
import { WaveGlyph } from "./controls";
import { SPEAKER_COLORS } from "../data";

interface EmptyDashboardProps {
  /** The real, honest model catalog from `GET /models` (as ModelRun shells). */
  models: ModelRun[];
  available: AvailableMap;
  active: ActiveMap;
  projects: Project[];
  /** Real upload: a user-picked audio file to send through the backend. */
  onFile: (file: File) => void;
  /** Synthetic demo: canned output, never sent through the backend. */
  onLoadDemo: () => void;
  onSettings: () => void;
  onOpenProject: (project: Project) => void;
  onProjects: () => void;
}

export function EmptyDashboard({ models, available, active, projects, onFile, onLoadDemo, onSettings, onOpenProject, onProjects }: EmptyDashboardProps) {
  const shown = models.filter((model) => active[model.id]).length;
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  return (
    <main className="page page-empty">
      <section className="empty-head">
        <div>
          <div className="eyebrow">Comparison workspace</div>
          <h1>No recording loaded</h1>
          <p>Upload audio to run it through every configured model at once and compare speaker boundaries, overlaps, and model agreement.</p>
        </div>
        <div className="animated-bars" aria-hidden="true">
          {SPEAKER_COLORS.slice(0, 5).map((color, index) => (
            <span key={color} style={{ background: color, animationDelay: `${index * 0.15}s` }} />
          ))}
        </div>
      </section>

      <section className="empty-grid">
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/wav,.wav"
          style={{ display: "none" }}
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) onFile(file);
            event.target.value = "";
          }}
        />
        <div
          className="drop-zone"
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => {
            event.preventDefault();
            const file = event.dataTransfer.files?.[0];
            if (file) onFile(file);
          }}
        >
          <span className="upload-icon">↥</span>
          <strong>Drop an audio file to compare</strong>
          <span>or click to browse</span>
          <span className="drop-actions">
            <button type="button" className="action-fill" onClick={(event) => { event.stopPropagation(); fileInputRef.current?.click(); }}>Browse files</button>
            <button type="button" className="action-outline" onClick={(event) => { event.stopPropagation(); onLoadDemo(); }}>Load demo</button>
          </span>
          <small>WAV · up to 2 hours</small>
          <small className="demo-caveat">Demo uses synthetic data — not real model results</small>
        </div>

        <aside className="models-summary">
          <div className="summary-head">
            <strong>Configured models</strong>
            <button type="button" onClick={onSettings}>Configure →</button>
          </div>
          <p>{shown} of {models.length} will run on your next upload</p>
          <div>
            {models.map((model) => {
              const isAvailable = available[model.id] ?? false;
              const on = active[model.id] && isAvailable;
              return (
                <div key={model.id} className="summary-model">
                  <span className={`status-dot ${on ? "is-on" : ""}`} />
                  <span>{model.name}</span>
                  <b>{!isAvailable ? "N/A" : on ? "READY" : "OFF"}</b>
                </div>
              );
            })}
          </div>
        </aside>
      </section>

      {projects.length > 0 && (
        <section className="recent-strip">
          <div className="recent-head">
            <span className="eyebrow">Recent recordings</span>
            <button type="button" onClick={onProjects}>View all →</button>
          </div>
          <div className="recent-grid">
            {projects.slice(0, 3).map((project) => (
              <button key={project.id} type="button" className="recent-card" onClick={() => onOpenProject(project)}>
                <span className="recent-icon"><WaveGlyph colors={SPEAKER_COLORS} /></span>
                <span>
                  <strong>{project.name}</strong>
                  <small>{project.duration} · {project.models} models</small>
                </span>
                <b>›</b>
              </button>
            ))}
          </div>
        </section>
      )}
    </main>
  );
}
