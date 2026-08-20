import type { Project, StudioMode } from "../types";
import { SPEAKER_COLORS } from "../data";
import { DeleteControls, WaveGlyph } from "./controls";

interface ProjectsViewProps {
  projects: Project[];
  onOpenProject: (project: Project) => void;
  onNew: () => void;
  onDelete: (project: Project) => Promise<void>;
  onGenerateReport: () => void;
  reportBusy: boolean;
  /** Which surface's recordings these are. Diarization and transcript recordings are
   * scored in different ways, so a row describes itself differently on each. */
  surface: StudioMode;
}

export function ProjectsView({
  projects,
  onOpenProject,
  onNew,
  onDelete,
  onGenerateReport,
  reportBusy,
  surface,
}: ProjectsViewProps) {
  const isTranscript = surface === "transcript";
  return (
    <main className="page narrow">
      <section className="view-head">
        <div>
          <h1>Projects</h1>
          <p>
            {projects.length} recording{projects.length === 1 ? "" : "s"} ·{" "}
            {isTranscript
              ? "read-aloud captures, scored against their reference"
              : "diarization results are saved and re-openable"}
          </p>
        </div>
        <div className="view-head-actions">
          {/* Both surfaces have a report now, but they are different documents: the
              diarization one is descriptive (no ground truth to score against), the
              transcript one scores accuracy and breaks it down by language. App
              picks by surface. */}
          <button className="ghost-btn" type="button" onClick={onGenerateReport} disabled={reportBusy || projects.length === 0}>
            {reportBusy ? "Generating…" : "Generate report"}
          </button>
          <button className="primary-btn" type="button" onClick={onNew}>+ New recording</button>
        </div>
      </section>

      <section className="project-list">
        {projects.length ? projects.map((project) => (
          <div
            key={project.audioFileId}
            role="button"
            tabIndex={0}
            className={`project-row ${project.fresh ? "is-fresh" : ""}`}
            onClick={() => onOpenProject(project)}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                onOpenProject(project);
              }
            }}
          >
            <span className="project-icon"><WaveGlyph colors={SPEAKER_COLORS} height={22} /></span>
            <span className="project-main">
              <span>
                <strong>{project.name}</strong>
                {project.fresh && <b>NEW</b>}
              </span>
              {/* A transcript recording has no speakers to detect and no models to
                  count; what it has is engines compared and, once a reference exists,
                  an error rate. Showing "0 speakers detected" would read as a failure
                  rather than as a category that does not apply. */}
              <small>
                {project.date} · {project.duration} ·{" "}
                {isTranscript
                  ? project.scored && project.bestWer != null
                    ? `best WER ${(project.bestWer * 100).toFixed(1)}%`
                    : "not scored"
                  : `${project.speakers} speakers detected`}
              </small>
            </span>
            <span className="project-colors">
              {SPEAKER_COLORS.slice(0, isTranscript ? project.engines : project.models).map((color) => (
                <i key={color} style={{ background: color }} />
              ))}
            </span>
            <span className="muted">
              {isTranscript
                ? `${project.engines} engine${project.engines === 1 ? "" : "s"}`
                : `${project.models} models`}
            </span>
            <DeleteControls onDelete={() => onDelete(project)} />
            <span className="row-arrow">›</span>
          </div>
        )) : (
          <div className="empty-list">
            {isTranscript
              ? "No read-aloud recordings yet. Generate a script on the Transcript surface and record one."
              : "No recordings yet. Upload one to get started."}
          </div>
        )}
      </section>
    </main>
  );
}
