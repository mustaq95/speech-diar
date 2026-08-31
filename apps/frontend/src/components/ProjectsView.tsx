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

/** The line under a recording's name: when it was made, how long it is, and the
 * one figure that describes it on this surface.
 *
 * A transcript recording has no speakers to detect and no models to count; what
 * it has is engines compared and, once a reference exists, an error rate.
 * Showing "0 speakers detected" would read as a failure rather than as a
 * category that does not apply.
 *
 * A saved script has no audio at all, so its duration is not 0:00 — it is
 * nothing that was ever measured, and printing a clock there would be inventing
 * a figure.
 *
 * Exported because ReportPicker labels the same recordings and must say the same
 * thing about them; two copies of this would drift.
 */
export function projectSubtitle(project: Project, isTranscript: boolean): string {
  const head = project.date + (project.hasAudio ? ` · ${project.duration}` : "");
  if (!project.hasAudio) {
    return `${head} · ${project.ttsCount > 0
      ? `TTS · ${project.ttsCount} clip${project.ttsCount === 1 ? "" : "s"}`
      : "script saved, not recorded yet"}`;
  }
  if (isTranscript) {
    return `${head} · ${project.scored && project.bestWer != null
      ? `best WER ${(project.bestWer * 100).toFixed(1)}%`
      : "not scored"}`;
  }
  return `${head} · ${project.speakers} speakers detected`;
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
          {/* Opens the recording picker rather than reporting on everything: an
              aggregate is total errors over total reference words, so one junk
              take moves the headline number and there has to be a way to leave it
              out. Both surfaces have a report, but they are different documents:
              the diarization one is descriptive (no ground truth to score
              against), the transcript one scores accuracy and breaks it down by
              language. App picks by surface. */}
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
              <small>{projectSubtitle(project, isTranscript)}</small>
            </span>
            <span className="project-colors">
              {SPEAKER_COLORS.slice(0, isTranscript ? project.engines : project.models).map((color) => (
                <i key={color} style={{ background: color }} />
              ))}
            </span>
            <span className="muted">
              {!project.hasAudio
                ? "read it aloud"
                : isTranscript
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
