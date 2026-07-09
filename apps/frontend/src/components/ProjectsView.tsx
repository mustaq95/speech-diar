import type { Project } from "../types";
import { SPEAKER_COLORS } from "../data";
import { WaveGlyph } from "./controls";

interface ProjectsViewProps {
  projects: Project[];
  onOpenProject: (project: Project) => void;
  onNew: () => void;
}

export function ProjectsView({ projects, onOpenProject, onNew }: ProjectsViewProps) {
  return (
    <main className="page narrow">
      <section className="view-head">
        <div>
          <h1>Projects</h1>
          <p>{projects.length} recording{projects.length === 1 ? "" : "s"} · diarization results are saved and re-openable</p>
        </div>
        <button className="primary-btn" type="button" onClick={onNew}>+ New recording</button>
      </section>

      <section className="project-list">
        {projects.length ? projects.map((project) => (
          <button key={project.id} type="button" className={`project-row ${project.fresh ? "is-fresh" : ""}`} onClick={() => onOpenProject(project)}>
            <span className="project-icon"><WaveGlyph colors={SPEAKER_COLORS} height={22} /></span>
            <span className="project-main">
              <span>
                <strong>{project.name}</strong>
                {project.fresh && <b>NEW</b>}
              </span>
              <small>{project.date} · {project.duration} · {project.speakers} speakers detected</small>
            </span>
            <span className="project-colors">
              {SPEAKER_COLORS.slice(0, project.models).map((color) => <i key={color} style={{ background: color }} />)}
            </span>
            <span className="muted">{project.models} models</span>
            <span className="row-arrow">›</span>
          </button>
        )) : (
          <div className="empty-list">No recordings yet. Upload one to get started.</div>
        )}
      </section>
    </main>
  );
}
