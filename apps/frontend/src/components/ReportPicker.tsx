import { useEffect, useRef, useState } from "react";
import type { Project, StudioMode } from "../types";
import { projectSubtitle } from "./ProjectsView";

/** Pick which recordings go into a report.
 *
 * The only modal in the app, and deliberately so — `DeleteControls` states the
 * inline-confirm preference and it still holds for one boolean on one row. This
 * is a multi-row selection with nowhere to live inline. Do not read it as a new
 * general pattern.
 *
 * It owns its own list rather than putting checkboxes on the Projects rows,
 * which keeps the row's click-to-open and its Space handler untouched.
 *
 * Everything starts ticked, so the old behaviour (report on all of it) is still
 * two clicks away and the picker is a narrowing step rather than a new chore.
 * Rows that cannot contribute — a script never read aloud, an unscored capture —
 * are listed and tickable with their real status showing. The popup reports what
 * exists; deciding what belongs in the report is the operator's call.
 */
interface ReportPickerProps {
  projects: Project[];
  surface: StudioMode;
  busy: boolean;
  onCancel: () => void;
  onGenerate: (selected: Project[]) => void;
}

export function ReportPicker({ projects, surface, busy, onCancel, onGenerate }: ReportPickerProps) {
  const isTranscript = surface === "transcript";
  const [selected, setSelected] = useState<Set<number>>(
    () => new Set(projects.map((project) => project.audioFileId)),
  );
  const panelRef = useRef<HTMLDivElement>(null);

  // Escape closes, same as VoiceSelect. Not while a report is being built: the
  // fetches would keep running with nothing left to receive them.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !busy) onCancel();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [busy, onCancel]);

  useEffect(() => {
    panelRef.current?.focus();
  }, []);

  const toggle = (audioFileId: number) => {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(audioFileId)) next.delete(audioFileId);
      else next.add(audioFileId);
      return next;
    });
  };

  const chosen = projects.filter((project) => selected.has(project.audioFileId));

  return (
    <div
      className="picker-backdrop"
      onClick={(event) => {
        if (event.target === event.currentTarget && !busy) onCancel();
      }}
    >
      <div className="picker-panel" ref={panelRef} tabIndex={-1} role="dialog" aria-modal="true" aria-labelledby="picker-title">
        <header className="picker-head">
          <div>
            <h2 id="picker-title">Recordings in report</h2>
            <p>{chosen.length} of {projects.length} selected</p>
          </div>
          <div className="picker-bulk">
            <button
              className="ghost-btn"
              type="button"
              disabled={busy || chosen.length === projects.length}
              onClick={() => setSelected(new Set(projects.map((project) => project.audioFileId)))}
            >
              Select all
            </button>
            <button
              className="ghost-btn"
              type="button"
              disabled={busy || chosen.length === 0}
              onClick={() => setSelected(new Set())}
            >
              Clear
            </button>
          </div>
        </header>

        <div className="picker-list">
          {projects.map((project) => (
            <label className="picker-row" key={project.audioFileId}>
              <input
                type="checkbox"
                checked={selected.has(project.audioFileId)}
                disabled={busy}
                onChange={() => toggle(project.audioFileId)}
              />
              <span className="picker-lines">
                <strong>{project.name}</strong>
                {/* The list's own subtitle, not a second copy of it, so what the
                    picker says about a recording can never drift from what the
                    row behind it says. */}
                <small>{projectSubtitle(project, isTranscript)}</small>
              </span>
            </label>
          ))}
        </div>

        <footer className="picker-foot">
          <button className="ghost-btn" type="button" onClick={onCancel} disabled={busy}>Cancel</button>
          <button
            className="primary-btn"
            type="button"
            disabled={busy || chosen.length === 0}
            onClick={() => onGenerate(chosen)}
          >
            {busy ? "Generating…" : "Generate report"}
          </button>
        </footer>
      </div>
    </div>
  );
}
