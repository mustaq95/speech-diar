import type { Nav, StudioMode } from "../types";
import { SPEAKER_COLORS, STUDIO_MODES } from "../data";
import { Segmented } from "./controls";

interface TopBarProps {
  nav: Nav;
  onNav: (nav: Nav) => void;
  onStudioMode: (mode: StudioMode) => void;
  /** Which surface the recordings list is scoped to. Every page except the two
   * studios is describing this rather than a page of its own. */
  listSurface: StudioMode;
  /** Which primary tab the transcript page counts as. That page is both the create
   * page and the viewer for its surface, so the tab comes from the click that led
   * there rather than from the page itself. */
  transcriptTab: Nav;
}

export function TopBar({ nav, onNav, onStudioMode, listSurface, transcriptTab }: TopBarProps) {
  // Every page belongs to a surface now, so the toggle always reflects one. The
  // transcript page is the only one that reports itself, because it IS the
  // transcript surface; everywhere else `listSurface` is the surface in play.
  const studioMode: StudioMode = nav === "transcript" ? "transcript" : listSurface;

  // Which primary tab is active. The transcript page serves both activities for
  // its surface, so it reports whichever one the caller asked for — inferring it
  // from page state made clicking Dashboard highlight Upload.
  const activeTab: Nav = nav === "transcript" ? transcriptTab : nav;

  return (
    <header className="topbar">
      <div className="brand-block">
        <div className="brand-mark" aria-hidden="true">
          {SPEAKER_COLORS.slice(0, 4).map((color, index) => (
            <span key={color} style={{ background: color, height: [9, 18, 13, 20][index] }} />
          ))}
        </div>
        <span className="brand-name">SPEECHDYN</span>
        <span className="brand-sep" />
        <span className="brand-sub">Multi-Model Diarization</span>
      </div>
      {/* Centre of the top bar, on the same row as the primary nav. The toggle
          switches evaluation surface, which is a different axis from the nav
          destinations either side of it — hence a segmented control rather than
          two more tabs. */}
      <div className="topbar-switcher">
        <Segmented
          value={studioMode}
          options={STUDIO_MODES}
          onChange={(mode) => onStudioMode(mode as StudioMode)}
          label="Evaluation surface"
        />
      </div>

      <nav className="top-nav" aria-label="Primary">
        {/* Both surfaces share these tabs; `activeTab` above resolves which one the
            transcript page counts as. */}
        <button
          className={activeTab === "dashboard" ? "is-active" : ""}
          type="button"
          onClick={() => onNav("dashboard")}
        >
          Dashboard
        </button>
        <button className={nav === "projects" ? "is-active" : ""} type="button" onClick={() => onNav("projects")}>
          Projects
        </button>
        <button className={activeTab === "upload" ? "is-active" : ""} type="button" onClick={() => onNav("upload")}>
          Upload
        </button>
        <button className={nav === "settings" ? "is-active" : ""} type="button" onClick={() => onNav("settings")}>
          Settings
        </button>
      </nav>
    </header>
  );
}
