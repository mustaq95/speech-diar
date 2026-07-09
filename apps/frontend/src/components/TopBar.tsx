import type { Nav } from "../types";
import { SPEAKER_COLORS } from "../data";

interface TopBarProps {
  nav: Nav;
  onNav: (nav: Nav) => void;
}

export function TopBar({ nav, onNav }: TopBarProps) {
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
      <nav className="top-nav" aria-label="Primary">
        <button className={nav === "dashboard" ? "is-active" : ""} type="button" onClick={() => onNav("dashboard")}>
          Dashboard
        </button>
        <button className={nav === "projects" ? "is-active" : ""} type="button" onClick={() => onNav("projects")}>
          Projects
        </button>
        <button className={nav === "upload" ? "is-active" : ""} type="button" onClick={() => onNav("upload")}>
          Upload
        </button>
        <button className={nav === "settings" ? "is-active" : ""} type="button" onClick={() => onNav("settings")}>
          Settings
        </button>
      </nav>
    </header>
  );
}
