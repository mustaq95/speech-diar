import type { ActiveMap, ModelRun, Project } from "./types";

export function fmt(t: number): string {
  const safe = Math.max(0, t);
  const m = Math.floor(safe / 60);
  const s = Math.floor(safe % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

export function hexA(hex: string, alpha: number): string {
  const c = hex.replace("#", "");
  const full = c.length === 3 ? c.split("").map((x) => x + x).join("") : c;
  const n = Number.parseInt(full, 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
}

export function activeSpeakers(model: ModelRun, t: number): number[] {
  return [...new Set(model.segs.filter((seg) => t >= seg.s && t < seg.e).map((seg) => seg.spk))].sort((a, b) => a - b);
}

export function countAt(model: ModelRun, t: number): number {
  return model.segs.filter((seg) => t >= seg.s && t < seg.e).length;
}

export function overlapsFor(model: ModelRun): Array<{ s: number; e: number }> {
  const bounds = [...new Set(model.segs.flatMap((seg) => [seg.s, seg.e]))].sort((a, b) => a - b);
  const out: Array<{ s: number; e: number }> = [];
  for (let i = 0; i < bounds.length - 1; i += 1) {
    const s = bounds[i];
    const e = bounds[i + 1];
    const mid = (s + e) / 2;
    if (countAt(model, mid) >= 2) {
      const last = out.at(-1);
      if (last && Math.abs(last.e - s) < 0.01) last.e = e;
      else out.push({ s, e });
    }
  }
  return out;
}

export function loadProjects(): Project[] {
  try {
    const raw = localStorage.getItem("speechdyn_projects");
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) return parsed;
    }
  } catch {
    return [];
  }
  return [];
}

export function saveProjects(projects: Project[]): void {
  try {
    localStorage.setItem("speechdyn_projects", JSON.stringify(projects));
  } catch {
    // Local persistence is optional for the static studio.
  }
}

/** Which models the user chose to run, from Settings. Null means no saved
 * preference yet (e.g. first visit) — caller falls back to catalog defaults. */
export function loadModelActive(): ActiveMap | null {
  try {
    const raw = localStorage.getItem("speechdyn_model_active");
    if (raw) {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return parsed;
    }
  } catch {
    return null;
  }
  return null;
}

export function saveModelActive(active: ActiveMap): void {
  try {
    localStorage.setItem("speechdyn_model_active", JSON.stringify(active));
  } catch {
    // Local persistence is optional for the static studio.
  }
}

export function boundsFor(models: ModelRun[]): number[] {
  return [...new Set(models.flatMap((model) => model.segs.flatMap((seg) => [seg.s, seg.e])))].sort((a, b) => a - b);
}
