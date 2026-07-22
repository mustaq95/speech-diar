import type { DiarizationEvaluation, Project } from "./types";
import { fmt } from "./utils";

/**
 * Aggregate diarization report — a descriptive comparison of what the models
 * actually produced across the saved recordings. There is no ground-truth
 * reference in the platform, so this never scores accuracy (no DER/JER); it
 * reports the real signals: speakers detected, turns, speech coverage,
 * processing time, and where the models disagree on speaker count.
 */

export interface ReportEntry {
  project: Project;
  evaluation: DiarizationEvaluation;
}

interface ModelStat {
  id: string;
  name: string;
  short: string;
  recordingsRun: number;
  recordingsDone: number;
  recordingsFailed: number;
  totalSegments: number;
  totalSpeechSec: number;
  doneDurationSec: number; // denominator for speech coverage
  speakerSum: number;
  processingMsSum: number;
  processingMsCount: number;
}

interface RecordingRow {
  name: string;
  durationSec: number;
  perModelSpeakers: Record<string, number>; // done models only
  speakerMin: number | null;
  speakerMax: number | null;
}

interface ReportData {
  generatedAt: Date;
  recordingCount: number;
  totalDurationSec: number;
  modelStats: ModelStat[];
  recordings: RecordingRow[];
}

function speechSeconds(segs: { s: number; e: number }[]): number {
  return segs.reduce((sum, seg) => sum + Math.max(0, seg.e - seg.s), 0);
}

export function computeReport(entries: ReportEntry[]): ReportData {
  const stats = new Map<string, ModelStat>();
  const recordings: RecordingRow[] = [];
  let totalDurationSec = 0;

  for (const { project, evaluation } of entries) {
    totalDurationSec += evaluation.durationSec;
    const perModelSpeakers: Record<string, number> = {};

    for (const model of evaluation.models) {
      let stat = stats.get(model.id);
      if (!stat) {
        stat = {
          id: model.id,
          name: model.name,
          short: model.short,
          recordingsRun: 0,
          recordingsDone: 0,
          recordingsFailed: 0,
          totalSegments: 0,
          totalSpeechSec: 0,
          doneDurationSec: 0,
          speakerSum: 0,
          processingMsSum: 0,
          processingMsCount: 0,
        };
        stats.set(model.id, stat);
      }
      stat.recordingsRun += 1;
      if (model.status === "failed") stat.recordingsFailed += 1;
      if (model.status === "done") {
        stat.recordingsDone += 1;
        stat.totalSegments += model.segs.length;
        stat.totalSpeechSec += speechSeconds(model.segs);
        stat.doneDurationSec += evaluation.durationSec;
        stat.speakerSum += model.numSpk;
        if (model.processingMs != null) {
          stat.processingMsSum += model.processingMs;
          stat.processingMsCount += 1;
        }
        perModelSpeakers[model.id] = model.numSpk;
      }
    }

    const speakerCounts = Object.values(perModelSpeakers);
    recordings.push({
      name: project.name,
      durationSec: evaluation.durationSec,
      perModelSpeakers,
      speakerMin: speakerCounts.length ? Math.min(...speakerCounts) : null,
      speakerMax: speakerCounts.length ? Math.max(...speakerCounts) : null,
    });
  }

  return {
    generatedAt: new Date(),
    recordingCount: entries.length,
    totalDurationSec,
    modelStats: [...stats.values()].sort((a, b) => a.name.localeCompare(b.name)),
    recordings,
  };
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function pct(part: number, whole: number): string {
  if (whole <= 0) return "—";
  return `${((part / whole) * 100).toFixed(1)}%`;
}

function avg(sum: number, count: number, digits = 1): string {
  if (count <= 0) return "—";
  return (sum / count).toFixed(digits);
}

function ms(sum: number, count: number): string {
  if (count <= 0) return "—";
  const seconds = sum / count / 1000;
  return `${seconds.toFixed(1)}s`;
}

export function buildReportHtml(report: ReportData): string {
  const modelIds = report.modelStats.map((stat) => stat.id);

  const overview = `
    <section class="cards">
      <div class="card"><span>${report.recordingCount}</span><label>Recordings</label></div>
      <div class="card"><span>${fmt(report.totalDurationSec)}</span><label>Total audio</label></div>
      <div class="card"><span>${report.modelStats.length}</span><label>Models compared</label></div>
    </section>`;

  const modelRows = report.modelStats
    .map(
      (stat) => `
      <tr>
        <td class="name"><strong>${escapeHtml(stat.name)}</strong><small>${escapeHtml(stat.id)}</small></td>
        <td>${stat.recordingsDone}${stat.recordingsFailed ? ` <span class="fail">(${stat.recordingsFailed} failed)</span>` : ""}</td>
        <td>${avg(stat.speakerSum, stat.recordingsDone)}</td>
        <td>${stat.totalSegments.toLocaleString()}</td>
        <td>${pct(stat.totalSpeechSec, stat.doneDurationSec)}</td>
        <td>${ms(stat.processingMsSum, stat.processingMsCount)}</td>
      </tr>`,
    )
    .join("");

  const recordingRows = report.recordings
    .map((rec) => {
      const cells = modelIds
        .map((id) => {
          const value = rec.perModelSpeakers[id];
          return `<td>${value == null ? "<span class='muted'>—</span>" : value}</td>`;
        })
        .join("");
      const spread = rec.speakerMin != null && rec.speakerMax != null ? rec.speakerMax - rec.speakerMin : 0;
      const spreadClass = spread >= 2 ? "diverge" : "";
      const spreadLabel = rec.speakerMin != null ? `${rec.speakerMin}–${rec.speakerMax}` : "—";
      return `
      <tr class="${spreadClass}">
        <td class="name">${escapeHtml(rec.name)}</td>
        <td>${fmt(rec.durationSec)}</td>
        ${cells}
        <td>${spreadLabel}</td>
      </tr>`;
    })
    .join("");

  const modelHeaders = report.modelStats.map((stat) => `<th>${escapeHtml(stat.short)}</th>`).join("");

  const generated = report.generatedAt.toLocaleString("en-US", {
    dateStyle: "long",
    timeStyle: "short",
  });

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Diarization Evaluation Report</title>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #1a1a1a;
    background: #f7f6f3;
    line-height: 1.55;
  }
  main { max-width: 960px; margin: 0 auto; padding: 56px 28px 96px; }
  header { border-bottom: 1px solid #e4e1d9; padding-bottom: 24px; margin-bottom: 36px; }
  h1 { font-size: 30px; margin: 0 0 6px; letter-spacing: -0.02em; }
  .sub { color: #6b6862; font-size: 14px; margin: 0; }
  h2 { font-size: 18px; margin: 40px 0 14px; letter-spacing: -0.01em; }
  p.note { color: #6b6862; font-size: 13px; margin: 0 0 18px; }
  .cards { display: flex; gap: 14px; flex-wrap: wrap; margin: 8px 0 4px; }
  .card {
    flex: 1 1 140px; background: #fff; border: 1px solid #e4e1d9; border-radius: 12px;
    padding: 18px 20px; display: flex; flex-direction: column; gap: 4px;
  }
  .card span { font-size: 26px; font-weight: 650; letter-spacing: -0.02em; }
  .card label { font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: #8a877f; }
  .table-wrap { overflow-x: auto; border: 1px solid #e4e1d9; border-radius: 12px; background: #fff; }
  table { border-collapse: collapse; width: 100%; font-size: 14px; }
  th, td { text-align: left; padding: 11px 14px; border-bottom: 1px solid #efece4; white-space: nowrap; }
  thead th { font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; color: #8a877f; font-weight: 600; }
  tbody tr:last-child td { border-bottom: none; }
  td.name { white-space: normal; }
  td.name small { display: block; color: #a29e94; font-size: 11px; }
  .muted { color: #c2beb4; }
  .fail { color: #b5473a; font-size: 12px; }
  tr.diverge td { background: #fdf6ec; }
  tr.diverge td.name::after { content: " ⚠"; color: #c78a2b; }
  footer { margin-top: 44px; color: #a29e94; font-size: 12px; }
</style>
</head>
<body>
<main>
  <header>
    <h1>Diarization Evaluation Report</h1>
    <p class="sub">Generated ${escapeHtml(generated)} · descriptive comparison of raw model output</p>
  </header>

  ${overview}

  <h2>Per-model summary</h2>
  <p class="note">Averaged over recordings each model completed. Speech coverage is total speaker-time over audio duration, so it can exceed 100% where speakers overlap.</p>
  <div class="table-wrap">
    <table>
      <thead>
        <tr><th>Model</th><th>Completed</th><th>Avg speakers</th><th>Total turns</th><th>Speech coverage</th><th>Avg runtime</th></tr>
      </thead>
      <tbody>${modelRows}</tbody>
    </table>
  </div>

  <h2>Per-recording speaker counts</h2>
  <p class="note">Speakers detected by each model. Rows where models disagree by 2 or more are flagged.</p>
  <div class="table-wrap">
    <table>
      <thead>
        <tr><th>Recording</th><th>Duration</th>${modelHeaders}<th>Spread</th></tr>
      </thead>
      <tbody>${recordingRows}</tbody>
    </table>
  </div>

  <footer>SPEECHDYN · Multi-Model Diarization · nothing on this page is fabricated; every figure is computed from the stored evaluation.</footer>
</main>
</body>
</html>`;
}

export function downloadReport(html: string): void {
  const blob = new Blob([html], { type: "text/html" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "diarization-report.html";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
