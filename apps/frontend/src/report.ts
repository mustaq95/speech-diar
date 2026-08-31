import type { DiarizationEvaluation, Project } from "./types";
import type { TranscriptReference, TranscriptRun } from "./types/diarization";
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

export function downloadReport(html: string, filename = "diarization-report.html"): void {
  const blob = new Blob([html], { type: "text/html" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}


/**
 * The transcript comparison as a standalone HTML report.
 *
 * Beside `buildReportHtml` above and sharing its light palette, which is
 * deliberately separate from the app's dark theme — a report gets printed and
 * emailed.
 *
 * It repeats the surface's two rules rather than relaxing them for a document:
 * an unmeasurable figure prints its reason, and every error rate is labelled with
 * the transport that produced it. A report is exactly where a number gets
 * detached from its context and quoted, so the context travels with it.
 */
/** A 0..1 rate as a percentage. Distinct from `pct(part, whole)` above, which
 * divides first — an error rate is already a ratio. */
function ratePct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function buildTranscriptReportHtml(
  runs: Array<{
    asrId: string;
    asrName?: string;
    transport?: string | null;
    source?: string;
    chunkIntervalSec?: number;
    chunkCount?: number;
    avgLatencyMs?: number;
    firstLatencyMs?: number;
    text?: string;
    metrics?: {
      wer: number; cer: number; werRaw: number; cerRaw: number;
      refWordCount: number; hypWordCount: number;
      subCount: number; delCount: number; insCount: number;
      rtf?: number;
    } | null;
  }>,
  engines: Array<{ asrId: string; name: string; transport: string }>,
): string {
  const nameOf = (asrId: string, fallback?: string) =>
    engines.find((engine) => engine.asrId === asrId)?.name ?? fallback ?? asrId;
  const transportLabel = (run: (typeof runs)[number]) => {
    const transport = run.transport ?? engines.find((e) => e.asrId === run.asrId)?.transport ?? "";
    if (transport === "stream") return "stream · engine VAD";
    // "file" must never fall through to "chunks": that engine was handed the whole
    // recording in one call after the fact, and labelling it chunked would credit
    // it with a boundary cost it never paid.
    if (transport === "file") return "file · whole recording";
    return run.chunkIntervalSec ? `chunks · ${run.chunkIntervalSec.toFixed(1)}s` : "chunks";
  };
  const referenceWords = runs.find((run) => run.metrics)?.metrics?.refWordCount ?? 0;

  const rows = runs
    .map((run) => {
      const m = run.metrics;
      const isStream = (run.transport ?? "") === "stream";
      return `<tr>
        <td><strong>${escapeHtml(nameOf(run.asrId, run.asrName))}</strong><br>
            <span class="sub">${escapeHtml(transportLabel(run))}${run.source ? ` · ${escapeHtml(run.source)}` : ""}</span></td>
        <td>${m ? ratePct(m.wer) : "<span class='sub'>not scored</span>"}</td>
        <td>${m ? ratePct(m.werRaw) : "—"}</td>
        <td>${m ? ratePct(m.cer) : "—"}</td>
        <td>${m ? m.hypWordCount : "—"}</td>
        <td>${m ? `${m.subCount} / ${m.delCount} / ${m.insCount}` : "—"}</td>
        <td>${run.avgLatencyMs != null ? `${Math.round(run.avgLatencyMs)} ms` : "—"}</td>
        <td>${
          m?.rtf != null
            ? m.rtf.toFixed(3)
            : `<span class='sub'>${isStream ? "real-time bound" : "—"}</span>`
        }</td>
      </tr>`;
    })
    .join("\n");

  const transcripts = runs
    .map(
      (run) => `<section class="transcript">
        <h3>${escapeHtml(nameOf(run.asrId, run.asrName))}</h3>
        <p class="sub">${escapeHtml(transportLabel(run))}</p>
        <p dir="auto">${escapeHtml(run.text ?? "")}</p>
      </section>`,
    )
    .join("\n");

  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Transcript evaluation</title>
<style>
  body { font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #f7f6f3; color: #1a1a1a; margin: 0; padding: 40px; }
  .wrap { max-width: 980px; margin: 0 auto; }
  h1 { font-size: 24px; margin: 0 0 4px; }
  h3 { font-size: 15px; margin: 0 0 2px; }
  .sub { color: #6b6b6b; font-size: 12px; }
  table { width: 100%; border-collapse: collapse; margin: 24px 0; background: #fff;
          border: 1px solid #e4e1d9; }
  th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid #e4e1d9;
           vertical-align: top; }
  th { font-size: 11px; text-transform: uppercase; letter-spacing: .5px; color: #6b6b6b; }
  .note { background: #fff; border: 1px solid #e4e1d9; padding: 14px 16px; margin: 20px 0;
          font-size: 13px; }
  .transcript { background: #fff; border: 1px solid #e4e1d9; padding: 14px 16px; margin: 12px 0; }
</style></head><body><div class="wrap">
<h1>Transcript evaluation</h1>
<p class="sub">${runs.length} engine(s) · ${referenceWords} reference words · normalized and raw rates both shown</p>

<div class="note">
  <strong>How to read this.</strong> Each engine ran on its own native transport, and every
  error rate below includes the cost of that transport. A chunked engine is cut at a fixed
  interval because it cannot accept a longer piece of audio, and a word split across a
  boundary counts against it; a streaming engine has no boundaries. These are therefore
  measurements of two <em>pipelines</em>, not of two models in the abstract.
  Real-Time Factor is blank for a streaming engine because a real-time protocol consumes
  audio at 1&times; by definition — there is no factor to report, so none is invented.
</div>

<table>
  <thead><tr>
    <th>Engine</th><th>WER</th><th>WER (raw)</th><th>CER</th><th>Words</th>
    <th>Sub / Del / Ins</th><th>Avg latency</th><th>RTF</th>
  </tr></thead>
  <tbody>${rows}</tbody>
</table>

<h2 style="font-size:16px">Transcripts</h2>
${transcripts}
</div></body></html>`;
}


/* ==========================================================================
 * Aggregate transcript report
 *
 * The diarization report above is descriptive by necessity — that surface has no
 * ground truth, so it can only say what the models produced. The transcript
 * surface DOES have ground truth: every recording is read from a known script.
 * So this one scores, and the executive question it answers is "which engine is
 * more accurate, and does that depend on the language".
 *
 * Written for someone who will read the tables and not the code, so each table
 * carries the one caveat that would otherwise let it be misread.
 * ========================================================================== */

export interface TranscriptReportEntry {
  project: Project;
  reference: TranscriptReference | null;
  runs: TranscriptRun[];
}

/** Human labels for the stored language-mix ids. An id with no label here still
 * groups correctly and prints as itself, so a new mix in .env never breaks this. */
const LANGUAGE_LABELS: Record<string, string> = {
  ar: "Arabic only",
  "mixed-50-50": "Mixed Arabic / English",
  en: "English only",
};

interface EngineTotals {
  asrId: string;
  /** (engine, feed mode) — what the per-recording `wer` record is keyed on. */
  key: string;
  /** Display name WITH its feed mode, because an engine contributes up to two
   *  columns (its live measurement and its batch one) and they are different
   *  pipelines. An unlabelled pair reads as one engine measured twice. */
  name: string;
  transport: string;
  recordings: number;
  /** Summed edit operations and reference words — WER is recomputed from these
   * rather than averaged, see `weightedWer`. */
  sub: number;
  del: number;
  ins: number;
  refWords: number;
  hypWords: number;
  /** Reference-word-weighted sums, for the rates that cannot be recomputed from
   * counts because character totals are not stored. */
  cerWeighted: number;
  werRawWeighted: number;
  latencySum: number;
  latencyCount: number;
  rtfSum: number;
  rtfCount: number;
}

function emptyTotals(key: string, asrId: string, name: string, transport: string): EngineTotals {
  return {
    key, asrId, name, transport, recordings: 0,
    sub: 0, del: 0, ins: 0, refWords: 0, hypWords: 0,
    cerWeighted: 0, werRawWeighted: 0,
    latencySum: 0, latencyCount: 0, rtfSum: 0, rtfCount: 0,
  };
}

/**
 * Error rate over a SET of recordings: total errors over total reference words.
 *
 * Not the mean of each recording's rate. A 10-word recording and a 500-word one
 * do not carry equal weight, and averaging the rates would let one short clip
 * swing the headline number. Recomputing from the summed S/D/I counts is the same
 * arithmetic WER already is, just over a larger denominator.
 */
function weightedWer(totals: EngineTotals): number | null {
  if (!totals.refWords) return null;
  return (totals.sub + totals.del + totals.ins) / totals.refWords;
}

/** A report column's identity: the engine and the mode it ran in. */
function runColumnKey(run: TranscriptRun): string {
  return `${run.asrId}::${run.source}`;
}

/** What produced this run, in the words the UI uses. "replay" is its own label
 *  and not folded into "stream": a replayed row's latency is the gateway's round
 *  trip, not how far behind a speaker the engine ran, so its speed columns are
 *  not comparable with a read-aloud's. */
function feedModeLabel(run: TranscriptRun): string {
  if (run.source !== "live") return "batch";
  return run.replayed ? "replay" : "stream";
}

function accumulate(totals: EngineTotals, run: TranscriptRun): void {
  const metrics = run.metrics;
  if (!metrics) return;
  totals.recordings += 1;
  totals.sub += metrics.subCount;
  totals.del += metrics.delCount;
  totals.ins += metrics.insCount;
  totals.refWords += metrics.refWordCount;
  totals.hypWords += metrics.hypWordCount;
  totals.cerWeighted += metrics.cer * metrics.refWordCount;
  totals.werRawWeighted += metrics.werRaw * metrics.refWordCount;
  if (run.avgLatencyMs != null) {
    totals.latencySum += run.avgLatencyMs;
    totals.latencyCount += 1;
  }
  if (metrics.rtf != null) {
    totals.rtfSum += metrics.rtf;
    totals.rtfCount += 1;
  }
}

interface TranscriptReportData {
  generatedAt: string;
  recordings: number;
  scoredRecordings: number;
  totalAudioSec: number;
  totalReferenceWords: number;
  engines: EngineTotals[];
  /** One row per language mix, each with per-engine totals. */
  byLanguage: Array<{ key: string; label: string; recordings: number; engines: EngineTotals[] }>;
  perRecording: Array<{
    name: string;
    date: string;
    duration: string;
    language: string;
    /** Measured Arabic share of the reference, when the script generator reported
     * it. The REQUESTED mix and the produced one differ, so both are shown. */
    measuredArabic: number | null;
    referenceWords: number;
    wer: Record<string, number | null>;
  }>;
}

/**
 * How many reference words a recording is scored against.
 *
 * This is the TOKENIZED count the WER denominator uses, not `reference.wordCount`
 * (a plain whitespace split of the script). They differ — punctuation-joined tokens
 * separate during normalization — and the report must not print both under one label:
 * a reader comparing the summary card against the tables would find two different
 * totals for the same thing. Every rate here divides by this number, so this is the
 * one the document reports. Falls back to the raw split for a recording that was never
 * scored, which has no denominator of its own.
 */
function scoredRefWords(entry: TranscriptReportEntry): number {
  const scored = entry.runs.find((run) => run.metrics)?.metrics?.refWordCount;
  return scored ?? entry.reference?.wordCount ?? 0;
}

export function computeTranscriptReport(entries: TranscriptReportEntry[]): TranscriptReportData {
  const overall = new Map<string, EngineTotals>();
  const byLanguage = new Map<string, Map<string, EngineTotals>>();
  const languageOrder: string[] = [];
  const perRecording: TranscriptReportData["perRecording"] = [];
  let totalAudioSec = 0;
  let totalReferenceWords = 0;
  let scoredRecordings = 0;

  for (const entry of entries) {
    const params = (entry.reference?.params ?? null) as
      | { languageMix?: string; languageSplit?: { arabic: number } }
      | null;
    const languageKey = params?.languageMix ?? (entry.reference ? "unspecified" : "no-reference");
    if (!byLanguage.has(languageKey)) {
      byLanguage.set(languageKey, new Map());
      languageOrder.push(languageKey);
    }
    const languageBucket = byLanguage.get(languageKey)!;

    totalAudioSec += entry.project.durationSec;
    totalReferenceWords += scoredRefWords(entry);
    if (entry.runs.some((run) => run.metrics)) scoredRecordings += 1;

    const wer: Record<string, number | null> = {};
    for (const run of entry.runs) {
      const transport = run.transport ?? "";
      // Keyed on (engine, feed mode), never the engine alone. One recording can
      // hold an engine's live result AND its batch one; summing them into a
      // single bucket would report the mean of two different pipelines under one
      // name, and the per-recording row would silently keep whichever came last.
      const key = runColumnKey(run);
      const name = `${run.asrName || run.asrId} (${feedModeLabel(run)})`;
      if (!overall.has(key)) overall.set(key, emptyTotals(key, run.asrId, name, transport));
      if (!languageBucket.has(key)) {
        languageBucket.set(key, emptyTotals(key, run.asrId, name, transport));
      }
      accumulate(overall.get(key)!, run);
      accumulate(languageBucket.get(key)!, run);
      wer[key] = run.metrics?.wer ?? null;
    }

    perRecording.push({
      name: entry.project.name,
      date: entry.project.date,
      duration: entry.project.duration,
      language: LANGUAGE_LABELS[languageKey] ?? languageKey,
      measuredArabic: params?.languageSplit?.arabic ?? null,
      referenceWords: scoredRefWords(entry),
      wer,
    });
  }

  return {
    generatedAt: new Date().toLocaleString(),
    recordings: entries.length,
    scoredRecordings,
    totalAudioSec,
    totalReferenceWords,
    engines: [...overall.values()],
    byLanguage: languageOrder.map((key) => ({
      key,
      label: LANGUAGE_LABELS[key] ?? (key === "no-reference" ? "No reference" : key),
      recordings: entries.filter((entry) => {
        const params = (entry.reference?.params ?? null) as { languageMix?: string } | null;
        const entryKey = params?.languageMix ?? (entry.reference ? "unspecified" : "no-reference");
        return entryKey === key;
      }).length,
      engines: [...byLanguage.get(key)!.values()],
    })),
    perRecording,
  };
}

function ratePctOrDash(value: number | null): string {
  return value == null ? "<span class='sub'>—</span>" : `${(value * 100).toFixed(1)}%`;
}

/** Best (lowest) value wins a highlight, so the comparison reads at a glance. */
function bestOf(values: Array<number | null>): number | null {
  const real = values.filter((value): value is number => value != null);
  return real.length ? Math.min(...real) : null;
}

export function buildTranscriptAggregateReportHtml(report: TranscriptReportData): string {
  const engines = report.engines;
  const engineHeads = engines.map((engine) => `<th>${escapeHtml(engine.name)}</th>`).join("");

  const overallRows = engines
    .map((engine) => {
      const wer = weightedWer(engine);
      const cer = engine.refWords ? engine.cerWeighted / engine.refWords : null;
      const werRaw = engine.refWords ? engine.werRawWeighted / engine.refWords : null;
      const latency = engine.latencyCount ? engine.latencySum / engine.latencyCount : null;
      const rtf = engine.rtfCount ? engine.rtfSum / engine.rtfCount : null;
      const isStream = engine.transport === "stream";
      const transportLabel =
        engine.transport === "stream"
          ? "stream · engine VAD"
          : engine.transport === "file"
            ? "file · whole recording"
            : "chunks";
      return `<tr>
        <td><strong>${escapeHtml(engine.name)}</strong><br>
            <span class="sub">${escapeHtml(transportLabel)}</span></td>
        <td class="num">${engine.recordings}</td>
        <td class="num">${ratePctOrDash(wer)}</td>
        <td class="num">${ratePctOrDash(werRaw)}</td>
        <td class="num">${ratePctOrDash(cer)}</td>
        <td class="num">${engine.sub} / ${engine.del} / ${engine.ins}</td>
        <td class="num">${engine.refWords} → ${engine.hypWords}</td>
        <td class="num">${latency == null ? "—" : `${Math.round(latency)} ms`}</td>
        <td class="num">${
          rtf == null
            ? `<span class="sub">${isStream ? "real-time bound" : "—"}</span>`
            : rtf.toFixed(3)
        }</td>
      </tr>`;
    })
    .join("\n");

  const languageRows = report.byLanguage
    .map((group) => {
      const cells = engines.map((engine) => {
        // Matched on `key`, not asrId: an engine contributes one column per feed
        // mode, and asrId would find whichever of its two came first and print
        // the batch figure under the stream column.
        const totals = group.engines.find((item) => item.key === engine.key);
        const wer = totals ? weightedWer(totals) : null;
        const best = bestOf(
          engines.map((other) => {
            const t = group.engines.find((item) => item.key === other.key);
            return t ? weightedWer(t) : null;
          }),
        );
        const winner = wer != null && best != null && wer === best && engines.length > 1;
        return `<td class="num${winner ? " win" : ""}">${ratePctOrDash(wer)}</td>`;
      });
      const words = group.engines[0]?.refWords ?? 0;
      return `<tr>
        <td><strong>${escapeHtml(group.label)}</strong></td>
        <td class="num">${group.recordings}</td>
        <td class="num">${words}</td>
        ${cells.join("")}
      </tr>`;
    })
    .join("\n");

  const recordingRows = report.perRecording
    .map((row) => {
      const cells = engines
        .map((engine) => `<td class="num">${ratePctOrDash(row.wer[engine.key] ?? null)}</td>`)
        .join("");
      return `<tr>
        <td>${escapeHtml(row.name)}<br><span class="sub">${escapeHtml(row.date)}</span></td>
        <td class="num">${escapeHtml(row.duration)}</td>
        <td>${escapeHtml(row.language)}${
          row.measuredArabic != null
            ? `<br><span class="sub">${Math.round(row.measuredArabic * 100)}% Arabic measured</span>`
            : ""
        }</td>
        <td class="num">${row.referenceWords}</td>
        ${cells}
      </tr>`;
    })
    .join("\n");

  // Lowest error rate wins. A reduce rather than a sort: one pass, no mutation of
  // the intermediate array, and finding a minimum never needed an ordering.
  const winner = engines
    .map((engine) => ({ name: engine.name, wer: weightedWer(engine) }))
    .filter((item): item is { name: string; wer: number } => item.wer != null)
    .reduce<{ name: string; wer: number } | null>(
      (best, item) => (best == null || item.wer < best.wer ? item : best),
      null,
    );

  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Transcript evaluation report</title>
<style>
  body { font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #f7f6f3; color: #1a1a1a; margin: 0; padding: 40px; }
  .wrap { max-width: 1040px; margin: 0 auto; }
  h1 { font-size: 24px; margin: 0 0 4px; }
  h2 { font-size: 16px; margin: 32px 0 8px; }
  .sub { color: #6b6b6b; font-size: 12px; }
  .cards { display: flex; gap: 12px; margin: 20px 0; flex-wrap: wrap; }
  .card { flex: 1 1 160px; background: #fff; border: 1px solid #e4e1d9; padding: 12px 14px; }
  .card b { display: block; font-size: 20px; margin-top: 2px; }
  table { width: 100%; border-collapse: collapse; margin: 8px 0 4px; background: #fff;
          border: 1px solid #e4e1d9; }
  th, td { text-align: left; padding: 9px 12px; border-bottom: 1px solid #e4e1d9;
           vertical-align: top; }
  th { font-size: 11px; text-transform: uppercase; letter-spacing: .5px; color: #6b6b6b; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  td.win { font-weight: 700; color: #0a7c42; }
  .note { background: #fff; border: 1px solid #e4e1d9; padding: 14px 16px; margin: 16px 0;
          font-size: 13px; }
  .note b { display: block; margin-bottom: 4px; }
</style></head><body><div class="wrap">

<h1>Transcript evaluation report</h1>
<p class="sub">${report.recordings} recording${report.recordings === 1 ? "" : "s"} ·
  ${report.scoredRecordings} scored · ${fmt(report.totalAudioSec)} of audio ·
  ${report.totalReferenceWords} reference words · generated ${escapeHtml(report.generatedAt)}</p>

<div class="cards">
  <div class="card"><span class="sub">Recordings scored</span><b>${report.scoredRecordings}</b></div>
  <div class="card"><span class="sub">Reference words</span><b>${report.totalReferenceWords}</b></div>
  <div class="card"><span class="sub">Engines compared</span><b>${engines.length}</b></div>
  <div class="card"><span class="sub">Lowest word error rate</span><b>${
    winner ? `${(winner.wer * 100).toFixed(1)}%` : "—"
  }</b><span class="sub">${winner ? escapeHtml(winner.name) : "not scored"}</span></div>
</div>

<div class="note">
  <b>How to read this.</b>
  Every recording was read aloud from a known script, so these are true error rates rather
  than model agreement. Word Error Rate is substitutions plus deletions plus insertions,
  over the number of words in the script; lower is better and it can exceed 100% when an
  engine emits more wrong words than the script contains.
  Rates over a set of recordings are computed as total errors over total reference words,
  <em>not</em> as an average of each recording's rate — otherwise one short clip would carry
  the same weight as a long one. Reference counts are of the tokenized script, which runs
  slightly above a plain word count because punctuation-joined tokens separate.
</div>

<h2>By engine</h2>
<table>
  <thead><tr>
    <th>Engine</th><th class="num">Scored</th><th class="num">WER</th>
    <th class="num">WER (raw)</th><th class="num">CER</th>
    <th class="num">Sub / Del / Ins</th><th class="num">Words ref → out</th>
    <th class="num">Avg latency</th><th class="num">RTF</th>
  </tr></thead>
  <tbody>${overallRows}</tbody>
</table>
<p class="sub">WER (raw) is the same comparison without Arabic normalisation — no tashkeel
  stripping, no alef/hamza or ta-marbuta folding, Arabic-Indic digits left as they are. The
  gap between the two columns is orthographic variation rather than misrecognition.
  CER is reference-word weighted, because character totals are not stored per run.</p>

<h2>By language</h2>
<table>
  <thead><tr>
    <th>Language of script</th><th class="num">Recordings</th>
    <th class="num">Reference words</th>${engineHeads}
  </tr></thead>
  <tbody>${languageRows}</tbody>
</table>
<p class="sub">Word Error Rate per engine, per language of the source script. The lower rate
  in each row is highlighted. Language is what the script was REQUESTED in; the per-recording
  table below also shows the Arabic share actually measured in the generated text, which
  differs because the generator follows a mix instruction loosely.</p>

<h2>By recording</h2>
<table>
  <thead><tr>
    <th>Recording</th><th class="num">Duration</th><th>Language</th>
    <th class="num">Ref words</th>${engineHeads}
  </tr></thead>
  <tbody>${recordingRows}</tbody>
</table>

<div class="note">
  <b>What these numbers are measurements of.</b>
  Each engine ran on its own native transport and every error rate includes the cost of that
  transport. A chunked engine is cut at a fixed interval because it cannot accept a longer
  piece of audio, and a word split across a boundary counts against it; a streaming engine has
  no boundaries. These are therefore comparisons of two <em>pipelines</em> as deployed, not of
  two models in the abstract. Real-Time Factor is blank for a streaming engine because a
  real-time protocol consumes audio at 1&times; by definition — there is no factor to report,
  so none is invented.
</div>

</div></body></html>`;
}
