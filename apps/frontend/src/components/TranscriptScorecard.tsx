import type { TranscriptRun } from "../types/diarization";
import type { TranscriptEngineInfo } from "../adapters";
import { buildTranscriptReportHtml, downloadReport } from "../report";

interface TranscriptScorecardProps {
  runs: TranscriptRun[];
  engines: TranscriptEngineInfo[];
  referenceWords: number;
}

/** One comparable figure per engine, plus how to read it.
 *
 * `lowerIsBetter` drives the bar length, because a bar that grows with the error
 * rate would make the worse engine look like the fuller one. `unavailable`
 * carries the reason a figure is missing, which is never rendered as zero. */
interface Tile {
  title: string;
  hint: string;
  lowerIsBetter: boolean;
  format: (value: number) => string;
  valueOf: (run: TranscriptRun) => number | null;
  unavailable?: (run: TranscriptRun) => string | null;
}

function pct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

const TILES: Tile[] = [
  {
    title: "Word Error Rate",
    hint: "S + D + I over reference words · lower is better",
    lowerIsBetter: true,
    format: pct,
    valueOf: (run) => run.metrics?.wer ?? null,
  },
  {
    title: "Character Error Rate",
    hint: "Edit distance per character · lower is better",
    lowerIsBetter: true,
    format: pct,
    valueOf: (run) => run.metrics?.cer ?? null,
  },
  {
    title: "Word Count",
    hint: "Words emitted against the reference",
    lowerIsBetter: false,
    format: (value) => String(Math.round(value)),
    valueOf: (run) => run.metrics?.hypWordCount ?? null,
  },
  {
    title: "Real-Time Factor",
    hint: "Processing time ÷ audio duration · lower is better",
    lowerIsBetter: true,
    format: (value) => value.toFixed(3),
    valueOf: (run) => run.metrics?.rtf ?? null,
    // Not a missing measurement: a real-time protocol consumes audio at 1x by
    // definition, so there is no factor to report. Saying so beats printing a
    // number that would look like one.
    unavailable: (run) =>
      run.transport === "stream" ? "real-time bound" : null,
  },
];

/**
 * The comparison scorecard.
 *
 * Two rules it follows without exception:
 *
 * 1. **Nothing is shown that was not measured.** A missing figure renders as its
 *    reason ("real-time bound", "no reference"), never as 0 — a 0.0 error rate is
 *    a perfect transcript and must not be how "unknown" looks.
 * 2. **Every figure carries its transport.** A chunked engine's error rate
 *    includes the cost of the chunking its own limits force on it; a streaming
 *    engine's does not. That is a real difference between the two PIPELINES, and
 *    it only misleads if a reader takes either number as the model's own.
 */
export function TranscriptScorecard({ runs, engines, referenceWords }: TranscriptScorecardProps) {
  const transportOf = (run: TranscriptRun) =>
    run.transport ?? engines.find((engine) => engine.asrId === run.asrId)?.transport ?? null;
  const nameOf = (run: TranscriptRun) =>
    engines.find((engine) => engine.asrId === run.asrId)?.name ?? run.asrName ?? run.asrId;

  const scored = runs.some((run) => run.metrics);
  const first = runs[0]?.metrics;

  return (
    <section className="panel scorecard">
      <div className="scorecard-head">
        <h2>Eval result</h2>
        <span className="mono muted">
          {scored
            ? `scored against the reference · ${first?.refWordCount ?? referenceWords} reference words · normalization on`
            : "no reference for this recording — timings only, no error rates"}
        </span>
        <button
          type="button"
          className="ghost-btn"
          onClick={() => downloadReport(buildTranscriptReportHtml(runs, engines))}
        >
          Export report →
        </button>
      </div>

      <div className="scorecard-grid">
        {TILES.map((tile) => {
          const values = runs
            .map((run) => tile.valueOf(run))
            .filter((value): value is number => value != null);
          const best = values.length
            ? tile.lowerIsBetter ? Math.min(...values) : Math.max(...values)
            : null;
          const worst = values.length
            ? tile.lowerIsBetter ? Math.max(...values) : Math.min(...values)
            : null;

          return (
            <div className="score-tile" key={tile.title}>
              <strong>{tile.title}</strong>
              <small className="muted">{tile.hint}</small>
              <span className="eyebrow">Bar length = better score</span>
              {runs.map((run) => {
                const value = tile.valueOf(run);
                const reason = tile.unavailable?.(run) ?? null;
                // Bar length is share-of-best, so the better engine is always the
                // fuller bar regardless of which direction is good.
                let fill = 0;
                if (value != null && best != null && worst != null) {
                  if (best === worst) fill = 100;
                  else if (tile.lowerIsBetter) fill = (best / value) * 100;
                  else fill = (value / best) * 100;
                }
                const transport = transportOf(run);
                return (
                  <div className="score-row" key={run.asrId}>
                    <div className="score-row-head">
                      <span>
                        <span className={`engine-dot ${transport ?? ""}`} aria-hidden="true" />
                        {nameOf(run)}
                      </span>
                      <b className="mono">
                        {value != null ? tile.format(value) : <span className="muted">{reason ?? "—"}</span>}
                      </b>
                    </div>
                    <div className="score-bar">
                      <i style={{ width: `${Math.max(0, Math.min(100, fill))}%` }} />
                    </div>
                    {/* The transport is on every row, not in a footnote: it is
                        what the figure above is a measurement OF. */}
                    <small className="score-transport mono">
                      {transport === "stream"
                        ? "stream · VAD"
                        : run.chunkIntervalSec
                          ? `chunks · ${run.chunkIntervalSec.toFixed(1)}s`
                          : "chunks"}
                      {run.source ? ` · ${run.source}` : ""}
                    </small>
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>

      {scored && (
        <div className="scorecard-ops">
          {runs.map((run) => (
            <div key={run.asrId} className="ops-row">
              <span>{nameOf(run)}</span>
              {run.metrics ? (
                <span className="mono">
                  {run.metrics.subCount} sub · {run.metrics.delCount} del · {run.metrics.insCount} ins
                  {" · raw WER "}
                  {pct(run.metrics.werRaw)}
                </span>
              ) : (
                <span className="muted">not scored</span>
              )}
            </div>
          ))}
          {/* The measured breakdown, in place of a claim about WHY the engines
              differ. "hamza handling" would be a story we cannot compute; these
              are counts we can. Raw WER sits beside the normalized one so
              normalization's own effect stays visible. */}
          <small className="muted">
            Raw WER is the same comparison without Arabic normalization (tashkeel, alef/hamza,
            ta-marbuta, Arabic-Indic digits). The gap between the two is what normalization absorbed.
          </small>
        </div>
      )}
    </section>
  );
}
