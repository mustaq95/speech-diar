import { useState } from "react";
import type { TranscriptRun } from "../types/diarization";
import type { TranscriptEngineInfo } from "../adapters";
import { buildTranscriptReportHtml, downloadReport } from "../report";
import { Segmented } from "./controls";

interface TranscriptScorecardProps {
  runs: TranscriptRun[];
  engines: TranscriptEngineInfo[];
  referenceWords: number;
}

/** One comparable figure per engine, plus how to read it.
 *
 * `lowerIsBetter` drives RANK (which engine sorts first, and which bar is
 * coloured best/worst) but NOT bar length. Bar length is the measured value
 * itself, as a share of the largest value in the tile.
 *
 * The bar used to be share-of-best, so a 6.7% WER drew a full bar and a 108%
 * WER drew a stub -- length ran opposite to the number printed beside it, and
 * every reader had to hold "longer means better here" in their head. Now length
 * tracks magnitude the way a bar chart normally does, and direction is carried
 * by the sort order and the colour instead.
 *
 * `unavailable` carries the reason a figure is missing, which is never rendered
 * as zero. */
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
/** A row's identity here is (engine, feed mode): a recording can hold an
 *  engine's live result AND its batch one, so keying on asrId alone gives React
 *  duplicate keys and the reader two identical labels over different numbers. */
function keyOf(run: TranscriptRun): string {
  return `${run.asrId}::${run.source}`;
}

type FeedFilter = "all" | "live" | "batch";

export function TranscriptScorecard({ runs, engines, referenceWords }: TranscriptScorecardProps) {
  const transportOf = (run: TranscriptRun) =>
    run.transport ?? engines.find((engine) => engine.asrId === run.asrId)?.transport ?? null;
  const nameOf = (run: TranscriptRun) =>
    engines.find((engine) => engine.asrId === run.asrId)?.name ?? run.asrName ?? run.asrId;

  const [feedFilter, setFeedFilter] = useState<FeedFilter>("all");
  const feedOf = (run: TranscriptRun): FeedFilter =>
    run.source === "live" ? "live" : "batch";
  const visibleRuns =
    feedFilter === "all" ? runs : runs.filter((run) => feedOf(run) === feedFilter);

  const scored = visibleRuns.some((run) => run.metrics);
  const first = visibleRuns[0]?.metrics;

  return (
    <section className="panel scorecard">
      <div className="scorecard-head">
        <h2>Eval result</h2>
        {/* Filter, not sort: mixing live and batch in one ranking hides the
            fact that they are measurements of two different pipelines. Always
            shown so every eval-result view -- fresh read-aloud or a saved
            recording rescored later -- reads the same. Sits next to the
            heading, before the reference-words line, so it reads as part of
            what the section IS rather than a control tucked over on the right. */}
        <Segmented<FeedFilter>
          value={feedFilter}
          options={[
            { value: "all", label: "All" },
            { value: "live", label: "Live" },
            { value: "batch", label: "Batch" },
          ]}
          onChange={setFeedFilter}
          label="Filter by feed mode"
        />
        <span className="mono muted">
          {scored
            ? `scored against the reference · ${first?.refWordCount ?? referenceWords} reference words · normalization on`
            : "no reference for this recording — timings only, no error rates"}
        </span>
        <button
          type="button"
          className="ghost-btn"
          onClick={() => downloadReport(buildTranscriptReportHtml(visibleRuns, engines))}
        >
          Export report →
        </button>
      </div>

      <div className="scorecard-grid">
        {TILES.map((tile) => {
          const values = visibleRuns
            .map((run) => tile.valueOf(run))
            .filter((value): value is number => value != null);
          const best = values.length
            ? tile.lowerIsBetter ? Math.min(...values) : Math.max(...values)
            : null;
          const worst = values.length
            ? tile.lowerIsBetter ? Math.max(...values) : Math.min(...values)
            : null;
          // Bar length is a share of the LARGEST value present, so the longest
          // bar is the biggest number regardless of which direction is good.
          const scale = values.length ? Math.max(...values) : null;
          // Ranked best-first, so the reader gets the ordering from the layout
          // and does not have to scan for the smallest number. A run with no
          // figure (queued, or real-time bound) sorts last rather than counting
          // as a zero, which would rank an unmeasured engine as the winner.
          // Copied before sorting: `visibleRuns` is derived from the caller's
          // array and the other tiles sort it differently. oxlint flags the
          // bare .sort(); the spread is the fix, and `toSorted` is not
          // available at this project's TS lib target.
          const ranked = [...visibleRuns].sort((a, b) => {
            const av = tile.valueOf(a);
            const bv = tile.valueOf(b);
            if (av == null && bv == null) return 0;
            if (av == null) return 1;
            if (bv == null) return -1;
            return tile.lowerIsBetter ? av - bv : bv - av;
          });

          return (
            <div className="score-tile" key={tile.title}>
              <strong>{tile.title}</strong>
              <small className="muted">{tile.hint}</small>
              <span className="eyebrow">
                Bar length = measured value · ranked best first
              </span>
              {ranked.map((run) => {
                const value = tile.valueOf(run);
                // A run the worker has not finished yet reports THAT, ahead of
                // any per-tile reason: "—" beside a finished engine's number
                // reads as "this engine scored nothing", which is a different
                // claim from "this engine has not answered yet".
                const pending = run.status === "queued" || run.status === "running";
                const reason = pending
                  ? (run.status === "running" ? "transcribing…" : "queued…")
                  : tile.unavailable?.(run) ?? null;
                // Proportional to the value itself. A floor of 1.5% keeps a
                // genuine near-zero (a 0.0% error rate) visible as a sliver
                // rather than as nothing, which would read as "not measured".
                let fill = 0;
                if (value != null && scale != null && scale > 0) {
                  fill = Math.max(1.5, (value / scale) * 100);
                } else if (value != null) {
                  fill = 1.5; // every value is 0 in this tile
                }
                // Colour carries the direction that bar length no longer does.
                // Only applied once there is a spread to rank: with every engine
                // on the same number, nothing is best or worst.
                const rankClass =
                  value == null || best == null || best === worst
                    ? ""
                    : value === best
                      ? " is-best"
                      : value === worst
                        ? " is-worst"
                        : "";
                const transport = transportOf(run);
                return (
                  <div className={`score-row${pending ? " is-pending" : ""}`} key={keyOf(run)}>
                    <div className="score-row-head">
                      <span>
                        <span className={`engine-dot ${transport ?? ""}`} aria-hidden="true" />
                        {nameOf(run)}
                      </span>
                      <b className="mono">
                        {value != null ? tile.format(value) : <span className="muted">{reason ?? "—"}</span>}
                      </b>
                    </div>
                    <div className={`score-bar${rankClass}`}>
                      <i style={{ width: `${Math.max(0, Math.min(100, fill))}%` }} />
                    </div>
                    {/* The transport is on every row, not in a footnote: it is
                        what the figure above is a measurement OF. */}
                    <small className="score-transport mono">
                      {transport === "stream"
                        ? "stream · VAD"
                        : transport === "file"
                          // Never falls through to "chunks": this engine was fed the
                          // whole recording in one call, after the fact, and reading
                          // it as a chunked live run would credit it with a boundary
                          // cost it never paid.
                          ? "file · whole recording"
                          : run.chunkIntervalSec
                            ? `chunks · ${run.chunkIntervalSec.toFixed(1)}s`
                            : "chunks"}
                      {run.source ? ` · ${run.replayed ? "live" : run.source}` : ""}
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
          {visibleRuns.map((run) => (
            <div key={keyOf(run)} className="ops-row">
              {/* The mode is part of the label, not a footnote: two rows for one
                  engine differ only by it, and an unlabelled pair reads as the
                  same run measured twice. */}
              <span>
                {nameOf(run)}
                <span className="muted">
                  {" · "}{run.source === "live" ? (run.replayed ? "live" : "stream") : "batch"}
                </span>
              </span>
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
