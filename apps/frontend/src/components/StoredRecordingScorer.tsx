import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { TranscriptReference, TranscriptRun } from "../types/diarization";
import type { TranscriptEngineInfo } from "../adapters";
import { fetchReference, fetchTranscripts, putReference, startTranscripts } from "../adapters";
import { TranscriptScorecard } from "./TranscriptScorecard";

interface StoredRecordingScorerProps {
  audioFileId: number;
  fileName: string;
  engines: TranscriptEngineInfo[];
  pollIntervalMs: number;
}

/**
 * Score a recording that already exists, against a reference supplied by hand.
 *
 * The second entry point to the comparison. The read-aloud flow scores what each
 * engine returned live; this one runs the engines over stored audio, so its rows
 * are `batch` and the scorecard labels them as such. A batch number and a live
 * number are different measurements of the same engine, and nothing here lets
 * them be mistaken for each other.
 *
 * Replacing the reference clears the previous scores on the backend rather than
 * recomputing them. That is deliberate: scores are written where a run finishes,
 * and a stale WER beside a new reference would be worse than an empty one.
 */
export function StoredRecordingScorer({
  audioFileId,
  fileName,
  engines,
  pollIntervalMs,
}: StoredRecordingScorerProps) {
  const [reference, setReference] = useState<TranscriptReference | null>(null);
  const [draft, setDraft] = useState("");
  const [runs, setRuns] = useState<TranscriptRun[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<number | null>(null);

  // Load whatever this recording already has, so reopening it shows its scores
  // instead of an empty form.
  useEffect(() => {
    let cancelled = false;
    setError(null);
    void (async () => {
      try {
        const [existing, transcripts] = await Promise.all([
          fetchReference(audioFileId),
          fetchTranscripts(audioFileId),
        ]);
        if (cancelled) return;
        setReference(existing);
        setDraft(existing?.text ?? "");
        setRuns(transcripts);
      } catch (caught) {
        if (!cancelled) setError((caught as Error).message);
      }
    })();
    return () => { cancelled = true; };
  }, [audioFileId]);

  // Only the engines this surface compares. A recording can hold transcripts from
  // engines outside the comparison — every upload auto-transcribes with the
  // default engine — and letting those into the scorecard would add a column
  // nobody asked to compare, with no score in it. They remain visible on the
  // Live Speech panel, which is where a single recording's transcripts belong.
  const compared = useMemo(() => {
    const ids = new Set(engines.map((engine) => engine.asrId));
    return runs.filter((run) => ids.has(run.asrId));
  }, [runs, engines]);

  const inFlight = compared.some((run) => run.status === "queued" || run.status === "running");

  // Poll only while something is actually running, on the host's own interval.
  const poll = useCallback(async () => {
    try {
      setRuns(await fetchTranscripts(audioFileId));
    } catch {
      // A failed poll is not worth surfacing: the next one either recovers or
      // the run's own error field explains what happened.
    }
  }, [audioFileId]);

  useEffect(() => {
    if (!inFlight) {
      if (timerRef.current != null) window.clearInterval(timerRef.current);
      timerRef.current = null;
      return;
    }
    timerRef.current = window.setInterval(() => void poll(), pollIntervalMs);
    return () => {
      if (timerRef.current != null) window.clearInterval(timerRef.current);
      timerRef.current = null;
    };
  }, [inFlight, poll, pollIntervalMs]);

  const handleRun = async () => {
    const text = draft.trim();
    if (!text || busy || !engines.length) return;
    setBusy(true);
    setError(null);
    try {
      setReference(await putReference(audioFileId, text, "pasted"));
      setRuns(await startTranscripts(audioFileId, engines.map((engine) => engine.asrId)));
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const referenceWords = draft.trim() ? draft.trim().split(/\s+/).length : 0;

  return (
    <>
      <section className="panel stored-scorer">
        <div className="stored-head">
          <div>
            <h2>Score a loaded recording</h2>
            <p className="muted">
              Paste what was actually said in <strong>{fileName}</strong> and run every engine over
              it. These runs read stored audio, so they are labelled <span className="mono">batch</span> —
              not the same measurement as a live read-aloud.
            </p>
          </div>
          <span className="mono muted">{referenceWords} reference words</span>
        </div>
        <textarea
          className="script-paste"
          dir="auto"
          placeholder="Paste the reference transcript for this recording…"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          disabled={busy || inFlight}
        />
        <div className="stored-actions">
          <button
            type="button"
            className="primary-btn"
            onClick={handleRun}
            disabled={!draft.trim() || busy || inFlight || !engines.length}
            title={
              !engines.length
                ? "No transcription engine is configured on this host"
                : !draft.trim()
                  ? "A reference is needed before anything can be scored"
                  : `Run ${engines.map((engine) => engine.name).join(" and ")}`
            }
          >
            {busy ? "Queueing…" : inFlight ? "Running…" : reference ? "Re-run and score" : "Run and score"}
          </button>
          {reference && (
            <small className="muted">
              Current reference: {reference.wordCount} words ({reference.source})
            </small>
          )}
        </div>
        {error && <p className="transcript-error">{error}</p>}
      </section>

      {compared.length > 0 && (
        <TranscriptScorecard runs={compared} engines={engines} referenceWords={referenceWords} />
      )}
    </>
  );
}
