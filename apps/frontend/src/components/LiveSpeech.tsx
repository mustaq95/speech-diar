import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import type { ActiveMap, ModelRun } from "../types";
import type { TranscriptionMode, TranscriptRun } from "../types/diarization";
import type { RuntimeConfig } from "../adapters";

const MODES: TranscriptionMode[] = ["online", "offline"];
import { SPEAKER_COLORS } from "../data";
import { activeSpeakers, hexA } from "../utils";

interface LiveSpeechProps {
  /** One run per engine that has transcribed this recording; empty when none
   * was ever started (or none is loaded). The mode toggle picks between them. */
  transcripts: TranscriptRun[];
  /** Models available to tint words by; only enabled ones are offered. */
  models: ModelRun[];
  active: ActiveMap;
  /** Per-mode engine name + availability. Null until `GET /config` lands. */
  runtimeConfig: RuntimeConfig | null;
  /** Undefined for the demo and before anything is uploaded — nothing real to transcribe. */
  onRun?: (mode: TranscriptionMode) => Promise<void>;
  /** Registers the playback-time callback that moves the current-word highlight. */
  wordSyncRef: (sync: ((time: number) => void) | null) => void;
}

/** A word is only tintable/highlightable once the aligner has placed it. */
interface PlacedWord {
  w: string;
  s: number;
  e: number;
  spk: number | null;
  overlap: boolean;
}

function fmtMs(ms: number): string {
  return ms >= 10_000 ? `${(ms / 1000).toFixed(0)}s` : `${(ms / 1000).toFixed(1)}s`;
}

/** Word spans are usually a fraction of a second, so the timeline's m:ss
 * formatting collapses start and end to the same value ("0:03–0:03"). Word
 * tooltips need sub-second resolution to say anything at all. */
function fmtWordTime(seconds: number): string {
  return `${seconds.toFixed(2)}s`;
}

/**
 * Rightmost word whose start is <= time. Binary search rather than a scan
 * because this runs on every animation frame over a transcript that can be
 * several thousand words.
 */
function wordIndexAt(starts: number[], time: number): number {
  let lo = 0;
  let hi = starts.length - 1;
  let found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (starts[mid] <= time) {
      found = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return found;
}

/**
 * The live-speech transcript, replacing the old event feed.
 *
 * Two things about this panel are deliberate:
 *
 * 1. **The highlight never goes through React state.** App.tsx drives playback
 *    with direct DOM writes at 60fps (`syncDom`) and only commits `setTime`
 *    when the set of speaking models changes — see commit 147312b, "Fix tab
 *    freeze on rapid timeline seeking". A state update per spoken word would
 *    undo exactly that. So the word spans render once, and `wordSyncRef` hands
 *    App.tsx a callback that moves a single class between two spans.
 *
 * 2. **Speaker tint is labelled, not implied.** No ASR engine here reports
 *    speakers, so the colors come from whichever diarization model is picked in
 *    the header — which is why that picker is visible rather than a hidden
 *    default. A word covered by two speakers gets the amber overlap treatment
 *    instead of one of them being chosen arbitrarily.
 *
 * 3. **The mode toggle selects a stored run, it does not relabel one.** A
 *    recording keeps one transcript per engine, so ONLINE and OFFLINE each show
 *    the words that engine actually produced, with that engine's own timings.
 */
export function LiveSpeech({ transcripts, models, active, runtimeConfig, onRun, wordSyncRef }: LiveSpeechProps) {
  const tintable = useMemo(() => models.filter((model) => active[model.id] && model.segs.length > 0), [models, active]);
  const [tintId, setTintId] = useState<string | null>(null);
  const effectiveTintId = tintId && tintable.some((m) => m.id === tintId) ? tintId : tintable[0]?.id ?? null;
  const tintModel = tintable.find((model) => model.id === effectiveTintId) ?? null;

  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);

  // The selected mode: which stored transcript is shown, and what a run started
  // now would use. `null` means "not chosen yet". With exactly one transcript it
  // falls back to that run's mode, so reopening a recording shows the words it
  // has instead of an empty panel; with both (nothing to infer) or none, the
  // host default. An explicit pick from the toggle wins and sticks. Same
  // nullable-override pattern as the tint picker above.
  const [modeOverride, setModeOverride] = useState<TranscriptionMode | null>(null);
  const modes = runtimeConfig?.transcriptionModes ?? null;
  const soleTranscriptMode = transcripts.length === 1 ? transcripts[0].mode : null;
  const mode: TranscriptionMode =
    modeOverride ?? soleTranscriptMode ?? runtimeConfig?.defaultTranscriptionMode ?? "offline";

  // The selected mode's own run is what the panel shows. When that mode has not
  // been run yet, the other engine's transcript is shown rather than an empty
  // panel — attributed, never passed off as this mode's output.
  const selected = transcripts.find((run) => run.mode === mode) ?? null;
  const transcript = selected ?? transcripts[0] ?? null;
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const spansRef = useRef<Array<HTMLSpanElement | null>>([]);
  const currentRef = useRef(-1);

  // Placement and tint are computed once per (transcript, tint model) — never
  // per frame. An unaligned word (no start/end) is kept in the text but cannot
  // be highlighted, so it is excluded here rather than given a made-up time.
  const placed = useMemo<PlacedWord[]>(() => {
    if (!transcript) return [];
    return transcript.words.flatMap((word) => {
      if (word.s == null || word.e == null) return [];
      const mid = (word.s + word.e) / 2;
      const speakers = tintModel ? activeSpeakers(tintModel, mid) : [];
      return [{
        w: word.w,
        s: word.s,
        e: word.e,
        spk: speakers.length === 1 ? speakers[0] : null,
        overlap: speakers.length > 1,
      }];
    });
  }, [transcript, tintModel]);

  const starts = useMemo(() => placed.map((word) => word.s), [placed]);

  // Register the hot-path callback. It reads only refs and the memoized
  // `starts`, so it never triggers a render; it just moves the class and keeps
  // the active word in view.
  useEffect(() => {
    spansRef.current = spansRef.current.slice(0, placed.length);
    currentRef.current = -1;
    if (placed.length === 0) {
      wordSyncRef(null);
      return () => wordSyncRef(null);
    }
    const sync = (time: number) => {
      const index = wordIndexAt(starts, time);
      // Past the end of a word with no successor yet: keep the last word lit
      // rather than flickering off between words.
      if (index === currentRef.current) return;
      spansRef.current[currentRef.current]?.classList.remove("is-current");
      currentRef.current = index;
      const node = spansRef.current[index];
      if (!node) return;
      node.classList.add("is-current");
      const body = bodyRef.current;
      if (!body) return;
      const offset = node.offsetTop - body.offsetTop;
      if (offset < body.scrollTop || offset > body.scrollTop + body.clientHeight - node.offsetHeight * 2) {
        body.scrollTo({ top: Math.max(0, offset - body.clientHeight / 2) });
      }
    };
    wordSyncRef(sync);
    return () => wordSyncRef(null);
  }, [placed, starts, wordSyncRef]);

  const handleRun = async () => {
    if (!onRun || running) return;
    setRunning(true);
    setRunError(null);
    try {
      await onRun(mode);
    } catch (error) {
      setRunError((error as Error).message);
    } finally {
      setRunning(false);
    }
  };

  // Now that both modes' runs are kept, this only fires when the selected mode
  // has no run of its own and the other engine's is standing in. The panel says
  // so rather than relabelling those words or hiding them.
  const selectedAsrName = modes?.[mode]?.asrName ?? "";
  const staleMode = transcript != null && transcript.mode !== mode;
  const hasCurrentModeTranscript = selected != null;
  const engineUnconfigured = modes != null && !modes[mode].configured;

  const runButton = onRun ? (
    <button
      type="button"
      className="ghost-btn"
      onClick={handleRun}
      disabled={running || engineUnconfigured}
      title={
        engineUnconfigured
          ? `${selectedAsrName} is not configured on this host — check .env`
          : `Transcribe with ${selectedAsrName || "the selected engine"}`
      }
    >
      {running ? "Queuing…" : hasCurrentModeTranscript ? "Re-run" : "Run transcript"}
    </button>
  ) : null;

  return (
    <>
      <div className="feed-title live-speech-head">
        <h2>Live speech</h2>
        {tintable.length > 1 && (
          <select
            className="tint-select"
            value={effectiveTintId ?? ""}
            onChange={(event) => setTintId(event.target.value)}
            title="Which diarization model's speaker colors tint the words"
          >
            {tintable.map((model) => (
              <option key={model.id} value={model.id}>{model.short}</option>
            ))}
          </select>
        )}
      </div>

      <div className="live-speech-meta">
        {/* The toggle picks which engine's transcript is shown, and what a run
            started now would use — present even with no transcript at all. A
            side whose engine is unconfigured on this host is shown but disabled,
            so the choice is visible, not hidden. Picking a mode that has no run
            yet keeps the other one on screen, attributed below. */}
        {modes && (
          <div className="mode-toggle" role="group" aria-label="Transcription mode">
            {MODES.map((m) => {
              const avail = modes[m];
              return (
                <button
                  key={m}
                  type="button"
                  className={`mode-chip ${m}${mode === m ? " is-active" : ""}`}
                  aria-pressed={mode === m}
                  disabled={!avail.configured}
                  onClick={() => setModeOverride(m)}
                  title={
                    !avail.configured
                      ? `${avail.asrName} is not configured on this host — check .env`
                      : m === "offline"
                        ? `${avail.asrName} — transcribes on this host; the audio never leaves the machine`
                        : `${avail.asrName} — transcribes via a remote service; the audio is streamed off this host`
                  }
                >
                  {m}
                </button>
              );
            })}
          </div>
        )}
        {hasCurrentModeTranscript && transcript ? (
          <>
            <span className="mono">
              {transcript.asrName}
              {transcript.asrMs != null && (
                <b
                  title={
                    transcript.mode === "online"
                      ? "Wall clock. This engine streams audio in real time, so the figure tracks the recording's length, not the model's speed."
                      : "Wall clock for local GPU inference."
                  }
                >
                  {" "}{fmtMs(transcript.asrMs)}
                </b>
              )}
            </span>
            {transcript.alignMs != null && (
              <span className="mono">
                {transcript.alignerName}<b> {fmtMs(transcript.alignMs)}</b>
              </span>
            )}
          </>
        ) : (
          <span className="mono">{selectedAsrName}</span>
        )}
        {runButton}
      </div>

      {/* The selected mode has no run yet, so the other engine's is on screen.
          Attributed rather than hidden: a transcript that took real time to make
          is worth showing, and must not be passed off as this mode's output. */}
      {staleMode && transcript && (
        <div className="live-speech-attrib">
          showing an earlier <b>{transcript.mode}</b> run · {transcript.asrName}
          {transcript.asrMs != null && ` ${fmtMs(transcript.asrMs)}`}
          {transcript.alignMs != null && ` · ${transcript.alignerName} ${fmtMs(transcript.alignMs)}`}
        </div>
      )}

      <div className="event-feed live-speech" ref={bodyRef} dir="auto">
        {/* The run button lives in the meta row above, in EVERY state — a
            finished transcript used to offer no action at all, which left no
            way to re-run after switching the mode toggle. */}
        {runError && <p className="live-speech-error">{runError}</p>}
        {!transcript ? (
          <div className="feed-empty">No transcript for this recording.</div>
        ) : transcript.status === "queued" ? (
          <div className="feed-empty">Transcript queued…</div>
        ) : transcript.status === "failed" ? (
          <div className="feed-empty">
            {/* The ASR text survives an alignment failure, so show whatever was
                produced rather than replacing it with the error alone. */}
            {transcript.text && <p className="live-speech-partial" dir="auto">{transcript.text}</p>}
            <p className="live-speech-error">{transcript.error ?? "Transcription failed"}</p>
          </div>
        ) : placed.length > 0 ? (
          <p className="live-speech-words" dir="auto">
            {placed.map((word, index) => {
              const color = word.spk != null ? SPEAKER_COLORS[word.spk % SPEAKER_COLORS.length] : null;
              return (
                // The separating space is a text node BETWEEN the spans, not
                // inside them: adjacent spans render with no gap at all, and
                // putting the space inside would stretch the highlight box past
                // the word. Load-bearing for Arabic too, where the words
                // otherwise run together in RTL.
                <Fragment key={`${index}-${word.s}`}>
                  {index > 0 && " "}
                  <span
                    ref={(node) => { spansRef.current[index] = node; }}
                    className={`ls-word${word.overlap ? " is-overlap" : ""}`}
                    style={color ? { color, background: hexA(color, 0.1) } : undefined}
                    title={`${fmtWordTime(word.s)}–${fmtWordTime(word.e)}`}
                  >
                    {word.w}
                  </span>
                </Fragment>
              );
            })}
          </p>
        ) : transcript.text ? (
          <>
            <p className="live-speech-partial" dir="auto">{transcript.text}</p>
            <div className="feed-empty">
              {transcript.stage === "align" ? <><span className="spinner small" /> Aligning words…</> : "No words could be aligned."}
            </div>
          </>
        ) : (
          <div className="feed-empty">
            <span className="spinner small" /> Transcribing…
          </div>
        )}
      </div>
    </>
  );
}
