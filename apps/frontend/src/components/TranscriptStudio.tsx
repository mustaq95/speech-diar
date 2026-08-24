import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Segmented, SliderControl } from "./controls";
import type { GeneratedScript, TranscriptReference, TranscriptRun } from "../types/diarization";
import type { RuntimeConfig, TranscriptEngineInfo } from "../adapters";
import {
  audioStreamUrl,
  fetchReference,
  fetchTranscripts,
  finalizeLiveSession,
  generateScript,
  liveStreamUrl,
  openLiveSession,
  sendLiveChunk,
} from "../adapters";
import { startRecording, type Recorder } from "../recording";
import { StoredRecordingScorer } from "./StoredRecordingScorer";
import { TranscriptScorecard } from "./TranscriptScorecard";
import { TtsStudio } from "./TtsStudio";

/** Human labels for the .env-supplied option ids. An id with no label here still
 * renders (as itself), so adding one to .env never blanks a control. */
const MIX_LABELS: Record<string, string> = {
  ar: "Arabic only",
  "mixed-50-50": "Mixed 50/50",
  en: "English",
};
const HARD_CASE_LABELS: Record<string, string> = {
  "proper-nouns": "Proper nouns",
  "numbers-dates": "Numbers & dates",
  "emirati-dialect": "Emirati dialect",
  // Retired id, same instruction under its old name. Labelled identically because
  // the distinction is internal; without an entry a stale config would render the
  // raw id as the chip text.
  "gulf-dialect": "Emirati dialect",
  "fast-speech": "Fast speech",
};

/** What one engine has produced so far in the current live run. */
interface LivePanel {
  text: string;
  latencies: number[];
  chunkCount: number;
  error: string | null;
}

const EMPTY_PANEL: LivePanel = { text: "", latencies: [], chunkCount: 0, error: null };

interface TranscriptStudioProps {
  /** Null until GET /config lands; the surface stays disabled until it does
   * rather than guessing at engines or control bounds. */
  runtimeConfig: RuntimeConfig | null;
  /** The recording currently open on the dashboard, when there is one. Enables
   * the second entry point: scoring stored audio against a pasted reference. */
  audioFileId?: number | null;
  fileName?: string;
  /** Whether the opened recording has audio yet. False for an entry created when
   * its script was generated but never read aloud: that one opens ready to RECORD,
   * not ready to play back. */
  hasAudio?: boolean;
  /** Which sub-activity this page is doing: scoring STT engines live, or
   * comparing TTS engines' synthesis of the same reference. Explicit state owned
   * by App.tsx, never inferred from what happens to be open. */
  transcriptSubMode: "stt" | "tts";
  onTranscriptSubMode: (next: "stt" | "tts") => void;
  /** Called once a capture is finalized and its backend row exists, so the
   * recordings list can pick it up. Without this the recording was stored but
   * never appeared anywhere in the UI. */
  onRecorded?: (audioFileId: number) => void;
  /** Called when a generated script is saved, so the list picks up the new entry.
   * Distinct from `onRecorded`: nothing has been recorded, and the user is still on
   * this page, so it must refresh the list without navigating away. */
  onScriptSaved?: (audioFileId: number) => void;
}

function fmtMs(ms: number): string {
  return ms >= 10_000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}

function mean(values: number[]): number | null {
  if (!values.length) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

/**
 * The transcript-evaluation surface: generate a script, read it aloud, and
 * compare what each engine returned against it.
 *
 * A full page rather than a pane inside the dashboard's studio layout, for a
 * structural reason: this flow starts with NO recording loaded (it records one),
 * and the dashboard only exists once a recording is open.
 *
 * Two transports run at once, because the engines have different native ones and
 * matching them artificially would mean handicapping one. A streaming engine gets
 * a WebSocket carrying continuous PCM; a request/response engine gets one short
 * WAV per interval. Every figure on screen is labelled with which, because a
 * stream's lag and a chunk's round-trip are not the same measurement.
 */
export function TranscriptStudio({
  runtimeConfig,
  audioFileId,
  fileName,
  hasAudio = true,
  transcriptSubMode,
  onTranscriptSubMode,
  onRecorded,
  onScriptSaved,
}: TranscriptStudioProps) {
  const config = runtimeConfig?.transcript ?? null;
  const engines = useMemo<TranscriptEngineInfo[]>(
    () => (config?.engines ?? []).filter((engine) => engine.configured),
    [config],
  );
  // Only for the header caption; TtsStudio owns the TTS panel itself.
  const ttsEngineCount = (runtimeConfig?.tts?.engines ?? []).filter((engine) => engine.configured).length;

  // --- a saved recording, restored ----------------------------------------
  // Opening one puts the whole session back: the script it was read from, the
  // settings it was generated with, its audio for playback, and what each engine
  // returned. Nothing here is re-run and nothing is re-scored — the numbers on
  // screen are the ones that were measured at the time.
  const [savedRef, setSavedRef] = useState<TranscriptReference | null>(null);
  const [savedRuns, setSavedRuns] = useState<TranscriptRun[]>([]);
  const [loadingSaved, setLoadingSaved] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [savedDuration, setSavedDuration] = useState<number | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    setDraftEdit(null);
    if (audioFileId == null) {
      setSavedRef(null);
      setSavedRuns([]);
      return;
    }
    let cancelled = false;
    setLoadingSaved(true);
    void (async () => {
      try {
        const [reference, runs] = await Promise.all([
          fetchReference(audioFileId),
          fetchTranscripts(audioFileId),
        ]);
        if (cancelled) return;
        setSavedRef(reference);
        setSavedRuns(runs);
      } catch (error) {
        if (!cancelled) console.error("Could not load the saved recording:", error);
      } finally {
        if (!cancelled) setLoadingSaved(false);
      }
    })();
    return () => { cancelled = true; };
  }, [audioFileId]);

  // A recording with a stored reference AND audio is a finished comparison: play it
  // back, show its scorecard, no paste-and-score panel (that only earns its place on
  // a recording with no ground truth yet, e.g. a diarization upload).
  const viewingSaved = audioFileId != null && savedRef != null && hasAudio;

  // A stored reference with NO audio is a script that was generated and never read.
  // Same pre-filled script and settings, but the recorder is live: this entry is a
  // work item, and Stop attaches the recording to this very row.
  const resumingScript = audioFileId != null && savedRef != null && !hasAudio;

  // The controls are restored from the params the script was generated with, so a
  // reopened recording shows what was actually asked for rather than the defaults.
  const savedParams = (savedRef?.params ?? null) as
    | { minutes?: number; languageMix?: string; hardCases?: string[]; generatorModel?: string;
        languageSplit?: { arabic: number; english: number } }
    | null;

  // --- script controls, seeded from the host's own options -----------------
  const lengths = config?.scriptLengthsMin ?? [];
  const [minutes, setMinutes] = useState<number | null>(null);
  const [mix, setMix] = useState<string | null>(null);
  const [hardCases, setHardCases] = useState<string[] | null>(null);
  // Explicit pick > the reopened recording's own params > the host's defaults.
  const effectiveMinutes = minutes ?? savedParams?.minutes ?? lengths[1] ?? lengths[0] ?? 1;
  const effectiveMix =
    mix ?? savedParams?.languageMix ?? config?.scriptLanguageMixes?.[1]
    ?? config?.scriptLanguageMixes?.[0] ?? "";
  const effectiveHardCases =
    hardCases ?? savedParams?.hardCases ?? config?.scriptHardCases?.slice(0, 4) ?? [];

  const [script, setScript] = useState<GeneratedScript | null>(null);
  // The script box's contents. `null` means "follow whatever is stored"; any
  // string means the operator has typed and owns the value until it is saved.
  // Replaces the old paste-only box: the script is editable whether it was
  // generated, restored, or typed here.
  const [draftEdit, setDraftEdit] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);
  const [scriptError, setScriptError] = useState<string | null>(null);

  // --- live run -----------------------------------------------------------
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [peaks, setPeaks] = useState<number[]>([]);
  const [panels, setPanels] = useState<Record<string, LivePanel>>({});
  const [runError, setRunError] = useState<string | null>(null);
  const [finalizing, setFinalizing] = useState(false);
  const [results, setResults] = useState<TranscriptRun[] | null>(null);


  const recorderRef = useRef<Recorder | null>(null);
  const socketsRef = useRef<Map<string, WebSocket>>(new Map());
  const sessionRef = useRef<string | null>(null);
  const tickRef = useRef<number | null>(null);
  /** Waveform peaks buffered between renders.
   *
   * The audio callback must stay latency-critical — it forwards PCM to the
   * streaming engine — so it never touches React state. Shrinking the block to
   * 64 ms would otherwise mean ~15 renders a second over a few hundred bars.
   * Peaks land here and the clock tick below flushes them, which decouples how
   * fast audio moves from how often the waveform repaints. */
  const peaksRef = useRef<number[]>([]);

  // Same precedence as `shownScript` below. Generating is an explicit action, so a
  // fresh script outranks a stored one; otherwise the panel showed the new script
  // while the run was scored against the old text.
  // What is STORED for this recording, before any local edit.
  const storedText = script?.text ?? savedRef?.text ?? "";
  const draft = draftEdit ?? storedText;
  // What the engines are scored against: the draft, since that is what will be
  // read aloud and what finalize writes as the reference.
  const referenceText = draft.trim();
  const dirty = referenceText !== storedText.trim();

  /** The recording row this capture belongs to: the one created with the script just
   * generated, or the script-only entry being resumed. Null for a pasted reference,
   * which has no row until finalize creates one. */
  const pendingAudioFileId =
    script?.audioFileId ?? (resumingScript ? audioFileId ?? null : null);
  const referenceWords = referenceText ? referenceText.split(/\s+/).length : 0;
  const chunkInterval = config?.chunkIntervalSec ?? 3;
  const socketTimeoutSec = config?.socketOpenTimeoutSec ?? 10;

  // Live results win while a capture is on screen; otherwise a reopened
  // recording's stored runs. Same shape either way, so the scorecard is unchanged.
  const shownRuns: TranscriptRun[] | null = results ?? (viewingSaved ? savedRuns : null);
  const runByEngine = useMemo(
    () => new Map((shownRuns ?? []).map((run) => [run.asrId, run])),
    [shownRuns],
  );

  // The script card's content, from whichever source exists. A reopened recording
  // shows the script it was READ FROM, with the settings it was generated with —
  // not an empty paste box, which is what it used to fall back to.
  const shownScript = useMemo(() => {
    if (script) {
      return {
        label: "GENERATED SCRIPT",
        text: script.text,
        wordCount: script.wordCount,
        minutes: script.params?.minutes as number | undefined,
        generatorModel: script.generatorModel,
        split: script.params?.languageSplit as { arabic: number; english: number } | undefined,
      };
    }
    if (savedRef) {
      return {
        // A pasted reference was never generated, so it is not called a script.
        label: savedRef.source === "script" ? "GENERATED SCRIPT" : "PASTED REFERENCE",
        text: savedRef.text,
        wordCount: savedRef.wordCount,
        minutes: savedParams?.minutes,
        generatorModel: savedParams?.generatorModel,
        split: savedParams?.languageSplit,
      };
    }
    return null;
  }, [script, savedRef, savedParams]);

  const streamEngines = useMemo(() => engines.filter((e) => e.transport === "stream"), [engines]);
  const chunkEngines = useMemo(() => engines.filter((e) => e.transport === "chunks"), [engines]);

  const cleanupRun = useCallback(() => {
    if (tickRef.current != null) window.clearInterval(tickRef.current);
    tickRef.current = null;
    for (const socket of socketsRef.current.values()) {
      if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: "stop" }));
      socket.close();
    }
    socketsRef.current.clear();
  }, []);

  // Releasing the microphone is not optional on unmount: a live capture and an
  // open socket would otherwise outlive the page.
  useEffect(() => () => {
    cleanupRun();
    void recorderRef.current?.stop();
  }, [cleanupRun]);

  const handleGenerate = async () => {
    if (generating) return;
    setGenerating(true);
    setScriptError(null);
    try {
      const generated = await generateScript({
        minutes: effectiveMinutes,
        languageMix: effectiveMix,
        hardCases: effectiveHardCases,
      });
      setScript(generated);
      setDraftEdit(null);
      setResults(null);
      // Drop the explicit picks so the controls fall through to the saved
      // params: what the script was actually generated with, which is also what
      // the other sub-mode reads. Without this each side keeps its own picks and
      // the two disagree about settings for one shared script.
      setMinutes(null);
      setMix(null);
      setHardCases(null);

      // The script is saved server-side as it is generated, so the entry exists now
      // rather than at Stop. Tell the list without leaving the page.
      onScriptSaved?.(generated.audioFileId);
    } catch (error) {
      setScriptError((error as Error).message);
    } finally {
      setGenerating(false);
    }
  };

  const toggleHardCase = (id: string) => {
    const current = effectiveHardCases;
    setHardCases(current.includes(id) ? current.filter((c) => c !== id) : [...current, id]);
  };

  const handleStart = async () => {
    if (recording || !engines.length) return;
    setRunError(null);
    setResults(null);
    peaksRef.current = [];
    setPeaks([]);
    setElapsed(0);
    setPanels(Object.fromEntries(engines.map((engine) => [engine.asrId, { ...EMPTY_PANEL }])));

    try {
      const session = await openLiveSession(
        engines.map((engine) => engine.asrId),
        chunkInterval,
        referenceText,
      );
      sessionRef.current = session.sessionId;

      // One socket per streaming engine, all opened BEFORE capture starts so no
      // audio is captured that has nowhere to go.
      const opening: Array<Promise<void>> = [];
      for (const engine of streamEngines) {
        const socket = new WebSocket(liveStreamUrl(session.sessionId, engine.asrId));
        socket.binaryType = "arraybuffer";
        socket.addEventListener("message", (event) => {
          const message = JSON.parse(event.data as string);
          if (message.type === "transcript") {
            setPanels((previous) => {
              const panel = previous[engine.asrId] ?? EMPTY_PANEL;
              return {
                ...previous,
                [engine.asrId]: {
                  ...panel,
                  text: panel.text ? `${panel.text} ${message.text}` : message.text,
                  latencies: [...panel.latencies, message.latencyMs],
                  chunkCount: panel.chunkCount + 1,
                },
              };
            });
          } else if (message.type === "error") {
            setPanels((previous) => ({
              ...previous,
              [engine.asrId]: { ...(previous[engine.asrId] ?? EMPTY_PANEL), error: message.message },
            }));
          }
        });
        socket.addEventListener("error", () => {
          setPanels((previous) => ({
            ...previous,
            [engine.asrId]: { ...(previous[engine.asrId] ?? EMPTY_PANEL), error: "socket failed" },
          }));
        });
        socketsRef.current.set(engine.asrId, socket);
        opening.push(
          new Promise<void>((resolve, reject) => {
            // The deadline is the point. A dropped upgrade — a dev proxy without
            // `ws: true` is the one that bit us — produces a socket that never
            // opens and never errors, so waiting on open/error alone hangs here
            // forever and the record button just does nothing. Failing loudly
            // after a bounded wait is what makes that diagnosable.
            const deadline = window.setTimeout(
              () => reject(new Error(
                `${engine.name}: the audio socket did not open within ${socketTimeoutSec}s. ` +
                "If this is the dev server, check that vite.config.ts proxies /api with ws: true.",
              )),
              socketTimeoutSec * 1000,
            );
            socket.addEventListener("open", () => { window.clearTimeout(deadline); resolve(); }, { once: true });
            socket.addEventListener("error", () => {
              window.clearTimeout(deadline);
              reject(new Error(`${engine.name}: the audio socket failed to connect`));
            }, { once: true });
          }),
        );
      }
      // All sockets open concurrently: they are independent connections, and
      // opening them one after another would delay capture by the sum of their
      // handshakes instead of the slowest one.
      await Promise.all(opening);

      const recorder = await startRecording({
        sampleRate: session.sampleRate,
        chunkIntervalSec: session.chunkIntervalSec,
        blockSamples: config?.recordBlockSamples ?? 1024,
        onBlock: ({ pcm, peak }) => {
          // Forwarded first and synchronously: this is the streaming engine's
          // audio path and nothing may sit in front of it.
          for (const socket of socketsRef.current.values()) {
            if (socket.readyState === WebSocket.OPEN) socket.send(pcm.buffer);
          }
          // Buffered, not committed to state. Capped because a long read must not
          // grow an unbounded array behind the waveform, and only the recent
          // shape is on screen anyway.
          const buffered = peaksRef.current;
          buffered.push(peak);
          if (buffered.length > 400) buffered.splice(0, buffered.length - 400);
        },
        onChunk: (wav, index) => {
          for (const engine of chunkEngines) {
            void sendLiveChunk(session.sessionId, engine.asrId, index, wav)
              .then((result) => {
                setPanels((previous) => {
                  const panel = previous[engine.asrId] ?? EMPTY_PANEL;
                  return {
                    ...previous,
                    [engine.asrId]: {
                      ...panel,
                      text: result.text ? (panel.text ? `${panel.text} ${result.text}` : result.text) : panel.text,
                      latencies: [...panel.latencies, result.latencyMs],
                      chunkCount: panel.chunkCount + 1,
                    },
                  };
                });
              })
              .catch((error: Error) => {
                setPanels((previous) => ({
                  ...previous,
                  [engine.asrId]: { ...(previous[engine.asrId] ?? EMPTY_PANEL), error: error.message },
                }));
              });
          }
        },
      });
      recorderRef.current = recorder;
      setRecording(true);
      // Elapsed comes from the sample count, so the clock can never drift away
      // from the audio actually captured.
      tickRef.current = window.setInterval(() => {
        setElapsed(recorder.elapsedSec());
        // One repaint per tick regardless of the audio block size.
        setPeaks([...peaksRef.current]);
      }, 200);
    } catch (error) {
      // Tear down everything the failed attempt opened — sockets, the capture,
      // the session id — so a retry starts clean instead of inheriting a dead
      // socket or a microphone still running.
      cleanupRun();
      const started = recorderRef.current;
      recorderRef.current = null;
      if (started) await started.stop().catch(() => undefined);
      sessionRef.current = null;
      setRecording(false);
      setRunError((error as Error).message);
    }
  };

  const handleStop = async () => {
    const recorder = recorderRef.current;
    const sessionId = sessionRef.current;
    if (!recorder || !sessionId) return;
    setRecording(false);
    setFinalizing(true);
    try {
      const wav = await recorder.stop();
      recorderRef.current = null;
      cleanupRun();
      const runs = await finalizeLiveSession(
        sessionId,
        wav,
        referenceText,
        script || savedRef?.source === "script" ? "script" : "pasted",
        script?.params ?? null,
        pendingAudioFileId,
      );
      setResults(runs);
      const recordedId = runs[0]?.audioFileId;
      if (recordedId != null) onRecorded?.(recordedId);
    } catch (error) {
      setRunError((error as Error).message);
    } finally {
      setFinalizing(false);
      sessionRef.current = null;
    }
  };

  const unconfigured = (config?.engines ?? []).filter((engine) => !engine.configured);

  // Switching sub-mode while a live capture is running must tear it down the
  // same way unmounting does — the microphone and sockets are real resources
  // and a switch is not a reason to leave them running unseen.
  const handleSubModeChange = (next: "stt" | "tts") => {
    if (next === transcriptSubMode) return;
    if (recording) {
      cleanupRun();
      void recorderRef.current?.stop();
      recorderRef.current = null;
      sessionRef.current = null;
      setRecording(false);
    }
    onTranscriptSubMode(next);
  };

  return (
    <main className="page transcript-page">
      <div className="transcript-head">
        <div>
          <span className="eyebrow">Evaluation</span>
          <h1>Transcript</h1>
          <p>
            {transcriptSubMode === "tts"
              ? "Generate a script, synthesize it with both engines, and compare the audio they return. Every figure here is measured."
              : "Read a script aloud; each engine is scored on its own native transport."}
          </p>
        </div>
        <div className="transcript-mode-toggle">
          <Segmented
            value={transcriptSubMode}
            options={[
              { value: "stt", label: "STT" },
              { value: "tts", label: "TTS" },
            ]}
            onChange={handleSubModeChange}
            label="Transcript sub-mode"
          />
          {/* Counted from what /config actually returned, so an unconfigured host
              says "1 engine" rather than claiming a comparison it cannot run. */}
          <small className="mono muted transcript-mode-caption">
            {transcriptSubMode === "tts"
              ? `speech synthesis · ${ttsEngineCount} ${ttsEngineCount === 1 ? "engine" : "engines"}`
              : `speech recognition · ${engines.length} ${engines.length === 1 ? "engine" : "engines"}`}
          </small>
        </div>
      </div>

      {transcriptSubMode === "tts" ? (
        <TtsStudio
          runtimeConfig={runtimeConfig}
          audioFileId={pendingAudioFileId ?? audioFileId ?? null}
          onScriptSaved={onScriptSaved}
        />
      ) : (
      <>
      {unconfigured.length > 0 && (
        <p className="transcript-notice">
          Not configured: {unconfigured.map((engine) => engine.name).join(", ")} (check .env)
        </p>
      )}

      <section className="transcript-grid">
        <div className="panel script-controls">
          <h2>Read it out</h2>

          {lengths.length > 0 && (
            <SliderControl
              label="Script length"
              value={effectiveMinutes}
              min={lengths[0]}
              max={lengths[lengths.length - 1]}
              step={1}
              format={(value) => `${value} min`}
              onChange={setMinutes}
              disabled={recording}
            />
          )}

          {config?.scriptLanguageMixes?.length ? (
            <div className="control-block">
              <span className="control-label">Language mix</span>
              <Segmented
                value={effectiveMix}
                options={config.scriptLanguageMixes.map((id) => ({ value: id, label: MIX_LABELS[id] ?? id }))}
                onChange={setMix}
                label="Language mix"
                disabled={recording}
              />
              <small className="muted">Script only; engines auto-detect.</small>
            </div>
          ) : null}

          {config?.scriptHardCases?.length ? (
            <div className="control-block">
              <span className="control-label">Hard cases</span>
              <div className="chip-row">
                {config.scriptHardCases.map((id) => (
                  <button
                    key={id}
                    type="button"
                    className={`case-chip${effectiveHardCases.includes(id) ? " is-on" : ""}`}
                    aria-pressed={effectiveHardCases.includes(id)}
                    onClick={() => toggleHardCase(id)}
                    disabled={recording}
                  >
                    {effectiveHardCases.includes(id) ? "✓" : "+"} {HARD_CASE_LABELS[id] ?? id}
                  </button>
                ))}
              </div>
            </div>
          ) : null}

          {config?.scriptModel ? (
            <button
              type="button"
              className="generate-btn"
              onClick={handleGenerate}
              disabled={generating || recording}
            >
              {generating ? "Generating…" : "Generate new script"}
            </button>
          ) : (
            <small className="muted">
              No script gateway configured — paste a reference below instead.
            </small>
          )}
          {scriptError && <p className="transcript-error">{scriptError}</p>}
        </div>

        {/* One editable box whether the script was generated, restored or typed
            here -- the same control the TTS side uses, so the identical script
            wraps identically on both. It was a read-only justified <p> when a
            script existed, which both blocked editing and broke lines
            differently from the textarea. Read-only only once the recording
            exists: the scores on screen were measured against this exact text,
            and editing it there would leave them describing something else. */}
        <div className="panel script-body">
          <div className="script-meta">
            <span className="mono">
              {shownScript
                ? `${shownScript.label} · ${shownScript.minutes} min · ${shownScript.wordCount} words`
                : `REFERENCE · ${referenceWords} words`}
              {/* The MEASURED split, not the requested one. A language mix is an
                  instruction the model follows loosely — asking for even halves
                  lands anywhere from 36% to 57% Arabic — so the card reports what
                  came back rather than repeating the label on the button. */}
              {shownScript?.split && !dirty
                ? ` · ${Math.round(shownScript.split.arabic * 100)}% AR / ${Math.round(shownScript.split.english * 100)}% EN measured`
                : null}
            </span>
            {/* The model that actually generated it, from the response — not
                a hardcoded vendor name. */}
            {shownScript?.generatorModel && !dirty && (
              <span className="script-model">{shownScript.generatorModel}</span>
            )}
            {dirty && !viewingSaved && (
              <span className="script-dirty mono">unsaved — recording will save it</span>
            )}
          </div>
          <textarea
            className="script-paste"
            dir="auto"
            placeholder="Generate a script, or paste the text you will read."
            value={draft}
            onChange={(event) => setDraftEdit(event.target.value)}
            disabled={recording}
            readOnly={viewingSaved}
          />
        </div>
      </section>

      <section className="panel recorder-bar">
        {viewingSaved ? (
          <>
            {/* A saved recording has nothing left to capture, so the control plays
                back what was actually spoken instead of offering to record over it.
                The audio is the same object the engines transcribed. */}
            <audio
              ref={audioRef}
              src={audioFileId != null ? audioStreamUrl(audioFileId) : undefined}
              preload="metadata"
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
              onEnded={() => setPlaying(false)}
              onTimeUpdate={(event) => setElapsed(event.currentTarget.currentTime)}
              onLoadedMetadata={(event) => setSavedDuration(event.currentTarget.duration)}
            />
            <button
              type="button"
              className="record-btn is-playback"
              onClick={() => {
                const audio = audioRef.current;
                if (!audio) return;
                if (audio.paused) void audio.play().catch(() => undefined);
                else audio.pause();
              }}
              title={playing ? "Pause" : "Play what was recorded"}
            >
              <span className={`play-glyph${playing ? " is-playing" : ""}`} aria-hidden="true" />
            </button>
            <div className="record-state">
              <strong>{playing ? "Playing" : "Recorded"}</strong>
              <span className="mono record-clock">
                {Math.floor(elapsed / 60)}:{String(Math.floor(elapsed % 60)).padStart(2, "0")}
                {savedDuration != null && Number.isFinite(savedDuration) && (
                  <span className="muted">
                    {" / "}{Math.floor(savedDuration / 60)}:{String(Math.floor(savedDuration % 60)).padStart(2, "0")}
                  </span>
                )}
              </span>
            </div>
            <div className="record-wave is-static" aria-hidden="true">
              {/* Deliberately not a waveform: the peaks were a live artefact of the
                  capture and were never stored, and drawing a plausible-looking one
                  from nothing would be inventing a picture of the audio. */}
              <span className="muted mono">{fileName ?? "recording"}</span>
            </div>
          </>
        ) : (
        <>
        <button
          type="button"
          className={`record-btn${recording ? " is-recording" : ""}`}
          onClick={recording ? handleStop : handleStart}
          disabled={!engines.length || finalizing || (!recording && !referenceText)}
          title={
            !engines.length
              ? "No transcription engine is configured on this host"
              : !referenceText && !recording
                ? "Generate or paste a reference first — without one there is nothing to score against"
                : recording ? "Stop and score" : "Start recording"
          }
        >
          <span className="record-glyph" aria-hidden="true" />
        </button>
        <div className="record-state">
          <strong className={recording ? "is-live" : ""}>
            {recording ? "Recording" : finalizing ? "Scoring…" : "Ready"}
          </strong>
          <span className="mono record-clock">
            {Math.floor(elapsed / 60)}:{String(Math.floor(elapsed % 60)).padStart(2, "0")}
          </span>
        </div>
        <div className="record-wave" aria-hidden="true">
          {peaks.map((peak, index) => (
            <i key={index} style={{ height: `${Math.max(4, peak * 100)}%` }} />
          ))}
        </div>
        </>
        )}
      </section>
      {runError && <p className="transcript-error">{runError}</p>}

      <section className="engine-panels">
        {engines.map((engine) => {
          // Live panel while capturing; the stored run when a saved recording is
          // open. Its avgLatencyMs is the value that was MEASURED at the time —
          // recomputing it from anything now would be a different number.
          const stored = runByEngine.get(engine.asrId);
          const live = panels[engine.asrId] ?? EMPTY_PANEL;
          const panel: LivePanel = viewingSaved && stored
            ? {
                text: stored.text ?? "",
                latencies: [],
                chunkCount: stored.chunkCount ?? 0,
                error: stored.error ?? null,
              }
            : live;
          const avg = viewingSaved && stored
            ? stored.avgLatencyMs ?? null
            : mean(live.latencies);
          return (
            <div className="panel engine-panel" key={engine.asrId}>
              <div className="engine-head">
                <span className="engine-name">
                  <span className={`engine-dot ${engine.transport}`} aria-hidden="true" />
                  {engine.name}
                </span>
                <span className="mono engine-stats">
                  {avg != null && <>lag {fmtMs(avg)}</>}
                  {" "}
                  {panel.text.split(/\s+/).filter(Boolean).length} words
                </span>
              </div>
              <div className="engine-body" dir="auto">
                {panel.error ? (
                  <p className="transcript-error">{panel.error}</p>
                ) : panel.text ? (
                  panel.text
                ) : (
                  <span className="muted">
                    {recording
                      ? "Listening…"
                      : loadingSaved
                        ? "Loading…"
                        : viewingSaved
                          ? "This engine returned nothing for this recording."
                          : "No transcript yet."}
                  </span>
                )}
              </div>
              <div className="engine-foot">
                {/* Labelled per transport, deliberately. A stream's lag and a
                    chunked engine's round-trip are different measurements, and
                    their chunk statistics are not comparable to each other. */}
                <div>
                  <span className="eyebrow">Transport</span>
                  <b className="mono">
                    {engine.transport === "stream"
                      ? "stream · VAD"
                      : `chunks · ${(
                          (viewingSaved && stored?.chunkIntervalSec) || chunkInterval
                        ).toFixed(1)}s`}
                  </b>
                </div>
                <div>
                  <span className="eyebrow">{engine.transport === "stream" ? "Segments" : "Chunks"}</span>
                  <b className="mono">{panel.chunkCount}</b>
                </div>
                <div>
                  <span className="eyebrow">{engine.transport === "stream" ? "Avg lag behind live" : "Avg chunk latency"}</span>
                  <b className="mono">{avg == null ? "—" : fmtMs(avg)}</b>
                </div>
              </div>
            </div>
          );
        })}
      </section>

      {shownRuns && shownRuns.length > 0 && (
        <TranscriptScorecard runs={shownRuns} engines={engines} referenceWords={referenceWords} />
      )}

      {/* Only for a recording that has NO reference yet — a diarization upload being
          scored for the first time. Keyed on the reference, not on `viewingSaved`: a
          saved script HAS a reference and no audio, so offering to paste one over the
          script you are about to read asks for something it already has. */}
      {audioFileId != null && savedRef == null && !loadingSaved && (
        <StoredRecordingScorer
          audioFileId={audioFileId}
          fileName={fileName ?? "this recording"}
          engines={engines}
          pollIntervalMs={runtimeConfig?.pollIntervalMs ?? 1500}
        />
      )}
      </>
      )}
    </main>
  );
}
