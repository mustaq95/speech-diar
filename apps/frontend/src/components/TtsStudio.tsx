import { useEffect, useMemo, useRef, useState } from "react";
import { Segmented, SliderControl } from "./controls";
import type { GeneratedScript, TranscriptReference, TtsRun } from "../types/diarization";
import type { RuntimeConfig, TtsEngineInfo } from "../adapters";
import {
  createReference,
  fetchReference,
  fetchTtsRuns,
  generateScript,
  putReference,
  synthesizeTts,
  ttsAudioUrl,
} from "../adapters";
import { decodeWaveformPeaks } from "../playback";
import { VoiceSelect } from "./VoiceSelect";

/** Same labels as TranscriptStudio's script controls — copied, not shared, per
 * this repo's "copy small, single-use blocks rather than extract" rule. */
const MIX_LABELS: Record<string, string> = {
  ar: "Arabic only",
  "mixed-50-50": "Mixed 50/50",
  en: "English",
};
const HARD_CASE_LABELS: Record<string, string> = {
  "proper-nouns": "Proper nouns",
  "numbers-dates": "Numbers & dates",
  "emirati-dialect": "Emirati dialect",
  "gulf-dialect": "Emirati dialect",
  "fast-speech": "Fast speech",
};

interface TtsStudioProps {
  /** Null until GET /config lands. */
  runtimeConfig: RuntimeConfig | null;
  /** The recording this comparison is scoped to, when one already exists —
   * either a reopened project or the row this page's STT side just created. */
  audioFileId?: number | null;
  /** Called when a generated script is saved, so Projects picks up the new entry
   * without navigating away — same contract as TranscriptStudio's prop. */
  onScriptSaved?: (audioFileId: number) => void;
}

/** Identity of one stored clip. Everything describing a clip (its run, peaks,
 * playhead, cache version, audio element) keys on this rather than on ttsId,
 * because an engine now has one clip per voice. `selectedVoice` deliberately
 * does NOT: it is a per-engine selection, not a property of a clip. */
function clipKey(ttsId: string, voice: string): string {
  return `${ttsId}::${voice}`;
}

function fmtMs(ms: number): string {
  return ms >= 10_000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}

/** m:ss, the shape a player's clock is read in. */
function fmtClock(seconds: number): string {
  const whole = Math.round(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

function fmtBytes(bytes: number): string {
  return bytes >= 1_048_576 ? `${(bytes / 1_048_576).toFixed(1)} MB` : `${Math.round(bytes / 1024)} KB`;
}

/**
 * The format line under a clip: only the parts the container actually stated.
 * MP3 carries no fixed sample width, so `bitDepth` is genuinely absent there
 * rather than 16 — each piece is dropped independently instead of printing a
 * default that would read as measured.
 */
function formatSummary(run: TtsRun, assumedRate: boolean): string {
  const parts: string[] = [];
  if (run.audioFormat) parts.push(run.audioFormat.toUpperCase());
  if (run.nativeSampleRate != null) {
    const khz = (run.nativeSampleRate / 1000).toFixed(run.nativeSampleRate % 1000 === 0 ? 0 : 1);
    parts.push(`${khz} kHz${assumedRate ? " (assumed)" : ""}`);
  }
  if (run.bitDepth != null) parts.push(`${run.bitDepth}-bit`);
  if (run.channels != null) parts.push(run.channels === 1 ? "mono" : run.channels === 2 ? "stereo" : `${run.channels}ch`);
  if (run.sizeBytes != null) parts.push(fmtBytes(run.sizeBytes));
  return parts.join(" · ");
}

/**
 * The TTS comparison surface: synthesize the SAME reference text a script
 * generates with each configured TTS engine and compare the clips. There is no
 * ground truth here to score against (no WER-equivalent for speech) — this is a
 * descriptive comparison of what each engine actually produced and how it was
 * measured, the same honesty rule the diarization report follows.
 */
export function TtsStudio({ runtimeConfig, audioFileId, onScriptSaved }: TtsStudioProps) {
  const scriptConfig = runtimeConfig?.transcript ?? null;
  const ttsConfig = runtimeConfig?.tts ?? null;
  const engines = useMemo<TtsEngineInfo[]>(
    () => (ttsConfig?.engines ?? []).filter((engine) => engine.configured),
    [ttsConfig],
  );
  const unconfigured = (ttsConfig?.engines ?? []).filter((engine) => !engine.configured);

  // --- a reopened recording's already-generated script and clips -----------
  const [savedRef, setSavedRef] = useState<TranscriptReference | null>(null);
  const [runs, setRuns] = useState<Record<string, TtsRun>>({});
  const [loadingSaved, setLoadingSaved] = useState(false);

  useEffect(() => {
    // A different recording is a different set of clips: everything keyed by
    // ttsId has to go, or the previous project's waveform survives the switch.
    decodedRef.current.clear();
    setDraftEdit(null);
    setCreatedId(null);
    setPeaksByEngine({});
    setProgress({});
    setClipVersion({});
    if (audioFileId == null) {
      setSavedRef(null);
      setRuns({});
      return;
    }
    let cancelled = false;
    setLoadingSaved(true);
    void (async () => {
      try {
        const [reference, ttsRuns] = await Promise.all([
          fetchReference(audioFileId),
          fetchTtsRuns(audioFileId),
        ]);
        if (cancelled) return;
        setSavedRef(reference);
        setRuns(Object.fromEntries(ttsRuns.map((run) => [clipKey(run.ttsId, run.voice ?? ""), run])));
      } catch (error) {
        if (!cancelled) console.error("Could not load the saved TTS comparison:", error);
      } finally {
        if (!cancelled) setLoadingSaved(false);
      }
    })();
    return () => { cancelled = true; };
  }, [audioFileId]);

  // --- script controls, copied from TranscriptStudio ------------------------
  const lengths = scriptConfig?.scriptLengthsMin ?? [];
  const [minutes, setMinutes] = useState<number | null>(null);
  const [mix, setMix] = useState<string | null>(null);
  const [hardCases, setHardCases] = useState<string[] | null>(null);
  const savedParams = (savedRef?.params ?? null) as
    | { minutes?: number; languageMix?: string; hardCases?: string[]; generatorModel?: string;
        languageSplit?: { arabic: number; english: number } }
    | null;
  const effectiveMinutes = minutes ?? savedParams?.minutes ?? lengths[1] ?? lengths[0] ?? 1;
  const effectiveMix =
    mix ?? savedParams?.languageMix ?? scriptConfig?.scriptLanguageMixes?.[1]
    ?? scriptConfig?.scriptLanguageMixes?.[0] ?? "";
  const effectiveHardCases =
    hardCases ?? savedParams?.hardCases ?? scriptConfig?.scriptHardCases?.slice(0, 4) ?? [];

  const [script, setScript] = useState<GeneratedScript | null>(null);
  // The row `createReference` just made for a typed script. Without this the
  // component has no id for it, and the clip URL, the download link and the
  // waveform decode all silently skip themselves.
  const [createdId, setCreatedId] = useState<number | null>(null);
  const [generating, setGenerating] = useState(false);
  const [scriptError, setScriptError] = useState<string | null>(null);

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
      setCreatedId(null);
      // Drop the explicit picks so the controls fall through to the saved
      // params: what the script was actually generated with, which is also what
      // the other sub-mode reads. Without this each side keeps its own picks and
      // the two disagree about settings for one shared script.
      setMinutes(null);
      setMix(null);
      setHardCases(null);

      setRuns({});
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

  // A fresh generation outranks the reopened row's own reference — same
  // precedence TranscriptStudio uses for `referenceText`/`shownScript`.
  const effectiveAudioFileId = script?.audioFileId ?? createdId ?? audioFileId ?? null;
  const referenceText = script?.text ?? savedRef?.text ?? "";

  // What the textarea holds. `null` means "follow whatever is stored"; any
  // string means the operator has typed and owns the value until it is saved.
  const [draftEdit, setDraftEdit] = useState<string | null>(null);
  const draft = draftEdit ?? referenceText;
  const setDraft = (next: string) => setDraftEdit(next);
  const draftWords = draft.trim() ? draft.trim().split(/\s+/).length : 0;
  // Compared against the STORED text, so regenerating or reopening clears the
  // flag on its own without anyone having to reset it.
  const dirty = draft.trim() !== referenceText.trim();
  const hasClips = Object.keys(runs).length > 0;

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

  const overLimit = ttsConfig?.maxInputChars != null && draft.trim().length > ttsConfig.maxInputChars;

  // --- synthesis --------------------------------------------------------
  const [synthesizing, setSynthesizing] = useState<Record<string, boolean>>({});
  // Locked while either engine runs: the draft is what gets saved, and editing
  // it mid-flight would change the text under a synthesis already in progress.
  const anyBusy = Object.values(synthesizing).some(Boolean);
  const [selectedVoice, setSelectedVoice] = useState<Record<string, string>>({});
  // Keyed by "<ttsId>:<version>", not ttsId alone: a re-synthesis replaces the
  // bytes behind an unchanged URL, and a ttsId-only guard skipped the re-decode,
  // leaving the PREVIOUS take's waveform beside the new take's numbers (and
  // seek clicks landing on the wrong audio).
  const decodedRef = useRef<Set<string>>(new Set());
  // Bumped per engine on every successful synthesis. Drives both the decode
  // guard and the cache-busting query param on the clip URL.
  const [clipVersion, setClipVersion] = useState<Record<string, number>>({});
  const [peaksByEngine, setPeaksByEngine] = useState<Record<string, number[]>>({});
  const audioRefs = useRef<Map<string, HTMLAudioElement>>(new Map());
  const [playingId, setPlayingId] = useState<string | null>(null);
  // Fraction of each clip already played, 0..1. Real playback position, which
  // is what tints the waveform -- not an animation.
  const [progress, setProgress] = useState<Record<string, number>>({});

  /** The recording id to synthesize against, saving the draft first if needed.
   *
   * Synthesis reads its text from a STORED reference, so a script that only
   * exists in the textarea has to be persisted before either engine can see
   * it. Doing it here rather than behind a separate Save button means the
   * obvious action does the obvious thing: type, press Synthesize, hear it.
   */
  const ensureSavedScript = async (): Promise<number | null> => {
    const text = draft.trim();
    if (!text) return null;
    if (effectiveAudioFileId == null) {
      const created = await createReference(text);
      setSavedRef(created);
      setCreatedId(created.audioFileId);
      setDraftEdit(null);
      onScriptSaved?.(created.audioFileId);
      return created.audioFileId;
    }
    if (dirty) {
      // Replaces the text AND drops every clip made from the old words -- the
      // warning above the textarea says so before this runs.
      const updated = await putReference(effectiveAudioFileId, text, "pasted");
      setSavedRef(updated);
      setScript(null);
      setDraftEdit(null);
      setRuns({});
      setPeaksByEngine({});
      setProgress({});
      decodedRef.current.clear();
    }
    return effectiveAudioFileId;
  };

  const handleSynthesize = async (engine: TtsEngineInfo, voice: string) => {
    setSynthesizing((previous) => ({ ...previous, [engine.ttsId]: true }));
    try {
      const targetId = await ensureSavedScript();
      if (targetId == null) return;
      const run = await synthesizeTts(targetId, engine.ttsId, voice);
      const key = clipKey(engine.ttsId, run.voice ?? voice);
      setRuns((previous) => ({ ...previous, [key]: run }));
      // New bytes for THIS voice only: drop its drawing and playhead, and leave
      // every other voice's clip untouched.
      setClipVersion((previous) => ({ ...previous, [key]: (previous[key] ?? 0) + 1 }));
      setPeaksByEngine((previous) => { const next = { ...previous }; delete next[key]; return next; });
      setProgress((previous) => ({ ...previous, [key]: 0 }));
    } catch (error) {
      // The engine's OWN failure already comes back as a normal "failed" TtsRun
      // (see the route). A thrown error here means the request itself never
      // reached that far (network, 404, 422) — render it the same way so the
      // card behaves identically either way, never silently.
      setRuns((previous) => ({
        ...previous,
        [clipKey(engine.ttsId, voice)]: {
          audioFileId: effectiveAudioFileId ?? 0,
          ttsId: engine.ttsId,
          ttsName: engine.name,
          status: "failed",
          error: (error as Error).message,
          voice,
          delivery: engine.delivery,
        },
      }));
    } finally {
      setSynthesizing((previous) => ({ ...previous, [engine.ttsId]: false }));
    }
  };

  // Real peaks decoded from the clip's own PCM bytes, same function the
  // diarization waveform uses — never a synthetic shape. Only for a wav clip:
  // both engines render wav today, but if one ever answers mp3 instead, no bars
  // are drawn rather than a wrong or flat picture.
  useEffect(() => {
    if (effectiveAudioFileId == null) return;
    let cancelled = false;
    for (const run of Object.values(runs)) {
      if (run.status !== "done" || run.audioFormat !== "wav" || !run.voice) continue;
      const key = clipKey(run.ttsId, run.voice);
      const version = clipVersion[key] ?? 0;
      const guard = `${key}:${version}`;
      if (decodedRef.current.has(guard)) continue;
      decodedRef.current.add(guard);
      void decodeWaveformPeaks(
        ttsAudioUrl(effectiveAudioFileId, run.ttsId, false, version, run.voice), 120,
      )
        .then((peaks) => { if (!cancelled) setPeaksByEngine((previous) => ({ ...previous, [key]: peaks })); })
        .catch((error: Error) => console.error(`Could not decode waveform for ${key}:`, error));
    }
    return () => { cancelled = true; };
  }, [runs, effectiveAudioFileId, clipVersion]);

  const togglePlay = (key: string) => {
    const audio = audioRefs.current.get(key);
    if (!audio) return;
    if (audio.paused) void audio.play().catch(() => undefined);
    else audio.pause();
  };

  return (
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
            />
          )}

          {scriptConfig?.scriptLanguageMixes?.length ? (
            <div className="control-block">
              <span className="control-label">Language mix</span>
              <Segmented
                value={effectiveMix}
                options={scriptConfig.scriptLanguageMixes.map((id) => ({ value: id, label: MIX_LABELS[id] ?? id }))}
                onChange={setMix}
                label="Language mix"
              />
              <small className="muted">Script only; engines auto-detect.</small>
            </div>
          ) : null}

          {scriptConfig?.scriptHardCases?.length ? (
            <div className="control-block">
              <span className="control-label">Hard cases</span>
              <div className="chip-row">
                {scriptConfig.scriptHardCases.map((id) => (
                  <button
                    key={id}
                    type="button"
                    className={`case-chip${effectiveHardCases.includes(id) ? " is-on" : ""}`}
                    aria-pressed={effectiveHardCases.includes(id)}
                    onClick={() => toggleHardCase(id)}
                  >
                    {effectiveHardCases.includes(id) ? "✓" : "+"} {HARD_CASE_LABELS[id] ?? id}
                  </button>
                ))}
              </div>
            </div>
          ) : null}

          {scriptConfig?.scriptModel ? (
            <button
              type="button"
              className="generate-btn"
              onClick={handleGenerate}
              disabled={generating}
            >
              {generating ? "Generating…" : "Generate new script"}
            </button>
          ) : (
            <small className="muted">No script gateway configured on this host.</small>
          )}
          {scriptError && <p className="transcript-error">{scriptError}</p>}
        </div>

        {/* Always editable, unlike the STT side which only offers a textarea when
            there is no generated script. Both engines synthesize whatever is in
            here, so being able to bring your own text -- or fix a generated
            one -- is the whole input to this comparison. */}
        <div className="panel script-body">
          <div className="script-meta">
            <span className="mono">
              {shownScript
                ? `${shownScript.label}${shownScript.minutes ? ` · ${shownScript.minutes} min` : ""}`
                : "YOUR SCRIPT"}
              {" · "}{draftWords} words · {draft.length} chars
              {shownScript?.split && !dirty
                ? ` · ${Math.round(shownScript.split.arabic * 100)}% AR / ${Math.round(shownScript.split.english * 100)}% EN measured`
                : null}
            </span>
            {shownScript?.generatorModel && !dirty && (
              <span className="script-model">{shownScript.generatorModel}</span>
            )}
            {dirty && <span className="script-dirty mono">unsaved — synthesizing will save it</span>}
          </div>
          <textarea
            className="script-paste"
            dir="auto"
            placeholder={loadingSaved ? "Loading…" : "Generate a script, or paste your own to synthesize."}
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            disabled={loadingSaved || anyBusy}
          />
          {overLimit && (
            <p className="transcript-error">
              {draft.length} chars, over this host's {ttsConfig?.maxInputChars}-char TTS limit —
              synthesis will be rejected until the script is shorter.
            </p>
          )}
          {dirty && hasClips && (
            // put_reference deletes clips made from the old words; say so before
            // they vanish rather than after.
            <p className="transcript-error">
              Synthesizing will save this edit and discard the clips made from the previous text.
            </p>
          )}
        </div>
      </section>

      <section className="engine-panels">
        {engines.map((engine) => {
          const busy = synthesizing[engine.ttsId] ?? false;
          // Which voices of this engine already have a stored clip. Drives both
          // the initial selection and the dots in the picker, so neither can
          // drift from what is actually stored.
          const synthesized = Object.values(runs)
            .filter((candidate) => candidate.ttsId === engine.ttsId && candidate.voice)
            .map((candidate) => candidate.voice as string);
          // No explicit pick yet: open on the newest stored clip's voice (the
          // route orders newest-first within an engine), else the engine
          // default. Defaulting straight to defaultVoice labelled a Sandra clip
          // as "Ruba" before per-voice storage existed.
          const voice = selectedVoice[engine.ttsId] ?? synthesized[0] ?? engine.defaultVoice;
          const key = clipKey(engine.ttsId, voice);
          // Absent simply means this voice has not been synthesized. No
          // mismatch case survives: a clip can only ever be its own voice's.
          const shownRun = runs[key];
          const peaks = peaksByEngine[key];
          const isPlaying = playingId === key;
          const played = progress[key] ?? 0;

          return (
            <div
              className="panel engine-panel is-tts"
              key={engine.ttsId}
              // One assignment drives every coloured element in the card: dot,
              // play button, waveform, download fill, avatar, first tile. Keyed
              // on delivery, extending the existing .engine-dot convention
              // (stream is already green, chunks already blue).
              style={{ "--engine": engine.delivery === "stream" ? "var(--good)" : "var(--accent)" } as React.CSSProperties}
            >
              <div className="engine-head">
                <span className="engine-name">
                  <span className={`engine-dot ${engine.delivery}`} aria-hidden="true" />
                  {engine.name}
                </span>
                <span className="mono engine-stats">
                  {engine.delivery === "stream" ? "stream · chunked PCM" : "single · full render"}
                </span>
              </div>
              {/* Driven entirely by .env: neither gateway exposes a list-voices
                  endpoint, so this is exactly what the host was configured with.
                  The subtitle is ENGINE-level config, true of every voice here. */}
              <div className="tts-voice-row">
                <span className="eyebrow">Voice</span>
                <VoiceSelect
                  voices={engine.voices}
                  value={voice}
                  onChange={(next) =>
                    setSelectedVoice((previous) => ({ ...previous, [engine.ttsId]: next }))
                  }
                  params={engine.synthesisParams}
                  synthesized={synthesized}
                  disabled={busy}
                  label={`${engine.name} voice`}
                />
                <span className="mono muted tts-voice-count">
                  {engine.voices.length} {engine.voices.length === 1 ? "voice" : "voices"}
                </span>
              </div>
              {/* Not `.engine-body`: that class reserves 150px of scrollable height
                  for an STT transcript, and a TTS card has no text to put there —
                  it would render as a tall empty well under the player. */}
              <div className="tts-body">
                {shownRun?.status === "failed" ? (
                  <p className="transcript-error">{shownRun.error ?? "Synthesis failed."}</p>
                ) : shownRun?.status === "done" ? (
                  <>
                    <audio
                      ref={(node) => {
                        if (node) audioRefs.current.set(key, node);
                        else audioRefs.current.delete(key);
                      }}
                      src={effectiveAudioFileId != null ? ttsAudioUrl(effectiveAudioFileId, engine.ttsId, false, clipVersion[key] ?? 0, voice) : undefined}
                      preload="none"
                      onPlay={() => setPlayingId(key)}
                      onPause={() => setPlayingId((id) => (id === key ? null : id))}
                      onEnded={() => {
                        setPlayingId((id) => (id === key ? null : id));
                        setProgress((previous) => ({ ...previous, [key]: 0 }));
                      }}
                      onTimeUpdate={(event) => {
                        const el = event.currentTarget;
                        if (!el.duration || !Number.isFinite(el.duration)) return;
                        setProgress((previous) => ({ ...previous, [key]: el.currentTime / el.duration }));
                      }}
                    />
                    <div className="tts-clip-row">
                      <button
                        type="button"
                        className="record-btn is-playback tts-play"
                        onClick={() => togglePlay(key)}
                        title={isPlaying ? "Pause" : "Play"}
                        aria-label={isPlaying ? `Pause ${engine.name}` : `Play ${engine.name}`}
                      >
                        <span className={`play-glyph${isPlaying ? " is-playing" : ""}`} aria-hidden="true" />
                      </button>
                      {peaks ? (
                        // Bars up to the playhead carry the engine colour, the rest
                        // stay dim. Both the shape and the split are real: peaks are
                        // decoded from the stored PCM, the split is currentTime.
                        <div
                          className="tts-wave"
                          onClick={(event) => {
                            const audio = audioRefs.current.get(key);
                            if (!audio?.duration || !Number.isFinite(audio.duration)) return;
                            const box = event.currentTarget.getBoundingClientRect();
                            const ratio = Math.min(1, Math.max(0, (event.clientX - box.left) / box.width));
                            audio.currentTime = ratio * audio.duration;
                            setProgress((previous) => ({ ...previous, [key]: ratio }));
                          }}
                          role="presentation"
                        >
                          {peaks.map((peak, index) => (
                            <i
                              key={index}
                              className={index < played * peaks.length ? "is-played" : undefined}
                              style={{ height: `${Math.max(6, peak * 100)}%` }}
                            />
                          ))}
                        </div>
                      ) : (
                        // Non-wav clip, or not decoded yet: never draw a wrong or
                        // flat picture of audio that was not actually decoded.
                        <span className="muted mono tts-wave-absent">
                          {shownRun.audioFormat && shownRun.audioFormat !== "wav"
                            ? `no waveform for ${shownRun.audioFormat}`
                            : "decoding…"}
                        </span>
                      )}
                      {/* The clock is the MEASURED duration of the stored clip, not
                          the <audio> element's own readout: the same number the
                          RTF divides by, so the two can never disagree on screen. */}
                      <span className="mono tts-clip-clock">
                        {shownRun.audioSec != null ? fmtClock(shownRun.audioSec) : "—"}
                      </span>
                    </div>
                    <div className="tts-clip-meta">
                      <span className="mono muted">
                        {formatSummary(shownRun, engine.ttsId === "hamsa-tts") || "format not reported"}
                      </span>
                      <span className="tts-clip-actions">
                        <button
                          type="button"
                          className="ghost-btn"
                          onClick={() => void handleSynthesize(engine, voice)}
                          disabled={busy || overLimit || !draft.trim()}
                        >
                          {busy ? "Synthesizing…" : "Re-synthesize"}
                        </button>
                        {effectiveAudioFileId != null && (
                          <a
                            className="tts-download"
                            href={ttsAudioUrl(effectiveAudioFileId, engine.ttsId, true, clipVersion[key] ?? 0, voice)}
                            download
                          >
                            ↓ Download {(shownRun.audioFormat ?? "audio").toUpperCase()}
                          </a>
                        )}
                      </span>
                    </div>
                  </>
                ) : busy ? (
                  <span className="muted">Synthesizing…</span>
                ) : (
                  <span className="muted">Not synthesized yet.</span>
                )}
              </div>
              {/* Every tile shows a measured figure or the REASON there is none —
                  never a 0 or a blank that reads as zero. There is deliberately no
                  "chars skipped" tile: neither engine reports what it declined to
                  say, so that number could only be invented. */}
              <div className="tts-tiles">
                <div className="score-tile">
                  <span className="eyebrow">Time to first audio</span>
                  <b className="mono">{shownRun?.firstAudioMs != null ? fmtMs(shownRun.firstAudioMs) : "—"}</b>
                  {/* The delivery sits under the figure because the two engines are
                      not measuring the same thing: a streamed body's first chunk is a
                      real head start, a single buffered response's "first audio" is
                      its whole synthesis by construction. */}
                  <small className="muted">
                    {engine.delivery === "stream" ? "first chunk off the stream" : "single response · ≈ total"}
                  </small>
                </div>
                <div className="score-tile">
                  <span className="eyebrow">Synthesis time</span>
                  <b className="mono">{shownRun?.synthMs != null ? fmtMs(shownRun.synthMs) : "—"}</b>
                  <small className="muted">
                    {shownRun?.audioSec != null ? `for ${shownRun.audioSec.toFixed(1)} s of audio` : "wall clock"}
                  </small>
                </div>
                <div className="score-tile">
                  <span className="eyebrow">Real-time factor</span>
                  <b className="mono">{shownRun?.rtf != null ? shownRun.rtf.toFixed(2) : "—"}</b>
                  <small className="muted">
                    {shownRun?.rtf != null ? "synthesis ÷ audio" : "needs a measured duration"}
                  </small>
                </div>
                <div className="score-tile">
                  <span className="eyebrow">Input characters</span>
                  <b className="mono">{shownRun?.textChars != null ? shownRun.textChars : "—"}</b>
                  {/* Both engines are handed the identical string — that is the whole
                      point of the comparison, so it is stated rather than left for the
                      reader to infer from two equal numbers. */}
                  <small className="muted">same text to both</small>
                </div>
              </div>
              {/* Only the FIRST synthesis gets the full-width primary button. Once a
                  clip exists, re-running lives beside Download in the meta row, where
                  it reads as an action on that clip rather than the card's headline. */}
              {shownRun?.status !== "done" && (
                <button
                  type="button"
                  className="generate-btn"
                  onClick={() => void handleSynthesize(engine, voice)}
                  disabled={busy || overLimit || !draft.trim()}
                  title={
                    !draft.trim()
                      ? "Write or generate a script first — there is nothing to synthesize yet"
                      : overLimit
                        ? "Reference text is over this host's TTS char limit"
                        : shownRun
                          ? "Try again"
                          : "Synthesize"
                  }
                >
                  {busy ? "Synthesizing…" : shownRun?.status === "failed" ? "Try again" : "Synthesize"}
                </button>
              )}
            </div>
          );
        })}
      </section>

      {/* No naturalness/quality score anywhere: nothing here measures how the
          speech actually sounds, only what was measured about producing it. */}
      <p className="muted" style={{ fontSize: 12 }}>
        This comparison measures synthesis timing and format only — no naturalness or
        quality score, which nothing here computes.
      </p>
    </>
  );
}
