import { useEffect, useMemo, useRef, useState } from "react";
import { Segmented, SliderControl } from "./controls";
import type { ClonePreview, ClonedVoice, GeneratedScript } from "../types/diarization";
import type { RuntimeConfig } from "../adapters";
import {
  clonePreviewAudioUrl,
  deleteClonedVoice,
  extractVoiceTokens,
  fetchClonedVoices,
  generateScript,
  previewClonedVoice,
  registerClonedVoice,
  uploadCloneReference,
} from "../adapters";
import { decodeWaveformPeaks } from "../playback";

/** Same labels as the other two sub-modes' script controls — copied, not
 * shared, per this repo's "copy small, single-use blocks" rule. */
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

function fmtMs(ms: number): string {
  return ms >= 10_000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}

function fmtClock(seconds: number): string {
  const whole = Math.round(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

function fmtBytes(bytes: number): string {
  return bytes >= 1_048_576 ? `${(bytes / 1_048_576).toFixed(1)} MB` : `${Math.round(bytes / 1024)} KB`;
}

/** The format line under a clip: only the parts actually probed. Each piece is
 * dropped independently rather than defaulted, the same rule TtsStudio's own
 * `formatSummary` follows — a printed 16-bit that was never measured reads
 * exactly like one that was. */
function clipSummary(v: {
  audioFormat?: string;
  nativeSampleRate?: number;
  channels?: number;
  sizeBytes?: number;
}): string {
  const parts: string[] = [];
  if (v.audioFormat) parts.push(v.audioFormat.toUpperCase());
  if (v.nativeSampleRate != null) {
    const khz = (v.nativeSampleRate / 1000).toFixed(v.nativeSampleRate % 1000 === 0 ? 0 : 1);
    parts.push(`${khz} kHz`);
  }
  if (v.channels != null) parts.push(v.channels === 1 ? "mono" : v.channels === 2 ? "stereo" : `${v.channels}ch`);
  if (v.sizeBytes != null) parts.push(fmtBytes(v.sizeBytes));
  return parts.join(" · ");
}

/** First initial, for the registry rail's avatar. */
function initial(name: string): string {
  return (name.trim()[0] ?? "?").toUpperCase();
}

/** A deterministic bar chart of a token array's SIZE, not its contents.
 *
 * The pod's tokens are opaque integers this repo never interprets, so the bars
 * are a fixed decorative row scaled to the count — and the count is printed
 * beside them. Deliberately not a plot of the actual values: that would look
 * like a measurement of something and it is not one. */
function TokenBars({ count }: { count: number }) {
  const bars = useMemo(() => {
    // A fixed pseudo-random shape seeded by the count, so the same extraction
    // always draws the same row rather than reshuffling on every render.
    let seed = count || 1;
    return Array.from({ length: 40 }, () => {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      return 25 + ((seed >> 8) % 75);
    });
  }, [count]);
  return (
    <div className="token-bars" aria-hidden="true">
      {bars.map((height, index) => (
        <i key={index} style={{ height: `${height}%` }} />
      ))}
    </div>
  );
}

interface CloneStudioProps {
  runtimeConfig: RuntimeConfig | null;
  /** Called when a generated script is saved, so Projects picks up the new
   * entry — the same contract the other two sub-modes use. */
  onScriptSaved?: (audioFileId: number) => void;
}

/**
 * The Clone sub-mode: register a speaker on the TTS pod from one reference
 * clip, then synthesize with it.
 *
 * Four steps in the order the pod actually performs them, each labelled with
 * the call it makes, because the two cloning calls fail independently and an
 * operator who cannot see which one failed cannot act on it.
 *
 * Nothing here rates how close the clone is to its source. No similarity or
 * naturalness measurement exists in this repo, and the honest artifact is A/B
 * playback of the two clips — which is what step 4 renders.
 */
export function CloneStudio({ runtimeConfig, onScriptSaved }: CloneStudioProps) {
  const scriptConfig = runtimeConfig?.transcript ?? null;
  const cloneConfig = runtimeConfig?.voiceClone ?? null;
  const previewEngine = useMemo(
    () => (runtimeConfig?.tts?.engines ?? []).find((e) => e.ttsId === cloneConfig?.previewTtsId) ?? null,
    [runtimeConfig, cloneConfig],
  );

  // --- the registry ---------------------------------------------------------
  const [voices, setVoices] = useState<ClonedVoice[]>([]);
  const [loadingVoices, setLoadingVoices] = useState(false);

  const refreshVoices = async (): Promise<ClonedVoice[]> => {
    const rows = await fetchClonedVoices();
    setVoices(rows);
    return rows;
  };

  useEffect(() => {
    let cancelled = false;
    setLoadingVoices(true);
    void (async () => {
      try {
        const rows = await fetchClonedVoices();
        if (!cancelled) setVoices(rows);
      } catch (error) {
        if (!cancelled) console.error("Could not load the cloned-voice registry:", error);
      } finally {
        if (!cancelled) setLoadingVoices(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // --- script controls, so there is something to read into the clip ---------
  const lengths = scriptConfig?.scriptLengthsMin ?? [];
  const [minutes, setMinutes] = useState<number | null>(null);
  const [mix, setMix] = useState<string | null>(null);
  const [hardCases, setHardCases] = useState<string[] | null>(null);
  const [script, setScript] = useState<GeneratedScript | null>(null);
  const [generating, setGenerating] = useState(false);
  const [scriptError, setScriptError] = useState<string | null>(null);

  const effectiveMinutes = minutes ?? lengths[0] ?? 1;
  const effectiveMix = mix ?? scriptConfig?.scriptLanguageMixes?.[0] ?? "";
  const effectiveHardCases = hardCases ?? scriptConfig?.scriptHardCases?.slice(0, 3) ?? [];

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
      // The generated passage IS the prompt text: it is what the operator is
      // about to read into the reference clip, and the pod needs the transcript
      // to match the audio word for word.
      setPromptText(generated.text);
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

  // --- step 1: the reference clip -------------------------------------------
  const [audioUrl, setAudioUrl] = useState("");
  const [promptText, setPromptText] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploaded, setUploaded] = useState<{ audioSec?: number } | null>(null);
  const [extracting, setExtracting] = useState(false);
  const [stepError, setStepError] = useState<string | null>(null);

  /** The voice being built right now. Null before the first extraction; set to
   * whatever the routes return, including a "failed" row — the row IS the
   * record of the attempt and hiding it would lose the pod's message. */
  const [working, setWorking] = useState<ClonedVoice | null>(null);

  const handleUpload = async (file: File) => {
    setUploading(true);
    setStepError(null);
    try {
      const result = await uploadCloneReference(file);
      setAudioUrl(result.audioUrl);
      setUploaded({ audioSec: result.audioSec });
    } catch (error) {
      setStepError((error as Error).message);
    } finally {
      setUploading(false);
    }
  };

  const handleExtract = async () => {
    if (extracting) return;
    setExtracting(true);
    setStepError(null);
    try {
      const result = await extractVoiceTokens(audioUrl.trim(), promptText.trim(), dialect);
      setWorking(result);
      setSpeakerId((previous) => previous || suggestedName());
      await refreshVoices();
    } catch (error) {
      setStepError((error as Error).message);
    } finally {
      setExtracting(false);
    }
  };

  // --- step 3: registration --------------------------------------------------
  const [speakerId, setSpeakerId] = useState("");
  const [dialect, setDialect] = useState<string>(cloneConfig?.defaultDialect ?? "msa");
  const [registering, setRegistering] = useState(false);

  useEffect(() => {
    if (cloneConfig?.defaultDialect) setDialect((d) => d || cloneConfig.defaultDialect);
  }, [cloneConfig]);

  const suggestedName = () => `voice-${voices.length + 1}`;

  /** Whether this name already exists in OUR catalogue. The pod would overwrite
   * its own copy in memory silently; this is the only warning available, and it
   * is about our record, not the pod's. */
  const nameClash = voices.some(
    (v) => v.speakerId && v.speakerId.toLowerCase() === speakerId.trim().toLowerCase() && v.id !== working?.id,
  );

  const handleRegister = async () => {
    if (!working || registering) return;
    setRegistering(true);
    setStepError(null);
    try {
      const result = await registerClonedVoice(working.id, speakerId.trim(), dialect);
      setWorking(result);
      await refreshVoices();
    } catch (error) {
      setStepError((error as Error).message);
    } finally {
      setRegistering(false);
    }
  };

  // --- step 4: hearing it ----------------------------------------------------
  const [testText, setTestText] = useState("");
  const [preview, setPreview] = useState<ClonePreview | null>(null);
  const [previewVersion, setPreviewVersion] = useState(0);
  const [previewing, setPreviewing] = useState(false);
  const [peaks, setPeaks] = useState<number[] | null>(null);
  const [playing, setPlaying] = useState(false);
  const [played, setPlayed] = useState(0);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const handlePreview = async () => {
    if (!working || previewing) return;
    setPreviewing(true);
    setStepError(null);
    try {
      const result = await previewClonedVoice(working.id, testText.trim());
      setPreview(result);
      setPreviewVersion((v) => v + 1);
      setPeaks(null);
      setPlayed(0);
    } catch (error) {
      setStepError((error as Error).message);
    } finally {
      setPreviewing(false);
    }
  };

  // Real peaks decoded from the stored clip's own PCM, the same function the
  // diarization waveform uses. Never drawn for a non-wav container: no picture
  // beats a wrong one.
  useEffect(() => {
    if (!working || !preview || preview.audioFormat !== "wav") return;
    let cancelled = false;
    void decodeWaveformPeaks(clonePreviewAudioUrl(working.id, false, previewVersion), 120)
      .then((decoded) => { if (!cancelled) setPeaks(decoded); })
      .catch((error: Error) => console.error("Could not decode the preview waveform:", error));
    return () => { cancelled = true; };
  }, [working, preview, previewVersion]);

  const handleDelete = async (voiceId: number) => {
    try {
      await deleteClonedVoice(voiceId);
      if (working?.id === voiceId) {
        setWorking(null);
        setPreview(null);
      }
      await refreshVoices();
    } catch (error) {
      setStepError((error as Error).message);
    }
  };

  const registered = voices.filter((v) => v.status === "registered");
  const canExtract = Boolean(cloneConfig?.configured) && audioUrl.trim() !== "" && promptText.trim() !== "";
  const promptOverLimit =
    cloneConfig?.maxPromptChars != null && promptText.trim().length > cloneConfig.maxPromptChars;

  if (!cloneConfig?.configured) {
    return (
      <p className="transcript-notice">
        Voice cloning is not configured on this host — set HAMSA_VOICE_CLONE_URL,
        HAMSA_LOAD_VOICE_URL, HAMSA_TTS_KEY and HAMSA_TTS_BEARER_TOKEN in .env. Nothing
        below would run, so it is not offered.
      </p>
    );
  }

  return (
    <>
      <section className="transcript-grid">
        <div className="panel script-controls">
          <h2>Read it out</h2>
          <p className="muted" style={{ fontSize: 12, marginTop: -4 }}>
            Generate the passage to read into the reference clip — it becomes the prompt text.
          </p>

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
              <small className="muted">Prompt text must match the audio exactly.</small>
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
            <button type="button" className="generate-btn" onClick={handleGenerate} disabled={generating}>
              {generating ? "Generating…" : "Generate new script"}
            </button>
          ) : (
            <small className="muted">No script gateway configured on this host.</small>
          )}
          {scriptError && <p className="transcript-error">{scriptError}</p>}
        </div>

        <div className="panel script-body">
          <div className="script-meta">
            <span className="mono">
              {script
                ? `REFERENCE SCRIPT · ${script.params?.minutes ?? effectiveMinutes} min · ${script.wordCount} words · ${script.text.length} chars`
                : "REFERENCE SCRIPT"}
            </span>
            {script && promptText.trim() !== script.text.trim() && (
              // The prompt is what gets SENT; this box is what was generated.
              // Once they differ, saying so beats letting the reader assume the
              // passage above is the transcript being aligned against the audio.
              <span className="script-dirty mono">prompt text edited — step 1 is what is sent</span>
            )}
            {script?.generatorModel && <span className="script-model">{script.generatorModel}</span>}
          </div>
          {/* Read-only, and step 1's prompt box is the editable authority. Both
              were editable and bound to one state, which put two identical
              textareas on screen showing the same words -- the "two controls for
              one value" shape CLAUDE.md already calls out for the STT/TTS script
              box. Kept as a textarea rather than a <p>, with the same class, so
              generating a script does not change the shape the text occupies.
              It still earns its place: once the prompt is edited, this shows
              what was actually generated, so the drift between the passage and
              the clip's real transcript is visible instead of lost. */}
          <textarea
            className="script-paste"
            dir="auto"
            readOnly
            placeholder="Generate a passage to read aloud — it fills the prompt text in step 1."
            value={script?.text ?? ""}
          />
        </div>
      </section>

      {/* Steps 1 and 2 sit side by side: extraction's output is what step 1
          produces, and reading them apart made the tokens look unrelated to the
          clip they came from. */}
      <section className="clone-grid">
        <div className="panel clone-step" style={{ "--engine": "var(--warn)" } as React.CSSProperties}>
          <div className="clone-step-head">
            <span className="clone-step-n">1</span>
            <h2>Reference clip</h2>
            <span className="mono muted clone-endpoint">POST /tts/voice_clone</span>
          </div>

          <label className="control-label" htmlFor="clone-url">
            Audio URL — downloaded by the pod, not by this API
          </label>
          <input
            id="clone-url"
            className="clone-input mono"
            type="url"
            dir="ltr"
            placeholder="https://…/reference.wav"
            value={audioUrl}
            onChange={(event) => setAudioUrl(event.target.value)}
            disabled={extracting}
          />
          {/* No reachability badge. A HEAD from this host would prove the URL
              resolves HERE, which says nothing about whether the pod can reach
              it — and a green tick that means the wrong thing is worse than no
              tick. The pod's own answer is the only real signal. */}
          <small className="muted">
            {cloneConfig.uploadsReachable ? (
              <>
                Or upload a clip — it is stored here and the resulting URL is used.{" "}
                <label className="clone-upload-link">
                  choose a file
                  <input
                    type="file"
                    accept="audio/*"
                    hidden
                    disabled={uploading || extracting}
                    onChange={(event) => {
                      const file = event.target.files?.[0];
                      if (file) void handleUpload(file);
                    }}
                  />
                </label>
                {uploading ? " · uploading…" : null}
                {uploaded?.audioSec != null ? ` · ${uploaded.audioSec.toFixed(1)}s uploaded` : null}
              </>
            ) : (
              // Stated rather than hidden: an operator who cannot find the
              // upload control needs to know it is a host setting, not a bug.
              "Uploads are unavailable on this host: the pod fetches the clip itself and cannot reach this API. Set VOICE_CLONE_PUBLIC_BASE_URL to enable them."
            )}
          </small>

          <label className="control-label clone-prompt-label" htmlFor="clone-prompt">
            Prompt text — transcript of the clip
            <span className="mono muted">must match the audio exactly</span>
          </label>
          <textarea
            id="clone-prompt"
            className="script-paste clone-prompt"
            dir="auto"
            placeholder="What is said in the reference clip, word for word."
            value={promptText}
            onChange={(event) => setPromptText(event.target.value)}
            disabled={extracting}
          />
          {promptOverLimit && (
            <p className="transcript-error">
              {promptText.trim().length} chars, over this host's {cloneConfig.maxPromptChars}-char
              limit — extraction will be rejected.
            </p>
          )}

          <p className="clone-note">
            The pod uses this transcript to separate voice identity from spoken content, so an
            approximate one weakens the clone. Shorter or noisy references produce a weaker clone;
            the API enforces no minimum.
          </p>

          <button
            type="button"
            className="generate-btn"
            onClick={() => void handleExtract()}
            disabled={!canExtract || extracting || promptOverLimit}
            title={
              !audioUrl.trim()
                ? "A reference clip URL is required"
                : !promptText.trim()
                  ? "The clip's transcript is required"
                  : "Extract voice tokens"
            }
          >
            {extracting ? "Extracting…" : "Extract voice tokens"}
          </button>
        </div>

        <div className="panel clone-step clone-extracted" style={{ "--engine": "var(--good)" } as React.CSSProperties}>
          <div className="clone-step-head">
            <span className="clone-step-n">2</span>
            <h2>Extracted</h2>
            {working?.status === "failed" ? (
              <span className="clone-badge is-bad mono">FAILED</span>
            ) : working && working.extractMs != null ? (
              <span className="clone-badge is-good mono">DONE {fmtMs(working.extractMs)}</span>
            ) : null}
          </div>

          {working?.status === "failed" ? (
            // The pod's message verbatim. The failure this endpoint actually
            // produces is an opaque text/plain 500, and rewording it would
            // remove the only clue there is.
            <p className="transcript-error clone-upstream">{working.error ?? "Extraction failed."}</p>
          ) : working ? (
            <>
              <div className="clone-token-block">
                <div className="clone-token-head">
                  <span className="mono">global_token_ids</span>
                  <b className="mono">{working.globalTokenCount ?? "—"}</b>
                </div>
                <TokenBars count={working.globalTokenCount ?? 0} />
              </div>
              <div className="clone-token-block">
                <div className="clone-token-head">
                  <span className="mono">semantic_token_ids</span>
                  <b className="mono">{working.semanticTokenCount ?? "—"}</b>
                </div>
                <TokenBars count={working.semanticTokenCount ?? 0} />
              </div>
              {/* Bars are a fixed shape scaled to the count, not a plot of the
                  values: the tokens are the pod's opaque representation and this
                  repo never interprets them. Said out loud so the picture is not
                  read as data. */}
              <small className="muted">Bar row is scaled to the count, not a plot of the values.</small>

              <div className="clone-echo">
                <span className="eyebrow">Prompt text returned</span>
                <p dir="auto">{working.returnedPromptText || "—"}</p>
                <small className={working.returnedPromptText === working.promptText ? "clone-match" : "muted"}>
                  {working.returnedPromptText === working.promptText
                    ? "Matches what you sent — carried into registration."
                    : "Differs from what was sent; the pod's copy is what it kept."}
                </small>
              </div>
            </>
          ) : (
            <p className="muted">Nothing extracted yet.</p>
          )}
        </div>
      </section>

      <section className="panel clone-step clone-register" style={{ "--engine": "var(--accent)" } as React.CSSProperties}>
        <div className="clone-step-head">
          <span className="clone-step-n">3</span>
          <h2>Register the voice</h2>
          <span className="mono muted clone-endpoint">
            POST /tts/load_voice_cloning · returns null on success
          </span>
        </div>

        <div className="clone-register-row">
          <div className="clone-field">
            <label className="control-label" htmlFor="clone-speaker">speaker_id</label>
            <input
              id="clone-speaker"
              className="clone-input mono"
              dir="ltr"
              placeholder="mariam"
              value={speakerId}
              onChange={(event) => setSpeakerId(event.target.value)}
              disabled={registering}
            />
            <small className="muted">
              Case-sensitive at registration, looked up case-insensitively when synthesizing.
            </small>
            {nameClash && (
              <span className="clone-badge is-warn mono">EXISTS HERE — REGISTER WOULD CLASH</span>
            )}
          </div>

          <div className="clone-field">
            <span className="control-label">dialect</span>
            <Segmented
              value={dialect}
              options={cloneConfig.dialects.map((id) => ({ value: id, label: id }))}
              onChange={setDialect}
              label="Dialect"
              disabled={registering}
            />
            <small className="muted">From .env — the pod has no list-dialects route.</small>
          </div>

          <div className="clone-field">
            <span className="control-label">Tokens attached</span>
            <span className="mono clone-token-summary">
              {working?.globalTokenCount != null
                ? `global ${working.globalTokenCount} · semantic ${working.semanticTokenCount ?? 0}`
                : "— nothing extracted yet"}
            </span>
            <small className="muted">Re-posted verbatim from step 2.</small>
          </div>

          <button
            type="button"
            className="generate-btn clone-register-btn"
            onClick={() => void handleRegister()}
            disabled={
              !working || working.status === "failed" || !speakerId.trim() || registering || nameClash
            }
            title={
              !working
                ? "Extract voice tokens first"
                : nameClash
                  ? "That name already exists in this catalogue"
                  : "Register voice"
            }
          >
            {registering ? "Registering…" : working?.status === "registered" ? "Re-register" : "Register voice"}
          </button>
        </div>
      </section>

      <section className="clone-grid">
        <div className="panel clone-step" style={{ "--engine": "var(--good)" } as React.CSSProperties}>
          <div className="clone-step-head">
            <span className="clone-step-n">4</span>
            <h2>Test the clone</h2>
            {working?.status === "registered" ? (
              <span className="clone-badge is-good mono">REGISTERED · USABLE NOW</span>
            ) : null}
            <span className="mono muted clone-endpoint">
              {previewEngine ? `via ${previewEngine.name}` : `via ${cloneConfig.previewTtsId}`}
              {working?.speakerId ? ` · speaker=${working.speakerId}` : ""}
            </span>
          </div>

          <textarea
            className="script-paste clone-prompt"
            dir="auto"
            placeholder="Type something for the cloned voice to say — anything, it need not match the reference."
            value={testText}
            onChange={(event) => setTestText(event.target.value)}
            disabled={previewing}
          />

          {preview ? (
            <>
              <audio
                ref={audioRef}
                src={working ? clonePreviewAudioUrl(working.id, false, previewVersion) : undefined}
                preload="none"
                onPlay={() => setPlaying(true)}
                onPause={() => setPlaying(false)}
                onEnded={() => { setPlaying(false); setPlayed(0); }}
                onTimeUpdate={(event) => {
                  const el = event.currentTarget;
                  if (!el.duration || !Number.isFinite(el.duration)) return;
                  setPlayed(el.currentTime / el.duration);
                }}
              />
              <div className="tts-clip-row">
                <button
                  type="button"
                  className="record-btn is-playback tts-play"
                  onClick={() => {
                    const audio = audioRef.current;
                    if (!audio) return;
                    if (audio.paused) void audio.play().catch(() => undefined);
                    else audio.pause();
                  }}
                  aria-label={playing ? "Pause the cloned voice" : "Play the cloned voice"}
                >
                  <span className={`play-glyph${playing ? " is-playing" : ""}`} aria-hidden="true" />
                </button>
                {peaks ? (
                  <div
                    className="tts-wave"
                    role="presentation"
                    onClick={(event) => {
                      const audio = audioRef.current;
                      if (!audio?.duration || !Number.isFinite(audio.duration)) return;
                      const box = event.currentTarget.getBoundingClientRect();
                      const ratio = Math.min(1, Math.max(0, (event.clientX - box.left) / box.width));
                      audio.currentTime = ratio * audio.duration;
                      setPlayed(ratio);
                    }}
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
                  <span className="muted mono tts-wave-absent">
                    {preview.audioFormat !== "wav" ? `no waveform for ${preview.audioFormat}` : "decoding…"}
                  </span>
                )}
                <span className="mono tts-clip-clock">
                  {preview.audioSec != null ? fmtClock(preview.audioSec) : "—"}
                </span>
              </div>
              <div className="tts-clip-meta">
                <span className="mono muted">{clipSummary(preview) || "format not reported"}</span>
                {working && (
                  <a className="tts-download" href={clonePreviewAudioUrl(working.id, true, previewVersion)} download>
                    ↓ Download {preview.audioFormat.toUpperCase()}
                  </a>
                )}
              </div>
              <div className="tts-tiles">
                <div className="score-tile">
                  <span className="eyebrow">Time to first audio</span>
                  <b className="mono">{preview.firstAudioMs != null ? fmtMs(preview.firstAudioMs) : "—"}</b>
                  <small className="muted">first chunk off the stream</small>
                </div>
                <div className="score-tile">
                  <span className="eyebrow">Synthesis time</span>
                  <b className="mono">{preview.synthMs != null ? fmtMs(preview.synthMs) : "—"}</b>
                  <small className="muted">
                    {preview.audioSec != null ? `for ${preview.audioSec.toFixed(1)} s of audio` : "wall clock"}
                  </small>
                </div>
              </div>
              {/* The reference clip is NOT rendered beside this unless this API
                  actually stored it. On a host cloning from someone else's URL
                  we never downloaded those bytes, and drawing a player for audio
                  we do not hold would be a control that cannot work. */}
              {working?.hasStoredClip ? (
                <small className="muted">Reference clip is stored here and can be compared byte for byte.</small>
              ) : (
                <small className="muted">
                  Reference clip was fetched by the pod from its URL; this API kept no copy, so there
                  is nothing local to play it back from.
                </small>
              )}
            </>
          ) : (
            <p className="muted">
              {working?.status === "registered"
                ? "Not synthesized yet."
                : "Register a voice first — the pod cannot synthesize with a name it does not hold."}
            </p>
          )}

          <button
            type="button"
            className="generate-btn"
            onClick={() => void handlePreview()}
            disabled={working?.status !== "registered" || !testText.trim() || previewing}
          >
            {previewing ? "Synthesizing…" : preview ? "Synthesize again" : "Synthesize"}
          </button>
        </div>

        <div className="panel clone-registry">
          <div className="clone-step-head">
            <h2>Voice registry</h2>
            <span className="mono muted">
              {previewEngine ? `${previewEngine.voices.length} built-in · ` : ""}
              {registered.length} cloned
            </span>
          </div>
          {/* "built-in" is the count .env holds for this engine, NOT a count
              discovered from the pod: it has no list-voices route, so the real
              number of voices it knows is not observable from here. */}
          <small className="muted">
            Built-in count is what .env lists for this engine; the pod exposes no way to enumerate
            its own voices.
          </small>

          <ul className="clone-voice-list">
            {loadingVoices && voices.length === 0 && <li className="muted">Loading…</li>}
            {!loadingVoices && voices.length === 0 && (
              <li className="muted">Nothing cloned on this host yet.</li>
            )}
            {voices.map((voice) => (
              <li key={voice.id} className={voice.id === working?.id ? "is-active" : undefined}>
                <span className="clone-avatar" aria-hidden="true">{initial(voice.speakerId || "?")}</span>
                <span className="clone-voice-main">
                  <b className="mono">{voice.speakerId || <em className="muted">unnamed</em>}</b>
                  <small className="muted">
                    {voice.dialect}
                    {voice.audioSec != null ? ` · ${voice.audioSec.toFixed(1)}s ref` : ""}
                    {voice.globalTokenCount != null ? ` · ${voice.globalTokenCount} global` : ""}
                  </small>
                </span>
                <span
                  className={`clone-badge mono ${
                    voice.status === "registered" ? "is-good" : voice.status === "failed" ? "is-bad" : "is-warn"
                  }`}
                >
                  {voice.status === "registered" ? "CLONED" : voice.status === "failed" ? "FAILED" : "TOKENS ONLY"}
                </span>
                <button
                  type="button"
                  className="clone-forget"
                  onClick={() => void handleDelete(voice.id)}
                  title="Forget this voice locally — the pod keeps it until it restarts"
                  aria-label={`Forget ${voice.speakerId || "this voice"}`}
                >
                  ×
                </button>
              </li>
            ))}
          </ul>

          {/* The single most surprising property of this feature, stated where
              the list of voices is, because that list is exactly what a restart
              invalidates. */}
          <p className="clone-note">
            This is what THIS deployment cloned, not what the pod currently holds. The pod keeps
            registrations in memory and exposes no route to list them, so a restart drops every
            voice here and re-registering is what brings them back. Neither cloning call is billed,
            but each holds a concurrency slot while it runs.
          </p>
        </div>
      </section>

      {stepError && <p className="transcript-error">{stepError}</p>}
    </>
  );
}
