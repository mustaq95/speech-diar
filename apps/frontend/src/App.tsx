import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ConfigBar } from "./components/ConfigBar";
import { EmptyDashboard } from "./components/EmptyDashboard";
import { Insights } from "./components/Insights";
import { ModelStatusStrip } from "./components/ModelStatusStrip";
import { ProcessingScreen } from "./components/ProcessingScreen";
import { ProjectsView } from "./components/ProjectsView";
import { SettingsView } from "./components/SettingsView";
import { MIN_PX_PER_SEC, Studio } from "./components/Studio";
import { TopBar } from "./components/TopBar";
import { UploadingScreen } from "./components/UploadingScreen";
import { boundsFor, buildEvents, fmt, loadModelActive, loadProjects, saveModelActive, saveProjects } from "./utils";
import {
  DEMO_AUDIO_FILE_ID,
  audioStreamUrl,
  deriveDefaultEval,
  deriveDefaultParams,
  fetchEvaluation,
  fetchModelCatalog,
  fetchModelStatus,
  fetchRuntimeConfig,
  getDiarizationEvaluation,
  mergeActiveWithCatalog,
  modelRunFromMetadata,
  patchUploadTiming,
  retryModel,
  uploadAudio,
} from "./adapters";
import { speakerSignature, decodeWaveformPeaks } from "./playback";
import { isInFlight } from "./timing";
import type { ActiveMap, AvailableMap, DiarizationEvaluation, EvalConfig, ModelId, ModelMetadata, ModelRun, Nav, ParamMap, Project, Workflow } from "./types";
import type { ModelContainerStatus } from "./types/diarization";

const FLAT_WAVE_PEAKS = Array.from({ length: 210 }, () => 0.3);
const DEFAULT_POLL_INTERVAL_MS = 1500;
// Aim for a waveform bar roughly every few px of the zoom=1 timeline (Studio's
// MIN_PX_PER_SEC), so long recordings don't stretch a fixed bar count into a
// blocky bar chart. Floor keeps short clips unchanged; cap bounds DOM nodes.
const TARGET_PX_PER_BAR = 3;
const MIN_WAVE_BARS = 210;
const MAX_WAVE_BARS = 3000;

export default function App() {
  // All diarization data enters the UI through the adapter layer as the
  // unified contract — never as a backend's raw payload. No evaluation is
  // loaded until the user uploads real audio or explicitly asks for the demo.
  const [evaluation, setEvaluation] = useState<DiarizationEvaluation | null>(null);
  const isDemo = evaluation?.audioFileId === DEMO_AUDIO_FILE_ID;
  const models = evaluation?.models ?? [];
  const duration = evaluation?.durationSec ?? 0;

  const [catalog, setCatalog] = useState<ModelMetadata[]>([]);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [pollIntervalMs, setPollIntervalMs] = useState(DEFAULT_POLL_INTERVAL_MS);
  const [modelStatus, setModelStatus] = useState<ModelContainerStatus[]>([]);
  const catalogModels = useMemo(() => catalog.map(modelRunFromMetadata), [catalog]);
  const availableMap: AvailableMap = useMemo(
    () => Object.fromEntries(catalog.map((meta) => [meta.id, meta.available])),
    [catalog],
  );

  const [nav, setNav] = useState<Nav>("upload");
  const [workflow, setWorkflow] = useState<Workflow>("idle");
  const [uploadPct, setUploadPct] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [projects, setProjects] = useState<Project[]>(() => loadProjects());
  const [current, setCurrent] = useState<Project | null>(null);
  const [active, setActive] = useState<ActiveMap>({});
  const [evalCfg, setEvalCfg] = useState<EvalConfig>(() => deriveDefaultEval({ audioFileId: 0, durationSec: 0, models: [] }));
  const [params, setParams] = useState<ParamMap>({});
  const [streamInline, setStreamInline] = useState(true);
  const [glow, setGlow] = useState(true);
  const [feed, setFeed] = useState(true);
  const [speakerTick, setSpeakerTick] = useState(0);
  const [uploadName, setUploadName] = useState("");
  const [wavePeaks, setWavePeaks] = useState<number[]>(FLAT_WAVE_PEAKS);
  const [nowTick, setNowTick] = useState(() => Date.now());
  // Audio is primary; the waveform is best-effort. `audioReady` gates the
  // heavy waveform decode so it never competes with the audio element's own
  // load. `audioNotice` replaces blocking alerts on the recovery path.
  const [audioReady, setAudioReady] = useState(false);
  const [audioNotice, setAudioNotice] = useState<string | null>(null);

  const timeRef = useRef(0);
  const playingRef = useRef(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const sigRef = useRef("");
  const lastTsRef = useRef<number | null>(null);
  const playheadRef = useRef<HTMLDivElement | null>(null);
  const waveFillRef = useRef<HTMLDivElement | null>(null);
  const clockRef = useRef<HTMLElement | null>(null);
  const clock2Ref = useRef<HTMLElement | null>(null);
  const miniFillRef = useRef<HTMLDivElement | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const innerRef = useRef<HTMLDivElement | null>(null);
  const openRequestRef = useRef(0);
  const seekCommitTimer = useRef<number | null>(null);

  const playableUrl = evaluation && !isDemo ? audioStreamUrl(evaluation.audioFileId) : null;
  const hasInFlight = useMemo(() => models.some(isInFlight), [models]);
  const hasFailed = useMemo(() => models.some((model) => model.status === "failed"), [models]);

  const events = useMemo(() => buildEvents(models), [models]);
  const shownModels = useMemo(() => models.filter((model) => active[model.id]), [active, models]);
  const fileName = isDemo ? "Synthetic demo (no real audio)" : (current?.name ?? uploadName) || "No recording loaded";
  const durationText = fmt(duration);
  // Dashboard always shows the Studio shell; before anything is uploaded it
  // falls back to the configured model catalog so the layout looks the same.
  // With an evaluation open, append any model the user toggled on that never
  // ran here (e.g. a model added after this recording was uploaded) as a
  // shell row, so it shows immediately and can be run via the ↻ button.
  const dashboardModels = useMemo(() => {
    if (!evaluation) return catalogModels;
    const present = new Set(models.map((model) => model.id));
    const extra = catalogModels.filter((model) => active[model.id] && !present.has(model.id));
    return extra.length ? [...models, ...extra] : models;
  }, [evaluation, models, catalogModels, active]);

  // Fetch the real, honest model registry and runtime config once at boot.
  useEffect(() => {
    fetchModelCatalog()
      .then((list) => setCatalog(list))
      .catch((error: Error) => setCatalogError(`Could not load model list: ${error.message}`));
    fetchRuntimeConfig()
      .then((cfg) => setPollIntervalMs(cfg.pollIntervalMs))
      .catch(() => undefined);
  }, []);

  // Seed "which models run next" once the catalog first loads, merging in any
  // Settings choice already saved in localStorage. A one-time ref guard
  // instead of an `evaluation` dependency: starting a new recording or
  // clearing the loaded evaluation must never re-enable models the user
  // turned off in Settings.
  const seededActiveRef = useRef(false);
  useEffect(() => {
    if (catalog.length === 0 || seededActiveRef.current) return;
    seededActiveRef.current = true;
    setActive(mergeActiveWithCatalog(loadModelActive() ?? {}, catalog));
  }, [catalog]);

  // Reset param/eval-config defaults to the catalog shell whenever there's no
  // real evaluation loaded (before first upload, or after "+ New recording").
  useEffect(() => {
    if (catalog.length === 0 || evaluation) return;
    const shell: DiarizationEvaluation = { audioFileId: 0, durationSec: 0, models: catalogModels };
    setParams(deriveDefaultParams(shell));
    setEvalCfg(deriveDefaultEval(shell));
  }, [catalog, catalogModels, evaluation]);

  // Poll the real evaluation while any model is queued/running.
  useEffect(() => {
    if (!evaluation || isDemo || !hasInFlight) return;
    const audioFileId = evaluation.audioFileId;
    const id = window.setInterval(() => {
      fetchEvaluation(audioFileId)
        .then((next) => setEvaluation((prev) => (prev && prev.audioFileId === audioFileId ? { ...next, uploadMs: prev.uploadMs ?? next.uploadMs } : prev)))
        .catch((error: Error) => console.error("Poll failed:", error));
    }, pollIntervalMs);
    return () => window.clearInterval(id);
  }, [evaluation, isDemo, hasInFlight, pollIntervalMs]);

  // Model GPU-residency status is platform-wide, not tied to any one
  // evaluation's in-flight state — a model can start loading or unloading
  // because of a completely different evaluation's job. Poll continuously,
  // independent of `hasInFlight`, starting immediately rather than waiting
  // out the first interval tick.
  useEffect(() => {
    const poll = () => fetchModelStatus().then(setModelStatus).catch((error: Error) => console.error("Model status poll failed:", error));
    void poll();
    const id = window.setInterval(poll, pollIntervalMs);
    return () => window.clearInterval(id);
  }, [pollIntervalMs]);

  // Auto-advance from the dedicated processing screen once every model has settled.
  // Navigation is intentionally left untouched here: when "open dashboard on upload"
  // is off, the user decides when to switch to the Dashboard tab to see results.
  //
  // A failure holds the screen open: the per-model re-run button lives on these
  // cards, so dismissing on settle would pull it away exactly when it's wanted.
  // The Dashboard tab is still one click away, so nobody is stuck here.
  useEffect(() => {
    if (workflow === "processing" && evaluation && !hasInFlight && !hasFailed) setWorkflow("idle");
  }, [workflow, evaluation, hasInFlight, hasFailed]);

  // The saved project's speaker count is captured at upload time (before any
  // model has run) — refresh it once processing settles so the Projects list
  // doesn't keep showing a stale "0 speakers detected".
  useEffect(() => {
    if (!evaluation || isDemo || hasInFlight || !current || current.audioFileId !== evaluation.audioFileId) return;
    const speakers = Math.max(0, ...evaluation.models.map((run) => run.numSpk));
    if (speakers === current.speakers) return;
    const updated: Project = { ...current, speakers };
    setCurrent(updated);
    setProjects((prev) => {
      const next = prev.map((project) => (project.id === updated.id ? updated : project));
      saveProjects(next);
      return next;
    });
  }, [evaluation, isDemo, hasInFlight, current]);

  // Live "now" for elapsed-time labels, ticking only while something is running.
  useEffect(() => {
    if (!hasInFlight) return;
    const id = window.setInterval(() => setNowTick(Date.now()), 500);
    return () => window.clearInterval(id);
  }, [hasInFlight]);

  // Real waveform: decode the actual audio bytes once per evaluation (never a synthetic shape).
  // Deferred until the audio element has validated its source (`audioReady`),
  // so the full-file decode never competes with audio's own load. On a long
  // file under heavy load that race is what left the audio unplayable.
  useEffect(() => {
    if (!playableUrl) {
      setWavePeaks(FLAT_WAVE_PEAKS);
      return;
    }
    if (!audioReady) return;
    let cancelled = false;
    const bars = Math.min(MAX_WAVE_BARS, Math.max(MIN_WAVE_BARS, Math.round((duration * MIN_PX_PER_SEC) / TARGET_PX_PER_BAR)));
    decodeWaveformPeaks(playableUrl, bars)
      .then((peaks) => { if (!cancelled) setWavePeaks(peaks); })
      .catch((error: Error) => {
        console.error("Waveform decode failed:", error);
        if (!cancelled) setWavePeaks(FLAT_WAVE_PEAKS);
      });
    return () => { cancelled = true; };
  }, [playableUrl, duration, audioReady]);

  const syncDom = useCallback((nextTime = timeRef.current) => {
    const pct = duration > 0 ? nextTime / duration : 0;
    if (playheadRef.current) playheadRef.current.style.left = `${pct * 100}%`;
    if (waveFillRef.current) waveFillRef.current.style.width = `${pct * 100}%`;
    if (miniFillRef.current) miniFillRef.current.style.width = `${pct * 100}%`;
    if (clockRef.current) clockRef.current.textContent = fmt(nextTime);
    if (clock2Ref.current) clock2Ref.current.textContent = fmt(nextTime);
    if (playingRef.current && scrollRef.current && innerRef.current) {
      const px = pct * innerRef.current.offsetWidth;
      const view = scrollRef.current.clientWidth;
      const left = scrollRef.current.scrollLeft;
      if (px < left + view * 0.15) scrollRef.current.scrollLeft = Math.max(0, px - view * 0.15);
      else if (px > left + view * 0.82) scrollRef.current.scrollLeft = px - view * 0.82;
    }
  }, [duration]);

  const setPlaybackTime = useCallback((next: number, commit = true) => {
    const clamped = Math.max(0, Math.min(duration, next));
    timeRef.current = clamped;
    if (audioRef.current && playableUrl) {
      const delta = Math.abs(audioRef.current.currentTime - clamped);
      if (delta > 0.05) audioRef.current.currentTime = clamped;
    }
    // Audio position and the playhead/wave-fill/clock (all DOM refs, via
    // syncDom) update immediately on every call, so a seek always feels
    // instant. The expensive part -- re-rendering every model's segment
    // list to update the active-segment highlight -- is coalesced below,
    // so a burst of rapid clicks commits once instead of once per click.
    syncDom(clamped);
    if (!commit) return;
    if (seekCommitTimer.current != null) window.clearTimeout(seekCommitTimer.current);
    seekCommitTimer.current = window.setTimeout(() => {
      seekCommitTimer.current = null;
      const finalTime = timeRef.current;
      const sig = speakerSignature(shownModels, finalTime);
      if (sig !== sigRef.current) {
        sigRef.current = sig;
        setSpeakerTick((tick) => tick + 1);
      }
      setTime(finalTime);
    }, 60);
  }, [playableUrl, duration, shownModels, syncDom]);

  useEffect(() => {
    playingRef.current = playing;
  }, [playing]);

  useEffect(() => {
    // Only ever scheduled while actually playing -- an idle 60fps loop while
    // paused wastes CPU for no reason. Restarted by the `playing` dependency
    // whenever playback resumes; the running loop reschedules itself below.
    let raf = 0;
    const tick = (ts: number) => {
      if (!playingRef.current) {
        lastTsRef.current = null;
        return;
      }
      let next: number;
      const audio = audioRef.current;
      if (playableUrl && audio) {
        next = Math.min(duration, audio.currentTime);
      } else {
        if (lastTsRef.current == null) lastTsRef.current = ts;
        const dt = (ts - lastTsRef.current) / 1000;
        lastTsRef.current = ts;
        next = Math.min(duration, timeRef.current + dt);
      }
      timeRef.current = next;
      syncDom(next);
      const sig = speakerSignature(shownModels, next);
      if (sig !== sigRef.current) {
        sigRef.current = sig;
        setSpeakerTick((tick) => tick + 1);
        setTime(next);
      }
      if (next >= duration) {
        playingRef.current = false;
        setPlaying(false);
        audio?.pause();
        setTime(duration);
        return;
      }
      raf = requestAnimationFrame(tick);
    };
    if (playing) raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, playableUrl, duration, shownModels, syncDom]);

  useEffect(() => {
    timeRef.current = time;
    syncDom(time);
  }, [time, syncDom, speakerTick]);

  const persistProjects = (next: Project[]) => {
    setProjects(next);
    saveProjects(next);
  };

  // Any completed action that produces a ready evaluation (demo, upload, open
  // project) sends the user straight to the Dashboard tab to see the result.
  // That's the point of the action, not unwanted tab coupling — but nothing
  // else about tab access depends on it.
  const startFlow = useCallback(() => {
    audioRef.current?.pause();
    const demo = getDiarizationEvaluation();
    setEvaluation(demo);
    setActive(Object.fromEntries(demo.models.map((model) => [model.id, true])));
    setParams(deriveDefaultParams(demo));
    setEvalCfg(deriveDefaultEval(demo));
    setCurrent(null);
    timeRef.current = 0;
    sigRef.current = "";
    setTime(0);
    setNav("dashboard");
    setWorkflow("idle");
    setPlaying(false);
  }, []);

  const startRealUpload = useCallback((file: File) => {
    const modelIds: ModelId[] = catalogModels.filter((model) => active[model.id]).map((model) => model.id);
    if (modelIds.length === 0) {
      window.alert("Enable at least one model in Settings before uploading.");
      return;
    }
    audioRef.current?.pause();
    setEvaluation(null);
    setUploadName(file.name);
    setNav("upload");
    setWorkflow("uploading");
    setUploadPct(0);
    setPlaying(false);

    const uploadStart = performance.now();
    uploadAudio(file, modelIds, (pct) => setUploadPct(pct))
      .then(async (ack) => {
        const uploadMs = Math.round(performance.now() - uploadStart);
        const initial = await fetchEvaluation(ack.audioFileId);
        void patchUploadTiming(ack.audioFileId, uploadMs).catch((error: Error) => console.error("Could not save upload time:", error));
        const withUpload: DiarizationEvaluation = { ...initial, uploadMs };
        setEvaluation(withUpload);
        setActive(Object.fromEntries(withUpload.models.map((model) => [model.id, true])));
        setParams(deriveDefaultParams(withUpload));
        setEvalCfg(deriveDefaultEval(withUpload));
        const project: Project = {
          id: Date.now(),
          audioFileId: ack.audioFileId,
          name: file.name,
          date: new Date().toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" }),
          duration: fmt(withUpload.durationSec),
          models: withUpload.models.length,
          speakers: Math.max(0, ...withUpload.models.map((run) => run.numSpk)),
          fresh: true,
        };
        persistProjects([project, ...projects]);
        setCurrent(project);
        timeRef.current = 0;
        sigRef.current = "";
        setTime(0);
        // "Open dashboard as soon as audio uploads" is what actually moves the
        // user off the Upload tab; when it's off they stay put and can switch
        // to Dashboard whenever they choose.
        if (streamInline) setNav("dashboard");
        setWorkflow(streamInline ? "idle" : "processing");
      })
      .catch((error: Error) => {
        console.error(error);
        window.alert(`Upload failed: ${error.message}`);
        setWorkflow("idle");
      });
  }, [active, catalogModels, projects, streamInline]);

  // Play, self-healing under load. "No supported sources" means the browser
  // abandoned a source that stalled (networkState NETWORK_NO_SOURCE); a fresh
  // load() re-arms it. Retry a couple of times, waiting for `canplay` rather
  // than hammering play(), and restore the intended position so a reload
  // doesn't jump back to 0. Only after retries are spent do we tell the user,
  // inline, never a blocking alert.
  const attemptPlay = useCallback((targetTime: number, tries = 0) => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.play().then(() => {
      setAudioNotice(null);
      if (Math.abs(audio.currentTime - targetTime) > 0.3) audio.currentTime = targetTime;
    }).catch((error: Error) => {
      if (tries >= 2) {
        console.error("Audio play failed after retries:", error);
        setPlaying(false);
        setAudioNotice("Audio is still loading under heavy processing — press play again in a moment.");
        return;
      }
      const onReady = () => {
        audio.removeEventListener("canplay", onReady);
        audio.currentTime = targetTime;
        attemptPlay(targetTime, tries + 1);
      };
      audio.addEventListener("canplay", onReady, { once: true });
      audio.load();
    });
  }, []);

  const openProject = (project: Project) => {
    const requestId = ++openRequestRef.current;
    audioRef.current?.pause();
    setCurrent(project);
    setEvaluation(null);
    setPlaying(false);
    timeRef.current = 0;
    sigRef.current = "";
    setTime(0);
    setNav("dashboard");
    setWorkflow("loading");
    fetchEvaluation(project.audioFileId)
      .then((result) => {
        if (openRequestRef.current !== requestId) return; // a newer open superseded this one
        setEvaluation(result);
        setActive(Object.fromEntries(result.models.map((model) => [model.id, true])));
        setParams(deriveDefaultParams(result));
        setEvalCfg(deriveDefaultEval(result));
        setWorkflow("idle");
      })
      .catch((error: Error) => {
        if (openRequestRef.current !== requestId) return;
        console.error(error);
        window.alert(`Could not reopen recording: ${error.message}`);
        setWorkflow("idle");
        setCurrent(null);
      });
  };

  // Requeue one model against the audio already uploaded. The response carries
  // that model back as `queued`, which flips `hasInFlight` and re-arms the poll
  // effect above on its own — no separate refresh needed.
  const handleRetry = useCallback(
    async (modelId: string) => {
      if (!evaluation) return;
      const next = await retryModel(evaluation.audioFileId, modelId);
      setEvaluation((prev) =>
        prev && prev.audioFileId === next.audioFileId ? { ...next, uploadMs: prev.uploadMs ?? next.uploadMs } : prev,
      );
    },
    [evaluation],
  );

  // Clears the loaded session and sends the user to a fresh Upload tab —
  // used by "+ New recording", not by the Upload tab button itself (which
  // must never destroy a loaded recording just for switching tabs).
  const startNewUpload = () => {
    audioRef.current?.pause();
    timeRef.current = 0;
    sigRef.current = "";
    setTime(0);
    setCurrent(null);
    setEvaluation(null);
    setNav("upload");
    setWorkflow("idle");
    setUploadPct(0);
    setPlaying(false);
  };

  const step = (dir: -1 | 1) => {
    const bounds = boundsFor(shownModels);
    const target = dir > 0
      ? bounds.find((bound) => bound > timeRef.current + 0.3) ?? duration
      : [...bounds].reverse().find((bound) => bound < timeRef.current - 0.3) ?? 0;
    setPlaybackTime(target);
  };

  const toggleModel = (id: ModelRun["id"]) => {
    if (!evaluation && availableMap[id] === false) return;
    setActive((value) => {
      const next = { ...value, [id]: !value[id] };
      saveModelActive(next);
      return next;
    });
  };

  const setEval = <K extends keyof EvalConfig>(key: K, value: EvalConfig[K]) => {
    setEvalCfg((cfg) => ({ ...cfg, [key]: value }));
  };

  const setParam = <K extends keyof ParamMap[ModelRun["id"]]>(id: ModelRun["id"], key: K, value: ParamMap[ModelRun["id"]][K]) => {
    setParams((all) => ({ ...all, [id]: { ...all[id], [key]: value } }));
  };

  // Every tab is driven by `nav` alone — no other tab's state gates access to it.
  const handleNav = useCallback((next: Nav) => {
    setNav(next);
    if (next !== "dashboard") {
      setPlaying(false);
      audioRef.current?.pause();
    }
  }, []);
  return (
    <div className="app">
      {catalogError && <div className="catalog-error">{catalogError}</div>}
      {audioNotice && <div className="catalog-error" onClick={() => setAudioNotice(null)}>{audioNotice}</div>}
      {playableUrl && (
        <audio
          ref={audioRef}
          src={playableUrl}
          preload="metadata"
          onLoadStart={() => setAudioReady(false)}
          onLoadedMetadata={() => setAudioReady(true)}
          onCanPlay={() => setAudioReady(true)}
          onEnded={() => setPlaying(false)}
          onError={() => {
            // Non-fatal: a transient stream hiccup under load shouldn't crash
            // playback. Re-arm the element for the next interaction; the play
            // retry path recovers an active session. No blocking alert.
            console.error("Audio playback error:", audioRef.current?.error);
            audioRef.current?.load();
          }}
        />
      )}
      <TopBar nav={nav} onNav={handleNav} />
      <ModelStatusStrip catalog={catalog} status={modelStatus} />

      {nav === "projects" && <ProjectsView projects={projects} onOpenProject={openProject} onNew={startNewUpload} />}
      {nav === "settings" && (
        <SettingsView
          models={catalogModels}
          available={availableMap}
          active={active}
          evalCfg={evalCfg}
          params={params}
          projects={projects}
          streamInline={streamInline}
          glow={glow}
          feed={feed}
          onToggleModel={toggleModel}
          onEval={setEval}
          onParam={setParam}
          onStreamInline={() => setStreamInline((value) => !value)}
          onGlow={() => setGlow((value) => !value)}
          onFeed={() => setFeed((value) => !value)}
          onClear={() => { persistProjects([]); setCurrent(null); }}
        />
      )}

      {nav === "dashboard" && workflow === "loading" && (
        <main className="center-screen">
          <div className="upload-title">
            <span className="spinner" />
            <span>Loading recording…</span>
          </div>
        </main>
      )}

      {nav === "upload" && workflow === "idle" && (
        <EmptyDashboard
          models={catalogModels}
          available={availableMap}
          active={active}
          projects={projects}
          onFile={startRealUpload}
          onLoadDemo={startFlow}
          onSettings={() => handleNav("settings")}
          onOpenProject={openProject}
          onProjects={() => handleNav("projects")}
        />
      )}
      {nav === "upload" && workflow === "uploading" && <UploadingScreen fileName={uploadName} pct={uploadPct} />}
      {nav === "upload" && workflow === "processing" && evaluation && (
        <ProcessingScreen models={evaluation.models} durationSec={evaluation.durationSec} now={nowTick} onRetry={handleRetry} />
      )}
      {nav === "dashboard" && workflow !== "loading" && (
        <main className="dashboard">
          <ConfigBar models={dashboardModels} active={active} onToggle={toggleModel} />
          <section className="file-bar">
            <div>
              File: <strong>{fileName}</strong>
              <span className="mono">Duration {durationText}</span>
              {evaluation?.uploadMs != null && <span className="mono">Upload {evaluation.uploadMs} ms</span>}
            </div>
            <div className="zoom-control">
              <span className="eyebrow">Zoom</span>
              <button type="button" onClick={() => setZoom((value) => Math.max(1, value - 1))}>-</button>
              <span className="mono">{zoom}×</span>
              <button type="button" onClick={() => setZoom((value) => Math.min(50, value + 1))}>+</button>
            </div>
          </section>
          <section className="studio-layout">
            <Studio
              models={dashboardModels}
              active={active}
              modelStatus={modelStatus}
              duration={duration}
              wavePeaks={wavePeaks}
              glow={glow}
              time={time}
              zoom={zoom}
              playing={playing}
              now={nowTick}
              onRetry={evaluation && !isDemo ? handleRetry : undefined}
              onToggle={() => {
                if (timeRef.current >= duration) setPlaybackTime(0);
                // Side effects stay out of the state updater: React double-invokes
                // updaters in StrictMode, which would fire two play/retry chains.
                const next = !playing;
                setPlaying(next);
                const audio = audioRef.current;
                if (audio && playableUrl) {
                  if (next) {
                    const target = timeRef.current >= duration ? 0 : timeRef.current;
                    if (timeRef.current >= duration) audio.currentTime = 0;
                    attemptPlay(target);
                  } else {
                    audio.pause();
                  }
                }
              }}
              onStep={step}
              onSeek={(next) => setPlaybackTime(next)}
              playheadRef={(node) => { playheadRef.current = node; }}
              waveFillRef={(node) => { waveFillRef.current = node; }}
              clockRef={(node) => { clockRef.current = node; }}
              miniFillRef={(node) => { miniFillRef.current = node; }}
              scrollRef={(node) => { scrollRef.current = node; }}
              innerRef={(node) => { innerRef.current = node; }}
            />
            <Insights
              models={dashboardModels}
              active={active}
              time={time}
              events={events}
              feed={feed}
              clockRef={(node) => { clock2Ref.current = node; }}
            />
          </section>
        </main>
      )}
    </div>
  );
}
