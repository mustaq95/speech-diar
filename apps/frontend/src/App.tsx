import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ConfigBar } from "./components/ConfigBar";
import { EmptyDashboard } from "./components/EmptyDashboard";
import { Insights } from "./components/Insights";
import { ProcessingScreen } from "./components/ProcessingScreen";
import { ProjectsView } from "./components/ProjectsView";
import { SettingsView } from "./components/SettingsView";
import { Studio } from "./components/Studio";
import { TopBar } from "./components/TopBar";
import { UploadingScreen } from "./components/UploadingScreen";
import { boundsFor, buildEvents, fmt, loadProjects, saveProjects } from "./utils";
import {
  DEMO_AUDIO_FILE_ID,
  audioStreamUrl,
  deriveActiveFromCatalog,
  deriveDefaultEval,
  deriveDefaultParams,
  fetchEvaluation,
  fetchModelCatalog,
  fetchRuntimeConfig,
  getDiarizationEvaluation,
  modelRunFromMetadata,
  patchUploadTiming,
  uploadAudio,
} from "./adapters";
import { speakerSignature, decodeWaveformPeaks } from "./playback";
import { isInFlight } from "./timing";
import type { ActiveMap, AvailableMap, DiarizationEvaluation, EvalConfig, ModelId, ModelMetadata, ModelRun, Nav, ParamMap, Project, Workflow } from "./types";

const FLAT_WAVE_PEAKS = Array.from({ length: 210 }, () => 0.3);
const DEFAULT_POLL_INTERVAL_MS = 1500;

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

  const playableUrl = evaluation && !isDemo ? audioStreamUrl(evaluation.audioFileId) : null;
  const hasInFlight = useMemo(() => models.some(isInFlight), [models]);

  const events = useMemo(() => buildEvents(models), [models]);
  const shownModels = useMemo(() => models.filter((model) => active[model.id]), [active, models]);
  const fileName = isDemo ? "Synthetic demo (no real audio)" : (current?.name ?? uploadName) || "No recording loaded";
  const durationText = fmt(duration);
  // Dashboard always shows the Studio shell; before anything is uploaded it
  // falls back to the configured model catalog so the layout looks the same.
  const dashboardModels = evaluation ? models : catalogModels;

  // Fetch the real, honest model registry and runtime config once at boot.
  useEffect(() => {
    fetchModelCatalog()
      .then((list) => setCatalog(list))
      .catch((error: Error) => setCatalogError(`Could not load model list: ${error.message}`));
    fetchRuntimeConfig()
      .then((cfg) => setPollIntervalMs(cfg.pollIntervalMs))
      .catch(() => undefined);
  }, []);

  // Seed "which models run next" from the catalog until a real evaluation exists.
  useEffect(() => {
    if (catalog.length === 0 || evaluation) return;
    setActive(deriveActiveFromCatalog(catalog));
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

  // Auto-advance from the dedicated processing screen once every model has settled.
  // Navigation is intentionally left untouched here: when "open dashboard on upload"
  // is off, the user decides when to switch to the Dashboard tab to see results.
  useEffect(() => {
    if (workflow === "processing" && evaluation && !hasInFlight) setWorkflow("idle");
  }, [workflow, evaluation, hasInFlight]);

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
  useEffect(() => {
    if (!playableUrl) {
      setWavePeaks(FLAT_WAVE_PEAKS);
      return;
    }
    let cancelled = false;
    decodeWaveformPeaks(playableUrl)
      .then((peaks) => { if (!cancelled) setWavePeaks(peaks); })
      .catch((error: Error) => {
        console.error("Waveform decode failed:", error);
        if (!cancelled) setWavePeaks(FLAT_WAVE_PEAKS);
      });
    return () => { cancelled = true; };
  }, [playableUrl]);

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
    syncDom(clamped);
    const sig = speakerSignature(shownModels, clamped);
    if (sig !== sigRef.current) {
      sigRef.current = sig;
      setSpeakerTick((tick) => tick + 1);
    }
    if (commit) setTime(clamped);
  }, [playableUrl, duration, shownModels, syncDom]);

  useEffect(() => {
    playingRef.current = playing;
  }, [playing]);

  useEffect(() => {
    let raf = 0;
    const tick = (ts: number) => {
      if (playingRef.current) {
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
        }
      } else {
        lastTsRef.current = null;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playableUrl, duration, shownModels, syncDom]);

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
    setActive((value) => ({ ...value, [id]: !value[id] }));
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
      {playableUrl && (
        <audio
          ref={audioRef}
          src={playableUrl}
          preload="auto"
          onEnded={() => setPlaying(false)}
        />
      )}
      <TopBar nav={nav} onNav={handleNav} />

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
        <ProcessingScreen models={evaluation.models} durationSec={evaluation.durationSec} now={nowTick} />
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
              <button type="button" onClick={() => setZoom((value) => Math.min(6, value + 1))}>+</button>
            </div>
          </section>
          <section className="studio-layout">
            <Studio
              models={dashboardModels}
              active={active}
              duration={duration}
              wavePeaks={wavePeaks}
              glow={glow}
              time={time}
              zoom={zoom}
              playing={playing}
              now={nowTick}
              onToggle={() => {
                if (timeRef.current >= duration) setPlaybackTime(0);
                setPlaying((value) => {
                  const next = !value;
                  const audio = audioRef.current;
                  if (audio && playableUrl) {
                    if (next) {
                      if (timeRef.current >= duration) audio.currentTime = 0;
                      void audio.play().catch((error: Error) => {
                        console.error(error);
                        window.alert(`Could not play audio: ${error.message}`);
                        setPlaying(false);
                      });
                    } else {
                      audio.pause();
                    }
                  }
                  return next;
                });
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
