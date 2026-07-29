import type { ReactNode } from "react";
import type { ActiveMap, AvailableMap, EvalConfig, Metric, ModelRun, ParamMap, Project } from "../types";
import { FALLBACK_PARAMS } from "../adapters";
import { SPEAKER_COLORS } from "../data";
import { SegmentedMetric, SelectControl, SliderControl, Stepper, Toggle, WaveGlyph } from "./controls";

interface SettingsViewProps {
  models: ModelRun[];
  available: AvailableMap;
  active: ActiveMap;
  evalCfg: EvalConfig;
  params: ParamMap;
  projects: Project[];
  streamInline: boolean;
  glow: boolean;
  feed: boolean;
  onToggleModel: (id: ModelRun["id"]) => void;
  onEval: <K extends keyof EvalConfig>(key: K, value: EvalConfig[K]) => void;
  onParam: <K extends keyof ParamMap[ModelRun["id"]]>(id: ModelRun["id"], key: K, value: ParamMap[ModelRun["id"]][K]) => void;
  onStreamInline: () => void;
  onGlow: () => void;
  onFeed: () => void;
  onClear: () => void;
}

function Section({ title, sub, children }: { title: string; sub?: string; children: ReactNode }) {
  return (
    <section className="settings-section">
      <div className="settings-section-head">
        <h2>{title}</h2>
        {sub && <p>{sub}</p>}
      </div>
      {children}
    </section>
  );
}

function Row({ label, desc, children }: { label: string; desc?: string; children: ReactNode }) {
  return (
    <div className="settings-row">
      <div>
        <strong>{label}</strong>
        {desc && <p>{desc}</p>}
      </div>
      {children}
    </div>
  );
}

export function SettingsView({
  models,
  available,
  active,
  evalCfg,
  params,
  projects,
  streamInline,
  glow,
  feed,
  onToggleModel,
  onEval,
  onParam,
  onStreamInline,
  onGlow,
  onFeed,
  onClear,
}: SettingsViewProps) {
  return (
    <main className="page settings-page">
      <section className="view-head">
        <div>
          <h1>Settings</h1>
          <p>Configure the models, how their runs are scored against each other, and the workspace.</p>
        </div>
      </section>

      <Section
        title="Evaluation & scoring"
        sub="How model outputs are compared and where discrepancies are measured. Not wired up yet — coming soon."
      >
        <Row label="Reference / ground truth" desc="What every model is scored against.">
          <SelectControl
            label="Reference"
            value={evalCfg.reference}
            options={["Uploaded RTTM", "Human transcript", "Use a model as reference", "None - unsupervised compare"]}
            onChange={(value) => onEval("reference", value)}
            disabled
          />
        </Row>
        <Row label="Baseline model" desc="Anchors the side-by-side diff and agreement view.">
          <SelectControl
            label="Baseline model"
            value={evalCfg.baseline}
            options={models.map((model) => ({ value: model.id, label: model.short }))}
            onChange={(value) => onEval("baseline", value)}
            disabled
          />
        </Row>
        <Row label="Scoring collar" desc="Forgiveness window around each speaker boundary before it counts as an error.">
          <SliderControl
            label="Scoring collar"
            value={evalCfg.collar}
            min={0}
            max={500}
            step={10}
            format={(value) => `${value} ms`}
            onChange={(value) => onEval("collar", value)}
            disabled
          />
        </Row>
        <Row label="Ignore overlapping speech in scoring" desc="Exclude overlap regions from the error rate.">
          <Toggle on={evalCfg.ignoreOverlap} onClick={() => onEval("ignoreOverlap", !evalCfg.ignoreOverlap)} label="Ignore overlap" disabled />
        </Row>
        <Row label="Primary metric" desc="Headline number shown per model on the comparison.">
          <SegmentedMetric value={evalCfg.metric} onChange={(value: Metric) => onEval("metric", value)} disabled />
        </Row>
      </Section>

      <Section title="Models" sub="Which models run on each upload. Unavailable engines are listed but can't be enabled yet.">
        {models.map((model) => {
          const isAvailable = available[model.id] ?? false;
          const on = active[model.id] && isAvailable;
          // A model toggled on for an old recording that predates it has no
          // params entry (params derive from the evaluation's models); fall
          // back so the disabled placeholder body renders instead of crashing.
          const p = params[model.id] ?? FALLBACK_PARAMS;
          return (
            <article key={model.id} className={`model-card ${isAvailable ? "" : "is-unavailable"}`}>
              <div className="model-card-head">
                <WaveGlyph colors={SPEAKER_COLORS} />
                <span>
                  <strong>{model.name}</strong>
                  <small>{model.description}{!isAvailable && " · Not implemented yet"}</small>
                </span>
                <Toggle on={on} onClick={() => onToggleModel(model.id)} label={`${model.name} enabled`} disabled={!isAvailable} />
              </div>
              {on && (
                <div className="model-card-body">
                  <p className="settings-note">Per-model parameters below are not wired up yet — every model runs on its own defaults.</p>
                  <Row label="Min speakers"><Stepper label={`${model.short} min speakers`} value={p.min} min={1} max={p.max} onChange={(value) => onParam(model.id, "min", value)} disabled /></Row>
                  <Row label="Max speakers"><Stepper label={`${model.short} max speakers`} value={p.max} min={p.min} max={12} onChange={(value) => onParam(model.id, "max", value)} disabled /></Row>
                  <Row label="Clustering threshold"><SliderControl label={`${model.short} threshold`} value={p.thr} min={0.3} max={0.95} step={0.01} format={(value) => value.toFixed(2)} onChange={(value) => onParam(model.id, "thr", value)} disabled /></Row>
                  <Row label="Overlap detection" desc="Allow this model to report two speakers at once."><Toggle on={p.ovl} onClick={() => onParam(model.id, "ovl", !p.ovl)} label={`${model.short} overlap detection`} disabled /></Row>
                </div>
              )}
            </article>
          );
        })}
      </Section>

      <Section title="Processing">
        <Row label="Open dashboard as soon as audio uploads" desc="Land on the timeline right away and let each model's band update as its real status changes.">
          <Toggle on={streamInline} onClick={onStreamInline} label="Stream results inline" />
        </Row>
      </Section>

      <Section title="Display">
        <Row label="Neon glow on active speech" desc="Highlight currently-detected segments on each track.">
          <Toggle on={glow} onClick={onGlow} label="Neon glow" />
        </Row>
        <Row label="Live speech transcript" desc="Show the word-level transcript in the insights panel, highlighting each word as it is spoken. The transcript itself is always produced; this only controls the panel.">
          <Toggle on={feed} onClick={onFeed} label="Live speech" />
        </Row>
      </Section>

      <Section title="Speaker palette" sub="A color always maps to the same speaker index across every model.">
        <div className="palette-row">
          {SPEAKER_COLORS.map((color, index) => (
            <span key={color}><i style={{ background: color }} /> Speaker {index + 1}</span>
          ))}
        </div>
      </Section>

      <Section title="Data">
        <Row label="Saved recordings" desc={`${projects.length} recording${projects.length === 1 ? "" : "s"} stored locally on this device.`}>
          <button className="danger-btn" type="button" onClick={onClear}>Clear history</button>
        </Row>
      </Section>
    </main>
  );
}
