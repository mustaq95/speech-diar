import { useRef } from "react";
import type { ActiveMap, ModelRun } from "../types";
import { SPEAKER_COLORS } from "../data";
import { activeSpeakers, fmt, hexA, overlapsFor } from "../utils";
import { isInFlight, modelTimingLabel } from "../timing";
import { IconButton } from "./controls";

/** Percent along the timeline, safe when there's no duration yet (nothing loaded). */
function pct(value: number, duration: number): number {
  return duration > 0 ? (value / duration) * 100 : 0;
}

/** Fixed vertical gap between one model's row/band and the next, regardless of speaker count. */
const ROW_GAP = 8;

/** Minimum height for every model's row/band and gutter label, regardless of speaker count
 * (speaker lanes are sized as a percentage of the band's actual height, so more speakers
 * make the lanes thinner instead of growing the box off-screen). Bands still flex-grow to
 * equally fill any extra vertical space when fewer models are shown. */
const BAND_H = 104;

/** Minimum horizontal density (pixels per second of audio) at zoom = 1. Keeps
 * long recordings from being squeezed into one viewport width where every
 * segment is a sliver — past this density the timeline scrolls instead of
 * compressing further. Short clips still fill the viewport exactly, since
 * this is only a floor (see the `max()` in the width below). */
const MIN_PX_PER_SEC = 6;

interface StudioProps {
  models: ModelRun[];
  active: ActiveMap;
  duration: number;
  wavePeaks: number[];
  glow: boolean;
  time: number;
  zoom: number;
  playing: boolean;
  now: number;
  onToggle: () => void;
  onStep: (dir: -1 | 1) => void;
  onSeek: (time: number) => void;
  playheadRef: (node: HTMLDivElement | null) => void;
  waveFillRef: (node: HTMLDivElement | null) => void;
  clockRef: (node: HTMLElement | null) => void;
  miniFillRef: (node: HTMLDivElement | null) => void;
  scrollRef: (node: HTMLDivElement | null) => void;
  innerRef: (node: HTMLDivElement | null) => void;
}

function Transport({
  time,
  duration,
  playing,
  onToggle,
  onStep,
  clockRef,
  miniFillRef,
}: Pick<StudioProps, "time" | "duration" | "playing" | "onToggle" | "onStep" | "clockRef" | "miniFillRef">) {
  const disabled = duration <= 0;
  return (
    <div className="transport">
      <div className="transport-buttons">
        <IconButton label="Previous boundary" onClick={() => onStep(-1)} disabled={disabled}>⏮</IconButton>
        <IconButton label={playing ? "Pause" : "Play"} onClick={onToggle} primary disabled={disabled}>{playing ? "❚❚" : "▶"}</IconButton>
        <IconButton label="Next boundary" onClick={() => onStep(1)} disabled={disabled}>⏭</IconButton>
      </div>
      <div className="transport-clock">
        <strong ref={clockRef}>{fmt(time)}</strong>
        <span>/ {fmt(duration)}</span>
      </div>
      <div className="mini-progress">
        <span ref={miniFillRef} style={{ width: `${pct(time, duration)}%` }} />
      </div>
    </div>
  );
}

function tickStep(duration: number): number {
  if (duration > 600) return 60;
  if (duration > 240) return 30;
  if (duration > 120) return 15;
  if (duration > 60) return 10;
  return 5;
}

function Ruler({ duration }: Pick<StudioProps, "duration">) {
  const step = tickStep(duration);
  const ticks = [];
  for (let t = 0; t <= duration; t += step) ticks.push(t);
  // Append the exact end only if it won't collide with the last regular tick.
  const last = ticks.at(-1) ?? 0;
  if (duration - last >= step * 0.5) ticks.push(duration);
  return (
    <div className="ruler">
      {ticks.map((t, index) => (
        <span key={t} className="tick" style={{ left: `${pct(t, duration)}%` }}>
          <b className={index === ticks.length - 1 ? "is-last" : ""}>{fmt(t)}</b>
        </span>
      ))}
    </div>
  );
}

function Wave({ waveFillRef, time, duration, wavePeaks }: Pick<StudioProps, "waveFillRef" | "time" | "duration" | "wavePeaks">) {
  return (
    <div className="waveform">
      <div className="wave-bars">
        {wavePeaks.map((height, index) => <i key={index} style={{ height: `${height * 74 + 6}%` }} />)}
      </div>
      <div className="wave-fill" ref={waveFillRef} style={{ width: `${pct(time, duration)}%` }} />
    </div>
  );
}

function Band({
  model,
  glow,
  time,
  duration,
  now,
  gap,
}: {
  model: ModelRun;
  glow: boolean;
  time: number;
  duration: number;
  now: number;
  gap: boolean;
}) {
  const pad = 9;
  const numSpk = Math.max(model.numSpk, 1);
  const busy = isInFlight(model);
  const failed = model.status === "failed";

  const bandClass = `model-band${gap ? " model-row-gap" : ""}${busy ? " pending-band" : ""}${failed ? " failed-band" : ""}`;

  if (busy) {
    return (
      <div className={bandClass}>
        <span className="spinner" />
        <div>
          <p className="mono">{modelTimingLabel(model, duration, now)}</p>
        </div>
      </div>
    );
  }

  if (failed) {
    return (
      <div className={bandClass}>
        <span className="fail-mark">✕</span>
        <div>
          <p className="mono">{modelTimingLabel(model, duration, now)}</p>
        </div>
      </div>
    );
  }

  return (
    <div className={bandClass}>
      {overlapsFor(model).map((range, index) => (
        <span
          key={`overlap-${index}`}
          className="overlap-zone"
          style={{ left: `${pct(range.s, duration)}%`, width: `${pct(range.e - range.s, duration)}%` }}
        />
      ))}
      {model.segs.map((seg, index) => {
        const color = SPEAKER_COLORS[seg.spk % SPEAKER_COLORS.length];
        const on = time >= seg.s && time < seg.e;
        return (
          <span
            key={index}
            className={`segment ${on ? "is-active" : ""}`}
            title={`Speaker ${seg.spk + 1}`}
            style={{
              left: `${pct(seg.s, duration)}%`,
              width: `${pct(seg.e - seg.s, duration)}%`,
              top: `calc(${pad}px + (100% - ${pad * 2}px) * ${seg.spk / numSpk})`,
              height: `calc((100% - ${pad * 2}px) / ${numSpk} - 5px)`,
              borderColor: color,
              color: on ? "#fff" : color,
              background: hexA(color, on ? 0.42 : 0.15),
              boxShadow: on && glow ? `0 0 12px ${hexA(color, 0.6)}` : "none",
            }}
          >
            SPK {seg.spk + 1}
          </span>
        );
      })}
    </div>
  );
}

export function Studio({
  models,
  active,
  duration,
  wavePeaks,
  glow,
  time,
  zoom,
  playing,
  now,
  onToggle,
  onStep,
  onSeek,
  playheadRef,
  waveFillRef,
  clockRef,
  miniFillRef,
  scrollRef,
  innerRef,
}: StudioProps) {
  const shown = models.filter((model) => active[model.id]);
  const outerRef = useRef<HTMLDivElement | null>(null);
  const rulerH = 30;
  const waveH = 104;
  const minInnerH = rulerH + waveH + shown.length * BAND_H + ROW_GAP * Math.max(shown.length - 1, 0);

  const seek = (event: React.MouseEvent<HTMLDivElement>) => {
    const target = outerRef.current;
    if (!target) return;
    const rect = target.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / target.offsetWidth));
    onSeek(ratio * duration);
  };

  const modelRows = shown.length > 0 ? ` repeat(${shown.length}, minmax(${BAND_H}px, 1fr))` : "";

  return (
    <div className="studio-shell">
      <div
        className="studio-grid"
        style={{
          minHeight: minInnerH,
          gridTemplateRows: `${rulerH}px ${waveH}px${modelRows}`,
        }}
      >
        <div className="gutter-ruler" />
        <div className="gutter-transport">
          <Transport time={time} duration={duration} playing={playing} onToggle={onToggle} onStep={onStep} clockRef={clockRef} miniFillRef={miniFillRef} />
        </div>
        {shown.map((model, index) => (
          <div
            key={model.id}
            className={`gutter-model${index > 0 ? " model-row-gap" : ""}`}
            style={{ gridRow: 3 + index }}
          >
            <div>
              <span className="mono">{String(index + 1).padStart(2, "0")}</span>
              <strong title={model.name}>{model.short}</strong>
            </div>
            <div>
              <small>Speakers: {model.numSpk}</small>
              <span className="speaker-dots">
                {Array.from({ length: model.numSpk }).map((_, speaker) => <i key={speaker} style={{ background: SPEAKER_COLORS[speaker % SPEAKER_COLORS.length] }} />)}
              </span>
            </div>
            <small className="model-timing">{modelTimingLabel(model, duration, now)}</small>
          </div>
        ))}
        <div className="studio-scroll" ref={scrollRef}>
          <div
            className="studio-inner"
            ref={(node) => { innerRef(node); outerRef.current = node; }}
            style={{ width: `max(100%, ${duration * MIN_PX_PER_SEC * zoom}px)` }}
            onClick={seek}
          >
            <Ruler duration={duration} />
            <Wave waveFillRef={waveFillRef} time={time} duration={duration} wavePeaks={wavePeaks} />
            {shown.map((model, index) => (
              <Band key={model.id} model={model} glow={glow} time={time} duration={duration} now={now} gap={index > 0} />
            ))}
            <div ref={playheadRef} className="playhead" style={{ left: `${pct(time, duration)}%` }}>
              <span />
            </div>
          </div>
        </div>
      </div>
      {shown.length === 0 && <div className="no-models">Enable at least one model to compare tracks.</div>}
    </div>
  );
}

export function currentSpeakerRows(models: ModelRun[], active: ActiveMap, t: number) {
  return models.filter((model) => active[model.id]).map((model) => ({
    model,
    speakers: activeSpeakers(model, t),
    busy: isInFlight(model),
    failed: model.status === "failed",
  }));
}
