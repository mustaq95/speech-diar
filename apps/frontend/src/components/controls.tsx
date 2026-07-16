import { useState } from "react";
import type { ReactNode } from "react";
import type { Metric } from "../types";

interface ToggleProps {
  on: boolean;
  onClick: () => void;
  label: string;
  disabled?: boolean;
}

export function Toggle({ on, onClick, label, disabled = false }: ToggleProps) {
  return (
    <button
      className={`toggle ${on ? "is-on" : ""}`}
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-pressed={on}
      aria-label={label}
    >
      <span />
    </button>
  );
}

interface SelectProps<T extends string> {
  value: T;
  options: Array<T | { value: T; label: string }>;
  onChange: (value: T) => void;
  label: string;
  disabled?: boolean;
}

export function SelectControl<T extends string>({ value, options, onChange, label, disabled = false }: SelectProps<T>) {
  return (
    <select
      className="select-control"
      value={value}
      onChange={(event) => onChange(event.target.value as T)}
      aria-label={label}
      disabled={disabled}
    >
      {options.map((option) => {
        const value = typeof option === "string" ? option : option.value;
        const text = typeof option === "string" ? option : option.label;
        return (
          <option key={value} value={value}>
            {text}
          </option>
        );
      })}
    </select>
  );
}

interface SliderProps {
  value: number;
  min: number;
  max: number;
  step: number;
  label: string;
  format: (value: number) => string;
  onChange: (value: number) => void;
  disabled?: boolean;
}

export function SliderControl({ value, min, max, step, label, format, onChange, disabled = false }: SliderProps) {
  return (
    <div className="slider-control">
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        aria-label={label}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <span>{format(value)}</span>
    </div>
  );
}

export function SegmentedMetric({
  value,
  onChange,
  disabled = false,
}: {
  value: Metric;
  onChange: (value: Metric) => void;
  disabled?: boolean;
}) {
  const options: Metric[] = ["DER", "JER", "WDER"];
  return (
    <div className="segmented" role="tablist" aria-label="Primary metric">
      {options.map((option) => (
        <button
          key={option}
          type="button"
          disabled={disabled}
          className={value === option ? "is-active" : ""}
          onClick={() => onChange(option)}
          role="tab"
          aria-selected={value === option}
        >
          {option}
        </button>
      ))}
    </div>
  );
}

export function Stepper({
  value,
  min,
  max,
  label,
  onChange,
  disabled = false,
}: {
  value: number;
  min: number;
  max: number;
  label: string;
  onChange: (value: number) => void;
  disabled?: boolean;
}) {
  const clamp = (next: number) => Math.max(min, Math.min(max, next));
  return (
    <div className="stepper" aria-label={label}>
      <button type="button" disabled={disabled} onClick={() => onChange(clamp(value - 1))} aria-label={`${label} decrease`}>
        -
      </button>
      <span>{value}</span>
      <button type="button" disabled={disabled} onClick={() => onChange(clamp(value + 1))} aria-label={`${label} increase`}>
        +
      </button>
    </div>
  );
}

export function WaveGlyph({ colors, height = 18 }: { colors: string[]; height?: number }) {
  const bars = [9, 16, 12, 19, 14];
  return (
    <span className="wave-glyph" style={{ height }}>
      {bars.map((bar, index) => (
        <span key={index} style={{ height: bar, background: colors[index % colors.length] }} />
      ))}
    </span>
  );
}

export function IconButton({
  children,
  label,
  onClick,
  primary = false,
  disabled = false,
  className = "",
}: {
  children: ReactNode;
  label: string;
  onClick: () => void;
  primary?: boolean;
  disabled?: boolean;
  className?: string;
}) {
  return (
    <button className={`icon-btn ${primary ? "is-primary" : ""} ${className}`} type="button" aria-label={label} title={label} onClick={onClick} disabled={disabled}>
      {children}
    </button>
  );
}

/** Re-run one model, with a confirm step so a stray click can't discard a
 * result. State is per-instance: each row asks its own question, and nothing
 * has to be lifted into the timeline or the processing grid. */
export function RetryControls({
  modelId,
  onRetry,
  className = "",
}: {
  modelId: string;
  onRetry: (modelId: string) => Promise<void>;
  className?: string;
}) {
  const [confirming, setConfirming] = useState(false);
  // Blocks a second click from enqueuing a duplicate job for the same run
  // before the first response lands.
  const [pending, setPending] = useState(false);

  const confirm = async () => {
    setConfirming(false);
    setPending(true);
    try {
      await onRetry(modelId);
    } catch (error) {
      console.error("Retry failed:", error);
      window.alert(`Could not re-run this model: ${(error as Error).message}`);
    } finally {
      setPending(false);
    }
  };

  return (
    <div className={`retry-controls ${className}`}>
      {confirming ? (
        <>
          <IconButton label="Confirm re-run" className="is-confirm" onClick={() => void confirm()}>✓</IconButton>
          <IconButton label="Cancel re-run" className="is-cancel" onClick={() => setConfirming(false)}>✕</IconButton>
        </>
      ) : (
        <IconButton label="Re-run this model" onClick={() => setConfirming(true)} disabled={pending}>↻</IconButton>
      )}
    </div>
  );
}
