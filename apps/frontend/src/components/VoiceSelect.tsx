import { useEffect, useRef, useState } from "react";

/**
 * The voice picker for one TTS engine.
 *
 * Everything it renders comes from `.env` by way of GET /config: the voice
 * names, how many there are, and the engine's synthesis settings. Neither
 * gateway exposes a list-voices endpoint, so there is NO per-voice metadata
 * anywhere in this system — no gender, no locale, no per-voice sample rate.
 * The subtitle carries engine-level settings, which are true of every voice
 * this engine offers; presenting them as properties of the selected voice
 * would be inventing data.
 *
 * Rendered even when an engine has a single voice. A control that disappears
 * at N=1 made the two engine cards look like different features.
 */
interface VoiceSelectProps {
  voices: string[];
  value: string;
  onChange: (voice: string) => void;
  /** Engine-level settings shown under the name, joined for display here. */
  params: Record<string, string>;
  /** Voices that already have a stored clip. Read off the stored rows, so the
   * marker cannot drift from what actually exists. */
  synthesized?: string[];
  disabled?: boolean;
  /** Labels the listbox for screen readers, e.g. "TryHamsa TTS voice". */
  label: string;
}

export function VoiceSelect({ voices, value, onChange, params, synthesized, disabled, label }: VoiceSelectProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  // One listener pair for both dismissal routes. `pointerdown` rather than
  // `click` so the menu closes on press, before a click elsewhere resolves.
  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  const subtitle = Object.values(params).filter(Boolean).join(" · ");

  const step = (delta: number) => {
    const index = voices.indexOf(value);
    const next = voices[(index + delta + voices.length) % voices.length];
    if (next) onChange(next);
  };

  return (
    <div className="voice-select" ref={rootRef}>
      <button
        type="button"
        className="voice-trigger"
        onClick={() => setOpen((previous) => !previous)}
        onKeyDown={(event) => {
          if (event.key === "ArrowDown") {
            event.preventDefault();
            if (open) step(1);
            else setOpen(true);
          }
          if (event.key === "ArrowUp") { event.preventDefault(); step(-1); }
        }}
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={label}
      >
        <span className="voice-avatar" aria-hidden="true">{value.slice(0, 1).toUpperCase()}</span>
        <span className="voice-lines">
          <b>{value}</b>
          {subtitle && <small className="mono">{subtitle}</small>}
        </span>
        <span className="voice-caret" aria-hidden="true">▾</span>
      </button>

      {open && (
        <ul className="voice-menu" role="listbox" aria-label={label}>
          {voices.map((voice) => (
            <li key={voice}>
              <button
                type="button"
                role="option"
                aria-selected={voice === value}
                className={`voice-option${voice === value ? " is-active" : ""}`}
                onClick={() => { onChange(voice); setOpen(false); }}
              >
                <span className="voice-avatar" aria-hidden="true">{voice.slice(0, 1).toUpperCase()}</span>
                <span className="voice-lines"><b>{voice}</b></span>
                {synthesized?.includes(voice) && (
                  <span className="voice-has-clip" title="Already synthesized" aria-label="already synthesized" />
                )}
                {voice === value && <span className="voice-check" aria-hidden="true">✓</span>}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
