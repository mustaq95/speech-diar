import type { ActiveMap, ModelRun } from "../types";
import type { TranscriptionMode, TranscriptRun } from "../types/diarization";
import type { RuntimeConfig } from "../adapters";
import { SPEAKER_COLORS } from "../data";
import { fmt, hexA } from "../utils";
import { LiveSpeech } from "./LiveSpeech";
import { currentSpeakerRows } from "./Studio";

interface InsightsProps {
  models: ModelRun[];
  active: ActiveMap;
  time: number;
  /** Every engine's transcript for this recording; the panel picks by mode. */
  transcripts: TranscriptRun[];
  /** Settings toggle for the live-speech panel; the transcript itself always runs. */
  feed: boolean;
  /** Null until the first `GET /config` lands. Carries each mode's availability. */
  runtimeConfig: RuntimeConfig | null;
  onRunTranscript?: (mode: TranscriptionMode) => Promise<void>;
  wordSyncRef: (sync: ((time: number) => void) | null) => void;
  clockRef: (node: HTMLElement | null) => void;
}

export function Insights({ models, active, time, transcripts, feed, runtimeConfig, onRunTranscript, wordSyncRef, clockRef }: InsightsProps) {
  const rows = currentSpeakerRows(models, active, time);

  return (
    <aside className="insights">
      {feed && (
        <LiveSpeech
          transcripts={transcripts}
          models={models}
          active={active}
          runtimeConfig={runtimeConfig}
          onRun={onRunTranscript}
          wordSyncRef={wordSyncRef}
        />
      )}

      <div className="insights-head">
        <h2>Live Insights</h2>
        <span ref={clockRef} className="mono accent">{fmt(time)}</span>
      </div>
      <div className="eyebrow">Detected now</div>
      <div className="detected-list">
        {rows.map(({ model, speakers, busy, failed }) => (
          <div key={model.id} className="detected-row">
            <div>
              {busy && <span className="spinner small" />}
              <strong>{model.short}</strong>
              {!busy && !failed && speakers.length >= 2 && <b className="overlap-badge">OVERLAP</b>}
              {!busy && !failed && speakers.length === 1 && <span className="single-label">single</span>}
            </div>
            {busy ? (
              <em>{model.status === "queued" ? "queued..." : "running..."}</em>
            ) : failed ? (
              <em>failed</em>
            ) : speakers.length ? (
              <span className="speaker-tags">
                {speakers.map((speaker) => {
                  const color = SPEAKER_COLORS[speaker % SPEAKER_COLORS.length];
                  return (
                    <span key={speaker} style={{ borderColor: hexA(color, 0.55), background: hexA(color, 0.16) }}>
                      <i style={{ background: color }} /> Speaker {speaker + 1}
                    </span>
                  );
                })}
              </span>
            ) : (
              <em>silence</em>
            )}
          </div>
        ))}
      </div>
    </aside>
  );
}
