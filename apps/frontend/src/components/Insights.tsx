import { useEffect, useRef } from "react";
import type { ActiveMap, EventItem, ModelRun } from "../types";
import { SPEAKER_COLORS } from "../data";
import { fmt, hexA } from "../utils";
import { currentSpeakerRows } from "./Studio";

interface InsightsProps {
  models: ModelRun[];
  active: ActiveMap;
  time: number;
  events: EventItem[];
  feed: boolean;
  clockRef: (node: HTMLElement | null) => void;
}

export function Insights({ models, active, time, events, feed, clockRef }: InsightsProps) {
  const rows = currentSpeakerRows(models, active, time);
  const feedItems = events.filter((event) => active[event.id] && event.t <= time + 0.01).slice(-14).reverse();
  const newestKey = feedItems[0] ? `${feedItems[0].id}-${feedItems[0].t}` : "";
  const feedRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    feedRef.current?.scrollTo({ top: 0 });
  }, [newestKey]);

  return (
    <aside className="insights">
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

      {feed && (
        <>
          <div className="eyebrow feed-title">Event feed</div>
          <div className="event-feed" ref={feedRef}>
            {feedItems.length ? feedItems.map((event, index) => {
              const color = SPEAKER_COLORS[event.spk % SPEAKER_COLORS.length];
              return (
                <div key={`${event.id}-${event.t}-${index}`} className="feed-row">
                  <span className="mono">{fmt(event.t)}</span>
                  <i style={{ background: color }} />
                  <span><b>{event.short}</b> · Speaker {event.spk + 1}</span>
                  {event.overlap && <em>OVLP</em>}
                </div>
              );
            }) : (
              <div className="feed-empty">Press play to stream events</div>
            )}
          </div>
        </>
      )}
    </aside>
  );
}
