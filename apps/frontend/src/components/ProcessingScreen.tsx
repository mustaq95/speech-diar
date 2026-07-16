import type { ModelRun } from "../types";
import { isInFlight, modelTimingLabel } from "../timing";
import { RetryControls } from "./controls";

interface ProcessingScreenProps {
  models: ModelRun[];
  durationSec: number;
  now: number;
  onRetry: (modelId: string) => Promise<void>;
}

export function ProcessingScreen({ models, durationSec, now, onRetry }: ProcessingScreenProps) {
  return (
    <main className="center-screen">
      <section className="processing-wrap">
        <div className="processing-heading">
          <h1>Running diarization models</h1>
          <p>Each model processes the recording independently. Results appear as each one finishes — this is its real status, not a simulated progress bar.</p>
        </div>
        <div className="processing-grid">
          {models.map((model) => {
            const busy = isInFlight(model);
            const failed = model.status === "failed";
            return (
              <article key={model.id} className={`processing-card ${!busy ? "is-done" : ""} ${failed ? "is-failed" : ""}`}>
                <div className="processing-card-head">
                  <div>
                    {busy ? <span className="spinner" /> : <span className={failed ? "fail-mark" : "done-mark"}>{failed ? "✕" : "✓"}</span>}
                    <strong>{model.name}</strong>
                  </div>
                  {!busy && <RetryControls modelId={model.id} onRetry={onRetry} />}
                </div>
                <p>{modelTimingLabel(model, durationSec, now)}</p>
              </article>
            );
          })}
        </div>
      </section>
    </main>
  );
}
