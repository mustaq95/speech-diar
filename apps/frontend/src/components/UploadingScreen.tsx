interface UploadingScreenProps {
  fileName: string;
  pct: number;
}

export function UploadingScreen({ fileName, pct }: UploadingScreenProps) {
  const rounded = Math.round(pct);
  return (
    <main className="center-screen">
      <section className="upload-card" aria-live="polite">
        <div className="upload-title">
          <span className="spinner" />
          <span>Uploading recording...</span>
        </div>
        <div className="mono muted">{fileName}</div>
        <div className="progress-track">
          <span style={{ width: `${rounded}%` }} />
        </div>
        <div className="progress-meta">
          <span>Transfer</span>
          <span>{rounded}%</span>
        </div>
      </section>
    </main>
  );
}
