import type { ModelRun } from "./types";
import { activeSpeakers } from "./utils";

export function speakerSignature(models: ModelRun[], t: number): string {
  return models.map((model) => `${model.id}:${activeSpeakers(model, t).join(".")}`).join("|");
}

/** Peak amplitude (0..1) per bar, computed from real decoded audio — never a synthetic shape. */
export function computeWavePeaks(buffer: AudioBuffer, bars = 210): number[] {
  const samplesPerBar = Math.max(1, Math.floor(buffer.length / bars));
  const peaks: number[] = [];
  for (let i = 0; i < bars; i++) {
    const start = i * samplesPerBar;
    const end = Math.min(buffer.length, start + samplesPerBar);
    let max = 0;
    for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
      const data = buffer.getChannelData(ch);
      for (let j = start; j < end; j++) {
        const v = Math.abs(data[j]);
        if (v > max) max = v;
      }
    }
    peaks.push(Math.min(1, max));
  }
  return peaks;
}

/** Fetch audio from `url`, decode it, and return real per-bar peak amplitudes. */
export async function decodeWaveformPeaks(url: string, bars = 210): Promise<number[]> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not fetch audio for waveform (status ${response.status})`);
  const arrayBuffer = await response.arrayBuffer();
  const ctx = new AudioContext();
  try {
    const buffer = await ctx.decodeAudioData(arrayBuffer);
    return computeWavePeaks(buffer, bars);
  } finally {
    await ctx.close();
  }
}
