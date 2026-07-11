import type { ModelRun } from "./types";
import { activeSpeakers } from "./utils";

export function speakerSignature(models: ModelRun[], t: number): string {
  return models.map((model) => `${model.id}:${activeSpeakers(model, t).join(".")}`).join("|");
}

interface WavPcmInfo {
  numChannels: number;
  bitsPerSample: number;
  dataOffset: number;
  dataLength: number;
}

/** Locates the `fmt `/`data` sub-chunks in a WAV file's own bytes — no
 * decoding, just enough header parsing to know where the PCM samples are. */
function parseWavPcmInfo(buffer: ArrayBuffer): WavPcmInfo {
  const view = new DataView(buffer);
  let offset = 12; // skip "RIFF" + size (4) + "WAVE" (4)
  let numChannels = 1;
  let bitsPerSample = 16;
  let dataOffset = -1;
  let dataLength = 0;

  while (offset + 8 <= buffer.byteLength) {
    const chunkId = String.fromCharCode(view.getUint8(offset), view.getUint8(offset + 1), view.getUint8(offset + 2), view.getUint8(offset + 3));
    const declaredSize = view.getUint32(offset + 4, true);
    const chunkDataStart = offset + 8;

    if (chunkId === "fmt ") {
      numChannels = view.getUint16(chunkDataStart + 2, true);
      bitsPerSample = view.getUint16(chunkDataStart + 14, true);
    } else if (chunkId === "data") {
      dataOffset = chunkDataStart;
      // Some encoders (streamed/live recordings) write a placeholder size
      // larger than what's actually in the buffer -- cap it at the real bytes.
      dataLength = Math.min(declaredSize, buffer.byteLength - chunkDataStart);
      break;
    }
    offset = chunkDataStart + declaredSize + (declaredSize % 2); // chunks are word-aligned
  }

  if (dataOffset < 0) throw new Error("Not a valid WAV file (no data chunk found)");
  return { numChannels, bitsPerSample, dataOffset, dataLength };
}

/** Peak amplitude (0..1) per bar, scanned directly from the WAV's own PCM
 * bytes — never a synthetic shape, and no full-buffer float decode (which
 * doubles memory use and stalls the main thread on long recordings). */
export function computePcmWavePeaks(buffer: ArrayBuffer, bars = 210): number[] {
  const { numChannels, bitsPerSample, dataOffset, dataLength } = parseWavPcmInfo(buffer);
  if (bitsPerSample !== 16) return Array.from({ length: bars }, () => 0); // only 16-bit PCM WAVs are produced by this platform's uploads

  const view = new DataView(buffer);
  const bytesPerFrame = numChannels * 2;
  const totalFrames = Math.floor(dataLength / bytesPerFrame);
  const framesPerBar = Math.max(1, Math.floor(totalFrames / bars));
  const peaks: number[] = [];

  for (let bar = 0; bar < bars; bar++) {
    const startFrame = bar * framesPerBar;
    const endFrame = Math.min(totalFrames, startFrame + framesPerBar);
    let max = 0;
    for (let frame = startFrame; frame < endFrame; frame++) {
      const frameOffset = dataOffset + frame * bytesPerFrame;
      for (let ch = 0; ch < numChannels; ch++) {
        const v = Math.abs(view.getInt16(frameOffset + ch * 2, true)) / 32768;
        if (v > max) max = v;
      }
    }
    peaks.push(Math.min(1, max));
  }
  return peaks;
}

/** Fetch audio from `url` and return real per-bar peak amplitudes, scanned
 * straight from its PCM bytes (no AudioContext, no full Float32 decode). */
export async function decodeWaveformPeaks(url: string, bars = 210): Promise<number[]> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not fetch audio for waveform (status ${response.status})`);
  const arrayBuffer = await response.arrayBuffer();
  return computePcmWavePeaks(arrayBuffer, bars);
}
