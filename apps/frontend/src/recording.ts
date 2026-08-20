/**
 * Microphone capture for the read-aloud comparison.
 *
 * One capture, three consumers, because the two engines have different native
 * transports and both must hear the SAME audio:
 *
 *   1. a continuous PCM stream, forwarded frame by frame to a streaming engine;
 *   2. fixed-interval WAV chunks, POSTed one at a time to a request/response
 *      engine that cannot stream;
 *   3. amplitude peaks, for the live waveform.
 *
 * Chunk boundaries fall exactly on the interval. Nothing here inspects the audio
 * to pick a quieter cut point: choosing boundaries by looking at the signal is
 * the frontend guessing on the engine's behalf, and every guess it got wrong
 * would surface as a word error attributed to the engine. A word straddling a
 * boundary does split, and that cost belongs to the transport being measured.
 *
 * No sample rate, interval or buffer size is written here — they arrive from
 * `GET /config`, so moving environments stays an .env-only change.
 */

/** 16-bit mono PCM plus the peak of the block, as captured. */
export interface CapturedBlock {
  pcm: Int16Array;
  /** 0..1 peak amplitude, for the waveform. */
  peak: number;
}

export interface RecorderOptions {
  /** Target PCM rate, from GET /config. The browser may capture at another rate;
   * blocks are resampled to this one so both engines get identical audio. */
  sampleRate: number;
  /** Seconds of audio per chunk for the chunked transport. */
  chunkIntervalSec: number;
  /** Samples per callback, from GET /config. Sets how promptly PCM leaves for a
   * streaming engine and how soon a completed chunk is emitted: at 16 kHz, 4096
   * samples is 256 ms of built-in delay, 1024 is 64 ms. */
  blockSamples: number;
  /** Every captured block, for the stream transport and the waveform. */
  onBlock: (block: CapturedBlock) => void;
  /** One complete chunk, already a valid standalone WAV. */
  onChunk: (wav: Blob, index: number) => void;
}

export interface Recorder {
  stop: () => Promise<Blob>;
  /** Seconds captured so far, from the sample count — not a wall clock, so it
   * cannot drift away from the audio actually recorded. */
  elapsedSec: () => number;
}

/** Wrap 16-bit PCM in a real WAV header.
 *
 * Every chunk is a standalone file, not a headerless byte range: the gateway is
 * handed something it can decode on its own, rather than something it would
 * misread at the wrong rate. */
export function encodeWav(pcm: Int16Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + pcm.length * 2);
  const view = new DataView(buffer);
  const writeAscii = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };
  writeAscii(0, "RIFF");
  view.setUint32(4, 36 + pcm.length * 2, true);
  writeAscii(8, "WAVE");
  writeAscii(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeAscii(36, "data");
  view.setUint32(40, pcm.length * 2, true);
  new Int16Array(buffer, 44).set(pcm);
  return new Blob([buffer], { type: "audio/wav" });
}

/** Float32 [-1,1] to 16-bit PCM, clamped rather than wrapped.
 *
 * Clamping matters: a sample slightly over 1.0 wrapped by a bare cast becomes a
 * loud negative spike, which reads to a VAD as a transient and can split a
 * segment where no pause existed. */
export function toPcm16(samples: Float32Array): Int16Array {
  const pcm = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    pcm[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
  }
  return pcm;
}

/** Linear resample to the target rate.
 *
 * Only runs when the browser refuses the requested capture rate. Linear
 * interpolation is crude for audio, but the alternative here is worse than
 * crude: sending 48 kHz samples to an engine told they are 16 kHz produces
 * pitch-shifted audio and a garbage transcript. */
export function resample(samples: Float32Array, from: number, to: number): Float32Array {
  if (from === to) return samples;
  const ratio = from / to;
  const out = new Float32Array(Math.floor(samples.length / ratio));
  for (let i = 0; i < out.length; i += 1) {
    const position = i * ratio;
    const index = Math.floor(position);
    const next = Math.min(index + 1, samples.length - 1);
    const fraction = position - index;
    out[i] = samples[index] * (1 - fraction) + samples[next] * fraction;
  }
  return out;
}

function concatPcm(blocks: Int16Array[]): Int16Array {
  const total = blocks.reduce((sum, block) => sum + block.length, 0);
  const merged = new Int16Array(total);
  let offset = 0;
  for (const block of blocks) {
    merged.set(block, offset);
    offset += block.length;
  }
  return merged;
}

/**
 * Start capturing. Rejects when the browser denies microphone access, which is a
 * real outcome to surface rather than a state to retry silently.
 */
export async function startRecording(options: RecorderOptions): Promise<Recorder> {
  const { sampleRate, chunkIntervalSec, blockSamples, onBlock, onChunk } = options;

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      sampleRate,
      // Left on: this is speech evaluation on a laptop mic, and comparing two
      // engines on cleaner audio compares the engines rather than the room.
      // Identical for both engines either way, which is what has to hold.
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  const context = new AudioContext({ sampleRate });
  const source = context.createMediaStreamSource(stream);
  // ScriptProcessor rather than AudioWorklet: a worklet needs a separately
  // served module file, and this callback does nothing but copy a buffer.
  // Deprecated but universally available, and the work is trivial enough that
  // running it on the audio thread is not a risk.
  //
  // The block size is the latency floor for everything downstream, which is why
  // it comes from config rather than being left at the 4096 default.
  const processor = context.createScriptProcessor(blockSamples, 1, 1);

  const allPcm: Int16Array[] = [];
  let chunkPcm: Int16Array[] = [];
  let chunkSamples = 0;
  let totalSamples = 0;
  let chunkIndex = 0;
  let stopped = false;
  const samplesPerChunk = Math.round(chunkIntervalSec * sampleRate);

  processor.onaudioprocess = (event) => {
    if (stopped) return;
    const input = event.inputBuffer.getChannelData(0);
    const resampled = resample(input, event.inputBuffer.sampleRate, sampleRate);
    const pcm = toPcm16(resampled);

    let peak = 0;
    for (let i = 0; i < resampled.length; i += 1) {
      const magnitude = Math.abs(resampled[i]);
      if (magnitude > peak) peak = magnitude;
    }

    allPcm.push(pcm);
    chunkPcm.push(pcm);
    chunkSamples += pcm.length;
    totalSamples += pcm.length;
    onBlock({ pcm, peak });

    // Emit whole chunks on the interval. The remainder stays buffered for the
    // next one rather than being padded out, so no silence is invented.
    while (chunkSamples >= samplesPerChunk) {
      const merged = concatPcm(chunkPcm);
      const chunk = merged.slice(0, samplesPerChunk);
      const remainder = merged.slice(samplesPerChunk);
      onChunk(encodeWav(chunk, sampleRate), chunkIndex);
      chunkIndex += 1;
      chunkPcm = remainder.length ? [remainder] : [];
      chunkSamples = remainder.length;
    }
  };

  source.connect(processor);
  // A ScriptProcessor only runs while connected to the graph. Routed through a
  // silent gain node so it never plays the microphone back into the room.
  const silence = context.createGain();
  silence.gain.value = 0;
  processor.connect(silence);
  silence.connect(context.destination);

  return {
    elapsedSec: () => totalSamples / sampleRate,
    stop: async () => {
      stopped = true;
      // The tail below the chunk interval is still sent: it is real audio, and
      // dropping it would lose the end of what someone read.
      if (chunkSamples > 0) {
        onChunk(encodeWav(concatPcm(chunkPcm), sampleRate), chunkIndex);
      }
      processor.disconnect();
      silence.disconnect();
      source.disconnect();
      for (const track of stream.getTracks()) track.stop();
      await context.close();
      return encodeWav(concatPcm(allPcm), sampleRate);
    },
  };
}
