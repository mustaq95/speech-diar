"""Audio helpers shared by the transcription engines.

The platform's canonical stored shape is 16 kHz mono 16-bit PCM WAV: uploads
are transcoded to it in `apps/backend_api/routers/upload.py`. Recordings
ingested before that was true can still be any shape, so anything depending on
the canonical form has to check rather than assume.

`apps/background_worker/models/moss_transcribe/runner.py` has its own private
copy of this check, written before this module existed. It is deliberately
left alone (reworking a working model's runner is outside the live-speech
change); if the canonical shape ever changes, both must move.
"""

import io
import os
import shutil
import subprocess
import tempfile
import wave
from contextlib import suppress

SAMPLE_RATE = 16000


def is_canonical_wav(path: str) -> bool:
    """True when the file is already 16 kHz mono 16-bit PCM.

    Reads only the header off the path — a stored WAV can be hundreds of MB
    and must never be pulled into memory just to be inspected.
    """
    with suppress(Exception):
        with wave.open(path, "rb") as wav:
            return (
                wav.getsampwidth() == 2
                and wav.getcomptype() == "NONE"
                and wav.getframerate() == SAMPLE_RATE
                and wav.getnchannels() == 1
            )
    return False


def ensure_canonical_wav(path: str) -> tuple[str, str | None]:
    """Guarantee a 16 kHz mono 16-bit PCM WAV, transcoding via ffmpeg if needed.

    Returns `(usable_path, cleanup_path)`; `cleanup_path` is the temp file the
    caller must unlink afterwards, or None when the original was already
    canonical and is used as-is.

    Raises RuntimeError when the audio is neither canonical nor convertible,
    rather than passing non-canonical bytes downstream: Hamsa would interpret
    them at the declared sample rate (pitch-shifted audio, garbage transcript)
    and vLLM would reject an oversized file with a bare 400. A loud failure
    names the real problem.
    """
    if is_canonical_wav(path):
        return path, None
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"{path} is not 16kHz mono 16-bit PCM and ffmpeg is not installed to convert it"
        )
    fd, out = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    # Same args as upload.py's _transcode_to_wav_file (the canonical-shape
    # definition), reading a path instead of a stdin pipe.
    proc = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", path, "-vn", "-ar", str(SAMPLE_RATE), "-ac", "1", "-acodec", "pcm_s16le", out],
        capture_output=True,
    )
    if proc.returncode != 0:
        with suppress(OSError):
            os.unlink(out)
        raise RuntimeError(f"ffmpeg could not convert {path}: {proc.stderr.decode()[:300]}")
    return out, out


def split_wav_fixed(path: str, segment_seconds: float) -> list[tuple[bytes, float, float]]:
    """Cut a canonical WAV into fixed-length pieces: `(wav_bytes, start, end)`.

    Boundaries are exactly on the interval and are never moved to land in a
    silence. That is deliberate, not a shortcut: choosing cut points by
    inspecting the audio is this code guessing on the engine's behalf, and every
    guess it gets wrong would surface as a word error attributed to the engine.
    A word straddling a boundary does split, and that cost belongs to the
    transport being measured.

    Pure `wave` arithmetic, no ffmpeg subprocess: the caller has already run the
    file through `ensure_canonical_wav`, so a segment is just a frame-range
    re-wrapped with its own header.

    Returns a single whole-file piece when the audio already fits, so callers
    have one code path regardless of length.
    """
    with wave.open(path, "rb") as wav:
        channels, width, rate = wav.getnchannels(), wav.getsampwidth(), wav.getframerate()
        total_frames = wav.getnframes()
        frames = wav.readframes(total_frames)

    duration = total_frames / float(rate)
    if duration <= segment_seconds:
        with open(path, "rb") as fh:
            return [(fh.read(), 0.0, duration)]

    bytes_per_frame = channels * width
    frames_per_segment = max(1, int(segment_seconds * rate))
    pieces: list[tuple[bytes, float, float]] = []
    for start_frame in range(0, total_frames, frames_per_segment):
        end_frame = min(start_frame + frames_per_segment, total_frames)
        chunk = frames[start_frame * bytes_per_frame : end_frame * bytes_per_frame]
        buffer = io.BytesIO()
        # Each piece carries a real WAV header of its own: the gateway is handed
        # a standalone file, not a headerless byte range it would misread.
        with wave.open(buffer, "wb") as out:
            out.setnchannels(channels)
            out.setsampwidth(width)
            out.setframerate(rate)
            out.writeframes(chunk)
        pieces.append((buffer.getvalue(), start_frame / float(rate), end_frame / float(rate)))
    return pieces


def read_pcm16(path: str) -> bytes:
    """Raw little-endian 16-bit PCM frames from a canonical WAV — the payload
    format Hamsa's WebSocket expects. The caller has already ensured the file
    is canonical."""
    with wave.open(path, "rb") as wav:
        return wav.readframes(wav.getnframes())
