"""VibeVoice-ASR as an ASR engine -- local vLLM container, so audio stays on this
host and the engine is `offline`.

Delegates to `models/vibevoice/runner.py` for the same reason `moss/runner.py`
does: that module already owns the container call (base64 `input_audio` parts
over chat-completions, the local/remote split via VIBEVOICE_BASEURL, its TLS
switch). Duplicating it would duplicate every one of those decisions.

The native payload is returned untouched and is the same object the diarization
adapter reads; this package's `adapter.py` takes the text half.

Probed against the local container on 2026-09-01 with a 61 s code-switched clip:

  * Native shape: `{"audio_duration_sec": float, "segments": [{"Start": float,
    "End": float, "Speaker": int, "Content": str}, ...]}`.
  * **`Speaker` is not always present.** The segment carrying
    `"[Unintelligible Speech]"` had no `Speaker` key at all, so every read of it
    must be a `.get()`. An unguarded `[...]` here is the same class of bug as the
    ffprobe `KeyError` that once discarded a whole clip's metadata.
  * It genuinely code-switches (Arabic in Arabic script, English in Latin, inside
    one `Content` string) and emits non-speech markers `[Music]` and
    `[Unintelligible Speech]`. Measured WER 0.593.
  * 54.8 s of wall clock for 61 s of audio -- roughly 0.9x realtime, far slower
    than every hosted engine here. Live mode IS offered, and the chunks it is fed
    are byte-identical to every other engine's, so the input is fair. Its live
    LATENCY is not comparable: at ~0.9x realtime it cannot keep pace with a
    speaker and will fall progressively further behind across a session. Read that
    figure as a property of this pipeline, with its transport label attached.

There is no language parameter anywhere in this path: it is a chat-completions
call, so there is nothing to force and nothing to configure.

Configuration: VIBEVOICE_URL / VIBEVOICE_TIMEOUT_SEC (plus the VIBEVOICE_BASEURL
remote path), all already defined for the diarization model. No new settings.
"""

import os
import tempfile
from contextlib import suppress
from typing import Any

from apps.background_worker.models.vibevoice.runner import VibeVoiceRunner

#: `{"audio_duration_sec": float, "segments": [{"Start", "End", "Speaker"?,
#: "Content"}]}` -- note `Speaker` is optional, see above.
VibeVoiceRawOutput = dict[str, Any]

ENGINE = "vibevoice"

_runner = VibeVoiceRunner()


def run(audio_path: str) -> VibeVoiceRawOutput:
    """Transcribe `audio_path` with one whole-file call to the local container."""
    return _runner.run(audio_path)

def _run_on_bytes(wav_bytes: bytes, suffix: str = ".wav"):
    """Run the delegated runner over a blob of WAV bytes.

    The runner this package delegates to takes a PATH, because it is shared with
    the diarization side where the audio is always already a file. The live chunk
    route hands over bytes, so the chunk is spilled to a temporary file and
    removed afterwards. `delete=False` plus an explicit unlink, rather than a
    context manager, because the runner opens the path by name and some platforms
    will not reopen a still-open NamedTemporaryFile.
    """
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        handle.write(wav_bytes)
        handle.flush()
        handle.close()
        return _runner.run(handle.name)
    finally:
        with suppress(OSError):
            os.unlink(handle.name)


def transcribe_bytes(wav_bytes: bytes, filename: str = "chunk.wav") -> VibeVoiceRawOutput:
    """Transcribe one already-short WAV -- the live chunk path."""
    return _run_on_bytes(wav_bytes)
