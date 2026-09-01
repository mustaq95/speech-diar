"""MOSS-Transcribe-Diarize as an ASR engine -- local vLLM container, so audio
stays on this host and the engine is `offline`.

**This does not re-implement the container call.** It delegates to
`models/moss_transcribe/runner.py`, which already owns every measured constant
for talking to this pod (the `max_completion_tokens` spelling vLLM actually
honours, why `response_format=json` rather than `verbose_json`, why no custom
`prompt`). Two copies of that call would be two places to keep those findings
correct.

The native payload is the SAME object the diarization side receives, and it is
returned untouched. Two adapters now read it, each for the half it owns: the
diarization adapter takes the speaker turns, this package's `adapter.py` takes
the text. That is the runner/adapter rule holding, not a break from it -- an
adapter is still the only code that understands the shape.

**No `language` is sent, and that is measured.** Probed against this container on
2026-09-01 with a 61 s code-switched clip: omitted, `language=ar` and
`language=en` returned BYTE-IDENTICAL text (sha256 f7ac86aeac87337a, 943 chars),
while `language=auto` is rejected HTTP 400 against a 57-code list. vLLM validates
the field; the model ignores it.

**What this engine does to code-switched audio is worth knowing before reading
its score.** It does not transcribe English -- it TRANSLATES it into Arabic
("the data integration ... went live last Tuesday" came back as Arabic prose),
and one probe leaked a stray CJK token into an Arabic sentence. Measured WER
0.814 on the clip above, the weakest of the seven engines compared. That is the
model's real behaviour on this material and is reported, not corrected for.

Live mode is offered and is cheap here -- 4.1 s of wall clock for 61 s of audio,
so a 3 s chunk comes back well inside its own interval. Read its live numbers
knowing what the model is FOR, though: this is a joint ASR + diarization +
timestamping model built for one pass over long audio, so a 3 s window gives it
no context to diarize and its speaker labels there mean much less than they do in
batch.

Configuration: MOSS_TRANSCRIBE_URL / MOSS_TRANSCRIBE_TIMEOUT_SEC, both already
defined for the diarization model. This engine adds no new settings.
"""

import os
import tempfile
from contextlib import suppress
from typing import Any

from apps.background_worker.models.moss_transcribe.runner import MossTranscribeRunner

#: The native payload of the shared runner: `{"text": str, "usage": {...}}`,
#: where `text` carries `[start][Snn] ... [end]` markers inline.
MossRawOutput = dict[str, Any]

ENGINE = "moss-transcribe"

_runner = MossTranscribeRunner()


def run(audio_path: str) -> MossRawOutput:
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


def transcribe_bytes(wav_bytes: bytes, filename: str = "chunk.wav") -> MossRawOutput:
    """Transcribe one already-short WAV -- the live chunk path."""
    return _run_on_bytes(wav_bytes)
