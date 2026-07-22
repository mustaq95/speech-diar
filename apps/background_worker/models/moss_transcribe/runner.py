"""MOSS-Transcribe-Diarize (OpenMOSS, Apache-2.0) -- a 0.9B end-to-end model
performing ASR, speaker diarization and timestamping jointly in a single
autoregressive pass over up to ~90 minutes of audio.

Execution code only; returns the NATIVE payload (the model's own generated
transcript string, including the text this platform's contract has no field
for) untouched -- see `adapter.py` for the only code allowed to understand
that shape.

Container: the stock vllm/vllm-openai image, no custom build -- MOSS is
registered in vLLM upstream. GPU-supervisor-managed (residency cap, cold
start, idle unload -- see apps/background_worker/supervisor/). See
deploy/moss-transcribe/.

Why the transcription endpoint and not chat-completions: vLLM's model class
sets `supports_transcription_only = True`, so /v1/chat/completions is not
available for this model at all.

Why `response_format=json` and not `verbose_json`: the same class sets
`supports_segment_timestamp = False`, so vLLM returns no parsed segment list
-- only `{"text": ..., "usage": ...}`. The diarization lives entirely inside
that generated string. (Upstream's README does document a `verbose_json` with
parsed `segments`, but that is its SGLang Omni backend, not vLLM.)

Configuration (`.env`, read via `packages/config/settings.py`):
  MOSS_TRANSCRIBE_URL -- base URL of the container's vLLM server
    (POST {url}/v1/audio/transcriptions)
  MOSS_TRANSCRIBE_TIMEOUT_SEC -- request timeout
"""

import os
import shutil
import subprocess
import tempfile
from contextlib import suppress
from typing import Any
from wave import open as wave_open

import httpx

from packages.config.settings import get_settings

from ..base_model import ModelRunner

MossTranscribeRawOutput = dict[str, Any]


def _wav_duration_sec(path: str) -> float | None:
    with suppress(Exception):
        with wave_open(path, "rb") as wav:
            return wav.getnframes() / wav.getframerate()
    return None


def _is_canonical(path: str) -> bool:
    """True when the file is already 16 kHz mono 16-bit PCM -- the canonical
    shape the platform stores. Mirrors `_is_canonical_wav` in
    apps/backend_api/routers/upload.py, but reads the header off a path: the
    stored WAV can be hundreds of MB, so it must never be loaded into memory."""
    with suppress(Exception):
        with wave_open(path, "rb") as wav:
            return (
                wav.getsampwidth() == 2
                and wav.getcomptype() == "NONE"
                and wav.getframerate() == 16000
                and wav.getnchannels() == 1
            )
    return False


def _ensure_canonical(path: str) -> tuple[str, str | None]:
    """Guarantee the bytes we POST are canonical 16 kHz mono 16-bit.

    MOSS is the only model that ships the stored bytes verbatim to a size-gated
    endpoint (vLLM caps uploads at VLLM_MAX_AUDIO_CLIP_FILESIZE_MB, sized for
    16 kHz mono). A recording stored before uploads were canonicalized can be
    44.1 kHz stereo -- ~5.5x larger -- and trips that gate. Other engines
    resample internally, so they never hit it.

    Returns (send_path, cleanup_path). cleanup_path is the temp file to unlink
    after the request, or None when the original is used as-is.
    """
    if _is_canonical(path):
        return path, None
    if shutil.which("ffmpeg") is None:
        # Stripped environment: keep today's behavior rather than hard-failing.
        # Worker hosts run ffmpeg (upload depends on it), so this is rare.
        return path, None
    fd, out = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    # ffmpeg args match apps/backend_api/routers/upload.py:_transcode_to_wav_file
    # (the canonical-shape definition); here we read the file path directly
    # instead of a stdin pipe. Duplicated in the two places -- if the canonical
    # rate/layout ever changes, both must move.
    proc = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", path, "-vn", "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", out],
        capture_output=True,
    )
    if proc.returncode != 0:
        with suppress(OSError):
            os.unlink(out)
        return path, None
    return out, out


class MossTranscribeRunner(ModelRunner[MossTranscribeRawOutput]):
    model_id = "moss-transcribe"

    def run(self, audio_path: str, params: dict[str, Any] | None = None) -> MossTranscribeRawOutput:
        settings = get_settings()
        send_path, cleanup = _ensure_canonical(audio_path)
        try:
            with open(send_path, "rb") as f:
                response = httpx.post(
                    f"{settings.moss_transcribe_url}/v1/audio/transcriptions",
                    files={"file": (os.path.basename(send_path), f, "audio/wav")},
                    data={
                        # Matches --served-model-name in
                        # deploy/moss-transcribe/docker-compose.moss-transcribe.yml,
                        # so repointing MOSS_TRANSCRIBE_MODEL_ID does not break this.
                        "model": "moss-transcribe",
                        "response_format": "json",
                        "temperature": "0",
                        # vLLM's spelling, and load-bearing. Upstream's README
                        # documents `max_new_tokens` -- that is SGLang's parameter
                        # name, which vLLM ignores. Leaving this OFF does not mean
                        # "unlimited": vLLM falls back to the model's own
                        # generation_config.json, which pins max_tokens=5120, and
                        # every long transcript would silently stop partway through.
                        # vLLM clamps this against the context window rather than
                        # rejecting it (get_max_tokens() returns
                        # min(max_model_len - input_len, this, ...)), so asking for
                        # the full MOSS_TRANSCRIBE_MAX_MODEL_LEN just means "as much
                        # as the window allows after the audio" -- which is what a
                        # long file needs, since its audio can consume most of the
                        # 131,072 window on its own (12.5 tokens per audio second).
                        "max_completion_tokens": "131072",
                        # No `prompt`: vLLM's built-in default for this model is
                        # byte-identical to the Chinese instruction MOSS was
                        # trained against (see its DEFAULT_MOSS_TRANSCRIBE_DIARIZE
                        # _PROMPT). Overriding it with an English one degrades the
                        # output format.
                    },
                    timeout=settings.moss_transcribe_timeout_sec,
                )
            # Not raise_for_status(): httpx renders only "Client error '400 Bad
            # Request' for url ...", and vLLM puts the part that actually matters
            # in the body -- "Maximum file size exceeded", "Input length (N)
            # exceeds model's maximum context length (65536)". Losing that turns
            # a self-explanatory failure into a `docker logs` expedition; it
            # already did once. The message ends up in EvaluationResult.error and
            # is shown on the model's row.
            if response.status_code >= 400:
                raise RuntimeError(
                    f"moss-transcribe {response.status_code}: {response.text[:500]}"
                )
            return {
                "text": response.json()["text"],
                "audio_duration_sec": _wav_duration_sec(audio_path),
            }
        finally:
            if cleanup:
                with suppress(OSError):
                    os.unlink(cleanup)
