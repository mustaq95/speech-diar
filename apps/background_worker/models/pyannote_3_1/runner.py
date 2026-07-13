"""PyAnnote 3.1-specific execution code.

Runs pyannote.audio's speaker-diarization-3.1 pipeline (pyannote.audio>=4.0)
and returns its native output; only ``adapter.py`` is allowed to understand
that shape.

This is the sibling model to `pyannote` (which runs the newer
speaker-diarization-community-1 release): both pipelines' embedding stage is
WeSpeaker-based (community-1's model card cites the WeSpeaker toolkit
directly), so this slot compares the earlier 3.1 release -- an older
WeSpeaker checkpoint (pyannote/wespeaker-voxceleb-resnet34-LM) and its own
clustering -- against community-1's newer embedding and VBx clustering.

An earlier version of this runner ran the wenet-e2e/wespeaker toolkit
directly (Silero VAD + ResNet34-LM embeddings) with a custom UMAP-HDBSCAN
clustering step swapped in for wespeaker's default spectral clustering
(which badly under-detected speaker count via its eigengap heuristic). That
custom clustering folded every HDBSCAN noise point into its own singleton
speaker cluster -- harmless on short clips, but on a 32-minute recording
(~2,500 embedding subsegments) it produced 1168 "speakers". Rather than
reimplement 3D-Speaker's reassign/merge post-processing a second time, this
slot now runs 3.1's tuned pipeline instead, which found 5 speakers on the
same file (`pyannote` found 8).

pyannote.audio and torch are heavy, GPU-oriented deps (`uv sync --extra
models`, CUDA hosts only) and are NOT installed by default. They are
imported lazily inside `_load_pipeline()`, never at module scope, so that
importing this module (which happens whenever `REGISTRY` is built, i.e. in
every API/worker process and the test suite) never requires them.

Configuration (`.env`, read via `packages/config/settings.py`):
  HUGGINGFACE_TOKEN — required; both speaker-diarization-3.1 and its
    segmentation-3.0 dependency are gated models. Accept their conditions at
    https://huggingface.co/pyannote/speaker-diarization-3.1 and
    https://huggingface.co/pyannote/segmentation-3.0 first.
  DIARIZATION_DEVICE — "cpu" or "cuda".
"""

from typing import Any

from packages.config.settings import get_settings

from ..base_model import ModelRunner

# PyAnnote 3.1 native shape (serialized): [{"start", "end", "label": "SPEAKER_00"}]
Pyannote31RawOutput = list[dict[str, Any]]

MODEL_ID = "pyannote/speaker-diarization-3.1"

_pipeline = None


def _load_pipeline():
    global _pipeline
    if _pipeline is None:
        import torch
        from pyannote.audio import Pipeline

        settings = get_settings()
        pipeline = Pipeline.from_pretrained(MODEL_ID, token=settings.huggingface_token)
        pipeline.to(torch.device(settings.diarization_device))
        _pipeline = pipeline
    return _pipeline


def _load_waveform(audio_path: str):
    """Read a WAV file into the (channel, time) float32 tensor pyannote wants.

    Every upload on this platform is WAV (see the upload UI's "WAV · up to 2
    hours" constraint), so the stdlib `wave` module is enough here -- no
    extra dependency needed. This bypasses pyannote's default file-loading
    path (torchcodec), which requires system FFmpeg shared libraries
    (libavutil.so etc.) that aren't installed on this host; passing a
    pre-loaded {"waveform", "sample_rate"} dict is pyannote's own documented
    alternative (see pyannote.audio.core.io.Audio) and needs no FFmpeg.
    """
    import wave

    import numpy as np
    import torch

    with wave.open(audio_path, "rb") as wav_file:
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        raw = wav_file.readframes(wav_file.getnframes())

    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sample_width)
    if dtype is None:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    samples = np.frombuffer(raw, dtype=dtype)
    if sample_width == 1:
        samples = (samples.astype(np.float32) - 128.0) / 128.0
    else:
        samples = samples.astype(np.float32) / float(2 ** (8 * sample_width - 1))

    waveform = np.ascontiguousarray(samples.reshape(-1, num_channels).T)
    return torch.from_numpy(waveform), sample_rate


class Pyannote31Runner(ModelRunner[Pyannote31RawOutput]):
    model_id = "pyannote-3-1"

    def run(self, audio_path: str, params: dict[str, Any] | None = None) -> Pyannote31RawOutput:
        pipeline = _load_pipeline()
        waveform, sample_rate = _load_waveform(audio_path)
        output = pipeline({"waveform": waveform, "sample_rate": sample_rate})
        annotation = getattr(output, "speaker_diarization", output)
        return [
            {"start": segment.start, "end": segment.end, "label": label}
            for segment, _, label in annotation.itertracks(yield_label=True)
        ]
