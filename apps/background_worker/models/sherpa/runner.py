"""Sherpa-onnx-specific execution code.

Runs k2-fsa's sherpa-onnx offline speaker diarization (pyannote
segmentation-3.0 + 3D-Speaker ERes2Net embeddings, clustered) and returns its
native output; only ``adapter.py`` is allowed to understand that shape.

sherpa-onnx is a CPU-only, dependency-free ONNX runtime and is NOT installed
by default (`uv sync --extra models`). It is imported lazily inside
`_load_diarizer()`, never at module scope, so that importing this module
(which happens whenever `REGISTRY` is built, i.e. in every API/worker process
and the test suite) never requires it.

Configuration (`.env`, read via `packages/config/settings.py`):
  SHERPA_MODEL_DIR — directory holding the two ONNX model files, downloaded
    on first use if missing:
      segmentation.onnx — pyannote segmentation-3.0
        (https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2)
      embedding.onnx — NeMo TitaNet-small (English)
        (https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx)
        Chosen over the upstream example's zh-CN ERes2Net default because
        this platform's English test audio clustered into 4+ speakers with
        that embedding instead of the correct 2. For non-English corpora,
        drop a different embedding.onnx into SHERPA_MODEL_DIR (deleted
        first so this URL doesn't just re-download it) -- no code change
        needed.
"""

import tarfile
import tempfile
from pathlib import Path
from typing import Any

from packages.config.settings import get_settings

from ..base_model import ModelRunner

# Sherpa native shape: [{"start", "end", "speaker": int}]
SherpaRawOutput = list[dict[str, Any]]

SEGMENTATION_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
EMBEDDING_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx"

_diarizer = None


def _download(url: str, dest: Path) -> None:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        with requests.get(url, stream=True, timeout=300) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=1 << 20):
                tmp.write(chunk)
    tmp_path.rename(dest)


def _ensure_weights(model_dir: Path) -> tuple[Path, Path]:
    segmentation_path = model_dir / "segmentation.onnx"
    embedding_path = model_dir / "embedding.onnx"

    if not segmentation_path.exists():
        with tempfile.NamedTemporaryFile(suffix=".tar.bz2") as archive:
            _download(SEGMENTATION_URL, Path(archive.name))
            with tarfile.open(archive.name, "r:bz2") as tar:
                member = next(m for m in tar.getmembers() if m.name.endswith("model.onnx"))
                extracted = tar.extractfile(member)
                assert extracted is not None
                model_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=model_dir, delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                    tmp.write(extracted.read())
                tmp_path.rename(segmentation_path)

    if not embedding_path.exists():
        _download(EMBEDDING_URL, embedding_path)

    return segmentation_path, embedding_path


def _load_diarizer():
    global _diarizer
    if _diarizer is None:
        import sherpa_onnx

        settings = get_settings()
        model_dir = Path(settings.sherpa_model_dir).expanduser()
        segmentation_path, embedding_path = _ensure_weights(model_dir)

        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(segmentation_path)
                )
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(embedding_path)),
            clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1),
        )
        _diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
    return _diarizer


def _load_audio(audio_path: str, target_sample_rate: int):
    """Read a WAV file into the mono float32 array sherpa-onnx wants.

    Every upload on this platform is WAV (see the upload UI's "WAV · up to 2
    hours" constraint), so the stdlib `wave` module is enough here -- no
    extra dependency needed.
    """
    import wave

    import numpy as np

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

    samples = samples.reshape(-1, num_channels)
    mono = samples[:, 0]

    if sample_rate != target_sample_rate:
        duration = len(mono) / sample_rate
        num_target_samples = int(round(duration * target_sample_rate))
        source_times = np.arange(len(mono)) / sample_rate
        target_times = np.arange(num_target_samples) / target_sample_rate
        mono = np.interp(target_times, source_times, mono).astype(np.float32)

    return mono


class SherpaRunner(ModelRunner[SherpaRawOutput]):
    model_id = "sherpa"

    def run(self, audio_path: str, params: dict[str, Any] | None = None) -> SherpaRawOutput:
        diarizer = _load_diarizer()
        audio = _load_audio(audio_path, diarizer.sample_rate)
        result = diarizer.process(audio).sort_by_start_time()
        return [{"start": seg.start, "end": seg.end, "speaker": seg.speaker} for seg in result]
