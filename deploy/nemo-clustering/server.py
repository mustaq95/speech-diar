"""Thin FastAPI wrapper around NeMo's ClusteringDiarizer.

Execution only: loads the VAD (MarbleNet) + speaker embedding (TitaNet)
models once at startup, then for each request runs the cascaded pipeline
(VAD -> multi-scale embeddings -> spectral clustering) and hands back the
RTTM NeMo produced, untouched. No RTTM parsing happens here -- that is the
worker adapter's job (apps/background_worker/models/nemo_clustering/adapter.py),
per this platform's runner/adapter split.

There is no turnkey NIM for this pipeline (NIM only ships
`diarizer=sortformer`), so this container is built directly from the NGC
NeMo image -- see ../nemo_clustering_up.sh and docker-compose.nemo-clustering.yml.
"""

import audioop
import shutil
import tempfile
import threading
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from wave import open as wave_open

from fastapi import FastAPI, HTTPException, UploadFile
from omegaconf import OmegaConf

CONFIG_PATH = Path(__file__).parent / "diar_infer.yaml"

_lock = threading.Lock()
_diarizer = None


def _wav_duration_sec(path: Path) -> float | None:
    with suppress(Exception):
        with wave_open(str(path), "rb") as wav:
            return wav.getnframes() / wav.getframerate()
    return None


def _repair_audio_for_diarizer(path: Path) -> None:
    """The diarizer config (diar_infer.yaml) assumes mono input, and a WAV
    header whose declared frame count doesn't match the actual bytes on disk
    (a streamed/live-recorded placeholder size) is the likely cause of a 500
    on a real recording -- e.g. a ~1-billion-frame declared duration instead
    of the real ~32 minutes. Rewrites the file in place with a correct
    header and, if needed, a mono downmix; skipped entirely when the file is
    already good, which is the common case.
    """
    with open(path, "rb") as raw, wave_open(raw) as wav:
        n_channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        framerate = wav.getframerate()
        header_frames = wav.getnframes()
        data_start = raw.tell()

    bytes_per_frame = n_channels * sampwidth
    file_size = path.stat().st_size
    max_frames_in_buffer = max(0, file_size - data_start) // bytes_per_frame if bytes_per_frame else 0
    header_ok = header_frames == max_frames_in_buffer

    if n_channels <= 1 and header_ok:
        return  # already good -- skip rewriting the file

    with open(path, "rb") as raw, wave_open(raw) as wav:
        frames = wav.readframes(wav.getnframes())

    if n_channels > 1:
        frames = audioop.tomono(frames, sampwidth, 0.5, 0.5)
        n_channels = 1

    with wave_open(str(path), "wb") as fixed:
        fixed.setnchannels(n_channels)
        fixed.setsampwidth(sampwidth)
        fixed.setframerate(framerate)
        fixed.writeframes(frames)


def _load_diarizer():
    import os

    from nemo.collections.asr.models import ClusteringDiarizer

    cfg = OmegaConf.load(CONFIG_PATH)
    max_speakers = os.environ.get("NEMO_CLUSTERING_MAX_SPEAKERS")
    if max_speakers:
        cfg.diarizer.clustering.parameters.max_num_speakers = int(max_speakers)

    # __init__ only loads the VAD/speaker models; manifest_filepath/out_dir
    # are read later by diarize(), which server.py sets per request.
    cfg.diarizer.manifest_filepath = ""
    cfg.diarizer.out_dir = tempfile.mkdtemp(prefix="nemo-clustering-init-")

    return ClusteringDiarizer(cfg=cfg).to(cfg.device)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _diarizer
    _diarizer = _load_diarizer()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health/ready")
def health_ready() -> dict[str, str]:
    if _diarizer is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready"}


@app.post("/diarize")
async def diarize(file: UploadFile) -> dict[str, object]:
    if _diarizer is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    work_dir = Path(tempfile.mkdtemp(prefix="nemo-clustering-req-"))
    try:
        audio_path = work_dir / (file.filename or "audio.wav")
        audio_path.write_bytes(await file.read())
        _repair_audio_for_diarizer(audio_path)

        out_dir = work_dir / "out"
        with _lock:
            _diarizer._cfg.diarizer.out_dir = str(out_dir)
            _diarizer.diarize(paths2audio_files=[str(audio_path)])

        uttid = audio_path.stem
        rttm_path = out_dir / "pred_rttms" / f"{uttid}.rttm"
        rttm_text = rttm_path.read_text() if rttm_path.exists() else ""

        return {"audio_duration_sec": _wav_duration_sec(audio_path), "rttm": rttm_text}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
