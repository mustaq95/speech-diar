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
