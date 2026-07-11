"""Thin FastAPI wrapper around 3D-Speaker's ASR-free diarization pipeline
(VAD -> CAM++ speaker embeddings -> clustering, no transcription step at
all).

Execution only: loads the pipeline once at startup, then for each request
runs Diarization3Dspeaker(audio_path, speaker_num=None) -- never a fixed or
oracle speaker count, per this platform's "nothing fabricated" rule -- and
formats the returned segments as RTTM text, handed back untouched. No RTTM
parsing happens here -- that is the worker adapter's job
(apps/background_worker/models/speaker3d_clustering/adapter.py), per this
platform's runner/adapter split.

Overlap detection (an optional extra stage using pyannote/segmentation-3.0,
which needs a HuggingFace gated-repo token) is left off: this container
never sets an HF token, so include_overlap stays False.

The default CAM++ checkpoint 3D-Speaker loads
(iic/speech_campplus_sv_zh_en_16k-common_advanced) was trained on
Chinese/English speaker data. The VAD/CAM++/clustering approach is
acoustic-only (no phonetic/ASR content), so it is expected to generalize
across languages better than an ASR-dependent pipeline would, but that has
not been benchmarked for Arabic specifically -- treat cross-lingual
performance as unverified until validated against real audio.

No turnkey NIM or vendor image exists for this pipeline, so this container
clones modelscope/3D-Speaker directly at a pinned commit -- see
../3d_speaker_clustering_up.sh and docker-compose.3d-speaker-clustering.yml.
"""

import shutil
import tempfile
import threading
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from wave import open as wave_open

from fastapi import FastAPI, HTTPException, UploadFile

_lock = threading.Lock()
_pipeline = None


def _wav_duration_sec(path: Path) -> float | None:
    with suppress(Exception):
        with wave_open(str(path), "rb") as wav:
            return wav.getnframes() / wav.getframerate()
    return None


def _to_rttm(wav_id: str, segments: list[list[float]]) -> str:
    """Format 3D-Speaker's native [[start, end, speaker_id], ...] output as
    RTTM text -- the same line format infer_diarization.py's own
    save_diar_output() writes for --out_type rttm, so downstream parsing
    matches the upstream CLI tool's documented output exactly."""
    line = "SPEAKER {} 0 {:.3f} {:.3f} <NA> <NA> {} <NA> <NA>\n"
    return "".join(
        line.format(wav_id, start, end - start, int(speaker_id))
        for start, end, speaker_id in segments
    )


def _load_pipeline():
    import torch
    from speakerlab.bin.infer_diarization import Diarization3Dspeaker

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return Diarization3Dspeaker(device=device, include_overlap=False, speaker_num=None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    _pipeline = _load_pipeline()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health/ready")
def health_ready() -> dict[str, str]:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready"}


@app.post("/diarize")
async def diarize(file: UploadFile) -> dict[str, object]:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    work_dir = Path(tempfile.mkdtemp(prefix="3d-speaker-clustering-req-"))
    try:
        audio_path = work_dir / (file.filename or "audio.wav")
        audio_path.write_bytes(await file.read())

        with _lock:
            segments = _pipeline(str(audio_path), speaker_num=None)

        return {
            "audio_duration_sec": _wav_duration_sec(audio_path),
            "rttm": _to_rttm(audio_path.stem, segments),
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
