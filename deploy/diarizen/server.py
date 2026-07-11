"""Thin FastAPI wrapper around DiariZen's diarization pipeline (WavLM-Large +
Conformer local end-to-end diarization, followed by pyannote-3.1-style
global clustering across the whole file).

Execution only: loads the pretrained pipeline once at startup, then for each
request runs it against the uploaded audio and hands back the RTTM text the
pipeline itself produces (`Annotation.to_rttm()`), untouched. No RTTM parsing
happens here -- that is the worker adapter's job
(apps/background_worker/models/diarizen/adapter.py), per this platform's
runner/adapter split.

Model: BUT-FIT/diarizen-wavlm-large-s80-md-v2 -- the pretrained repo's own
config.toml caps simultaneous speaker overlap at 4 (max_speakers_per_chunk)
but the clustering stage's max_speakers is 20, so total distinct speakers in
a file is not capped anywhere near 8. Never pass a fixed or oracle speaker
count, per this platform's "nothing fabricated" rule.

Trained on AMI, AISHELL-4, AliMeeting, NOTSOFAR-1, MSDWild, DIHARD3, RAMC,
and VoxConverse -- English/Mandarin-dominant, no Arabic. Diarization
clusters voice embeddings rather than linguistic content, so it is expected
to generalize across languages better than an ASR-dependent pipeline would,
but that has not been benchmarked for Arabic specifically -- treat
cross-lingual performance as unverified until validated against real audio.

Pretrained weights are CC BY-NC 4.0 (non-commercial/research use only).

No turnkey NIM or vendor image exists for this pipeline, so this container
clones BUTSpeechFIT/DiariZen directly at a pinned commit -- see
../diarizen_up.sh and docker-compose.diarizen.yml.

No PyPI-compiled PyTorch C++ extension loads against this NGC image's
custom-patched libtorch (see the Dockerfile note next to where the broken
`_torchaudio*.so` is deleted). That leaves `torchaudio.load()` -- the one
call DiariZen's own inference.py makes -- without a working backend, so it
is monkeypatched below to a soundfile-based equivalent with the same
(waveform, sample_rate) return contract, before the pipeline module (which
imports torchaudio) is ever imported.

PyTorch 2.6 changed `torch.load`'s default `weights_only` from False to
True; the fork's `Model.from_pretrained` (via lightning_fabric's `pl_load`)
doesn't pass it explicitly, so loading the pretrained checkpoints -- pickled
years before that default flipped -- fails on an unlisted global
(`torch.torch_version.TorchVersion`). `torch.load` is wrapped below to
default `weights_only=False` unless the caller sets it explicitly; the
checkpoints come from BUT-FIT's and pyannote's official Hugging Face repos,
not user-uploaded data, so this doesn't change what's trusted here.
"""

import shutil
import tempfile
import threading
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from wave import open as wave_open

import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, HTTPException, UploadFile

_MODEL_ID = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"

_lock = threading.Lock()
_pipeline = None


def _load_via_soundfile(path, *args, **kwargs) -> tuple[torch.Tensor, int]:
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T), sample_rate


torchaudio.load = _load_via_soundfile

_torch_load = torch.load


def _load_weights_only_false(*args, **kwargs):
    # lightning_fabric's pl_load passes weights_only=None explicitly (its own
    # "unset" sentinel) rather than omitting the kwarg, so setdefault() alone
    # doesn't override it -- torch.load treats an explicit None the same as
    # not-specified and falls back to its own 2.6+ default of True.
    if kwargs.get("weights_only") is None:
        kwargs["weights_only"] = False
    return _torch_load(*args, **kwargs)


torch.load = _load_weights_only_false


def _wav_duration_sec(path: Path) -> float | None:
    with suppress(Exception):
        with wave_open(str(path), "rb") as wav:
            return wav.getnframes() / wav.getframerate()
    return None


def _load_pipeline():
    from diarizen.pipelines.inference import DiariZenPipeline

    return DiariZenPipeline.from_pretrained(_MODEL_ID)


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

    work_dir = Path(tempfile.mkdtemp(prefix="diarizen-req-"))
    try:
        audio_path = work_dir / (file.filename or "audio.wav")
        audio_path.write_bytes(await file.read())

        with _lock:
            result = _pipeline(str(audio_path), sess_name=audio_path.stem)

        return {
            "audio_duration_sec": _wav_duration_sec(audio_path),
            "rttm": result.to_rttm(),
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
