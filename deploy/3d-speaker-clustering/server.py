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

Clustering backend: `Diarization3Dspeaker.__init__` always builds its own
`spectral` clustering backend internally (`speakerlab.bin.infer_diarization
.get_cluster_backend`, hardcoded, not exposed as a constructor argument).
3D-Speaker's own docs describe spectral clustering as suitable for "medium-
length audio (<30min) with relatively few speakers (<6)", recommending
UMAP-HDBSCAN instead for longer/many-speaker audio -- exactly the regime
this platform's real eval sample (a 32min/8-speaker recording) sits in, and
where spectral clustering under-detected badly (1 of 8 speakers). Rather
than fork the vendored library, `_load_pipeline` swaps in a
`umap_hdbscan`-backed `CommonClustering` after construction: same
post-processing (minor-cluster filtering, cosine-similarity merge) 3D-Speaker
itself always applies regardless of backend, just a different clustering
step. UMAP-HDBSCAN is density-based and takes no speaker-count parameter at
all -- strictly more unsupervised than spectral's own `oracle_num` escape
hatch, which stays unset either way.

`speakerlab.process.cluster.UmapHdbscan` builds `umap.UMAP(...)` with no
`random_state`, so its dimensionality reduction is non-deterministic --
rerunning the *same* audio through the *same* code produced 2, 3, 8, and 10
speakers across five back-to-back runs during verification, none of them
wrong exactly, just not reproducible. `_SeededUmapHdbscan` below overrides
`__call__` to pass a fixed seed into the same `umap.UMAP` construction
`UmapHdbscan` itself does -- not a speaker-count hint, just pinning the
clustering algorithm's own randomness so this platform's evaluation results
are stable across runs, which a comparison tool depends on. Single-threaded
as a result (UMAP's own tradeoff for a fixed seed), which is a non-issue at
this pipeline's audio-length scale.

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


class _SeededUmapHdbscan:
    """Wraps `speakerlab.process.cluster.UmapHdbscan` to pin UMAP's random
    seed -- see the module docstring for why. Delegates every constructor
    argument straight through; only `__call__` differs, and only by adding
    `random_state`."""

    def __init__(self, *, seed: int = 0, **kwargs):
        from speakerlab.process.cluster import UmapHdbscan

        self._inner = UmapHdbscan(**kwargs)
        self._seed = seed

    def __call__(self, X, **kwargs):
        import hdbscan
        import umap

        umap_X = umap.UMAP(
            n_neighbors=self._inner.n_neighbors,
            min_dist=0.0,
            n_components=min(self._inner.n_components, X.shape[0] - 2),
            metric=self._inner.metric,
            random_state=self._seed,
        ).fit_transform(X)
        return hdbscan.HDBSCAN(
            min_samples=self._inner.min_samples,
            min_cluster_size=self._inner.min_cluster_size,
        ).fit_predict(umap_X)


def _load_pipeline():
    import torch
    from speakerlab.bin.infer_diarization import Diarization3Dspeaker
    from speakerlab.process.cluster import CommonClustering

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = Diarization3Dspeaker(device=device, include_overlap=False, speaker_num=None)
    # Same mer_cos/min_cluster_size post-processing get_cluster_backend() used
    # for spectral -- only the clustering step itself changes.
    cluster = CommonClustering(cluster_type="umap_hdbscan", mer_cos=0.8, min_cluster_size=4)
    cluster.cluster = _SeededUmapHdbscan(min_cluster_size=4)
    pipeline.cluster = cluster
    return pipeline


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
