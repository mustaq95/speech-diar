# 3D-Speaker CAM++ Clustering Diarizer (FSMN VAD + CAM++ speaker embeddings +
# clustering) behind a thin FastAPI wrapper. GPU inference runs in its own
# container, outside the API/worker process -- the worker calls it over HTTP.
#
# No turnkey NIM or vendor image exists for this pipeline, so this clones
# modelscope/3D-Speaker directly at a pinned commit. Base is plain Ubuntu
# (multi-arch, no CUDA toolkit needed at the OS level) with torch+torchaudio
# installed together via a single `pip install torch torchaudio` -- pip's
# CUDA wheels bundle their own CUDA runtime libs, and installing both in one
# command lets pip resolve a matched, ABI-compatible pair. An NGC PyTorch
# base was tried first (matching deploy/nemo-clustering/Dockerfile's
# approach) but its custom torch build (e.g. "2.5.0a0+e000cf0") has no
# matching public torchaudio wheel and fails to build from source against
# it -- plain pip torch+torchaudio avoids that mismatch entirely.
FROM ubuntu:22.04

WORKDIR /app

# build-essential/python3-dev: hdbscan (a clustering dep pulled in below)
# compiles a C extension at install time; there's no prebuilt aarch64 wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg python3 python3-pip python3-dev build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Matched pair, resolved together so their CUDA/ABI versions agree.
RUN pip install --no-cache-dir torch torchaudio

# Pinned commit, not `main` -- so a rebuild months from now doesn't silently
# pick up upstream changes to the pipeline this adapter's parsing depends on.
ARG THREED_SPEAKER_COMMIT=065629c313eaf1a01c65c640c46d77e61e9607b4
RUN git clone https://github.com/modelscope/3D-Speaker.git . && \
    git checkout "$THREED_SPEAKER_COMMIT"

# Drop torch/torchaudio/numpy from 3D-Speaker's requirements.txt: torch and
# torchaudio are already installed as a matched pair above, and
# requirements.txt's numpy upper bound (<1.24) predates torch's own numpy
# needs by several years -- let torch's dependency resolution pick numpy.
# addict/datasets/simplejson/funasr/fastcluster/hdbscan: transitive runtime
# deps of modelscope's audio pipeline and 3D-Speaker's clustering backend
# that aren't declared by modelscope/3D-Speaker's own package metadata --
# found by actually running the pipeline (ImportError one at a time), not
# guessed. `pip install modelscope[audio]` alone does not pull all of these.
RUN grep -viE '^(torch|torchaudio|numpy)' requirements.txt > requirements.filtered.txt && \
    pip install --no-cache-dir numpy -r requirements.filtered.txt && \
    pip install --no-cache-dir modelscope "pyannote.audio>=3.0" \
        addict datasets simplejson funasr fastcluster hdbscan \
        "fastapi>=0.115" "uvicorn[standard]>=0.30" "python-multipart>=0.0.9"

COPY server.py ./

EXPOSE 9021

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9021"]
