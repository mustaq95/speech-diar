# DiariZen (WavLM-Large + Conformer local end-to-end diarization, followed
# by pyannote-3.1-style global clustering) behind a thin FastAPI wrapper.
# GPU inference runs in its own container, outside the API/worker process --
# the worker calls it over HTTP.
#
# Not pip-installable as a standalone dependency: DiariZen isn't on PyPI, and
# it vendors a modified fork of pyannote-audio directly in-tree (not the pip
# package). That fork occupies the same `pyannote` import namespace as the
# real pyannote.audio>=4.0 this platform already pins for the `pyannote`
# model's `[models]` extra -- installing both into one process would break
# one of them. Isolating DiariZen in its own container sidesteps that
# entirely, same reasoning as deploy/nemo-clustering and
# deploy/3d-speaker-clustering.
#
# Uses the NGC PyTorch container (multi-arch: amd64 + arm64) rather than the
# Docker Hub pytorch/pytorch image, which only publishes amd64 -- this
# platform also runs on DGX Spark (Grace, aarch64), same reasoning as
# deploy/nemo-clustering/Dockerfile's NGC base. 24.10-py3 (the tag
# deploy/3d-speaker-clustering uses) predates GB10 (Grace Blackwell) support
# and fails at runtime with "No supported GPU(s) detected"; 26.06-py3 is
# built against a newer driver (610.43+) than this host has (580.159.03) and
# refuses to run. 26.02-py3 matches the same release cycle as
# nvcr.io/nvidia/nemo:26.02, which this platform's nemo-clustering service
# already runs successfully against this exact host driver.
FROM nvcr.io/nvidia/pytorch:26.02-py3

WORKDIR /app

# Pinned commit, not `main` -- so a rebuild months from now doesn't silently
# pick up upstream changes to the pipeline this adapter's parsing depends on.
ARG DIARIZEN_COMMIT=a60b18151dbbe246e4199d8ef5cd2ece3872ea94
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg && \
    rm -rf /var/lib/apt/lists/* && \
    git clone https://github.com/BUTSpeechFIT/DiariZen.git . && \
    git checkout "$DIARIZEN_COMMIT"

# Drop numpy from DiariZen's requirements.txt before installing: it pins an
# exact version (==1.26.4) while the NGC base image already ships a
# GPU-matched, arm64-native numpy build -- letting pip "satisfy" the exact
# pin would downgrade numpy under the preinstalled torch and risk an ABI
# mismatch (same reasoning as deploy/3d-speaker-clustering/Dockerfile). Also
# drop onnxruntime-gpu: PyPI ships no aarch64 wheel for it (this image is
# multi-arch for DGX Spark/Grace), and DiariZenPipeline's inference path
# (diarizen/pipelines/inference.py) is pure PyTorch -- onnxruntime is only
# used by other scripts in the repo we don't call.
RUN grep -viE '^(numpy|onnxruntime-gpu)' requirements.txt > requirements.filtered.txt && \
    pip install --no-cache-dir -r requirements.filtered.txt && \
    pip install --no-cache-dir -e .

# The vendored pyannote-audio fork: only the base install (its own
# requirements.txt is read directly by its setup.py) -- the README's
# `dev,testing` extras (pytest, jupyter, papermill, black) are for
# development, not needed to run inference. Also drop the top-level
# torch/torchaudio floors (>=2.0.0 / >=2.1.1) from its requirements.txt to
# avoid pip reinstalling torch itself -- speechbrain still pulls in a PyPI
# torchaudio wheel transitively regardless (handled below).
RUN cd pyannote-audio && \
    grep -vE '^(torch|torchaudio)[[:space:]]' requirements.txt > requirements.filtered.txt && \
    mv requirements.filtered.txt requirements.txt && \
    pip install --no-cache-dir -e .

# speechbrain (a pyannote-audio dependency) floors torchaudio>=2.1.0 with no
# ceiling, so despite stripping the top-level pin above, pip still pulled in
# the latest torchaudio (2.11.0) transitively. torchaudio>=2.9 requires the
# separate `torchcodec` package for `.load()`, and this fork's code (built
# for the ~2.1-2.3 era) imports `AudioMetaData` from torchaudio's top level,
# a name later torchaudio versions removed. Force-reinstall a torchaudio
# version old enough to still export it, `--no-deps` so pip doesn't drag in
# a PyPI torch build and evict the NGC base's own (which torchvision is
# pinned against) -- 2.1.1, matching the fork's floor, has no cp312/aarch64
# wheel on PyPI; 2.8.0 is the oldest available here and still has
# `AudioMetaData`.
RUN pip install --no-cache-dir --no-deps --force-reinstall "torchaudio==2.8.0"

# No PyPI-compiled PyTorch C++ extension loads against this NGC image's
# custom-patched libtorch (missing symbols like `torch_library_impl` /
# `torch_from_blob` that standard libtorch exports) -- confirmed for both
# torchaudio's own extension and torchcodec. Deleting the broken extension
# .so makes torchaudio's `_load_lib` treat it as "not found" (a handled
# case) instead of "found but broken" (which it deliberately re-raises, by
# torchaudio's own design -- see torchaudio/_extension/utils.py). This only
# disables torchaudio's compiled I/O backend; `torchaudio.functional.resample`
# and everything else pyannote's `Audio` class needs is pure-tensor-op
# Python that still works. server.py replaces `torchaudio.load` with a
# soundfile equivalent (same file/tensor contract) before importing
# DiariZen's pipeline, which is the only place `.load()` itself is called.
RUN find / -path '*/torchaudio/lib/_torchaudio*.so' -delete 2>/dev/null || true

# NumPy 2.0 removed the `np.NaN` / `np.NAN` aliases (renamed to `np.nan`);
# this fork predates that and the NGC base ships NumPy 2.x (kept, not
# downgraded -- see the numpy filtering above). Patched in place rather than
# pinning an older NumPy under the preinstalled torch.
RUN grep -rlE 'np\.(NaN|NAN)\b' pyannote-audio --include='*.py' | \
    xargs sed -i -E 's/np\.(NaN|NAN)\b/np.nan/g'

RUN pip install --no-cache-dir "fastapi>=0.115" "uvicorn[standard]>=0.30" "python-multipart>=0.0.9"

COPY server.py ./

EXPOSE 9022

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9022"]
