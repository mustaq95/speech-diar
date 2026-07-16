# MOSS-Transcribe-Diarize (OpenMOSS, Apache-2.0) -- a 0.9B end-to-end model
# performing ASR, speaker diarization and timestamping jointly in a single
# autoregressive pass. GPU inference runs in its own container, outside the
# API/worker process -- the worker calls it over HTTP.
#
# Almost nothing is needed on top of the stock image: unlike deploy/vibevoice/,
# there is no plugin to install and no launcher to copy, because MOSS is
# registered in vLLM upstream (vllm/model_executor/models/registry.py as of the
# v0.25.1 release). The model id and every serve flag live in
# docker-compose.moss-transcribe.yml, appended to this image's inherited
# ["vllm", "serve"] entrypoint.
#
# The v0.25.1 base is chosen for two reasons, both load-bearing on this host:
#   1. It is a CUDA 13 build (CUDA_VERSION=13.0.2), so the toolkit ptxas at
#      /usr/local/cuda/bin/ptxas understands this GPU's sm_121a (GB10/DGX
#      Spark Blackwell). A CUDA 12.x base cannot assemble for it at all.
#      Triton must also be pointed AT that ptxas -- see TRITON_PTXAS_PATH in
#      the compose file; the base image alone is necessary but not sufficient.
#   2. It publishes a linux/arm64 manifest, so it runs on this Grace host
#      without the NGC tag-hunting the other deploy/*/Dockerfiles need.
# Upstream's README predates the mainline registration and pins a nightly wheel
# (commit 68b4a1d5, 0.23.1rc1.dev949); that build is now OLDER than this tagged
# release, so following it would ship a rotting wheel pin for no benefit.
FROM vllm/vllm-openai:v0.25.1

# vLLM keeps its audio decode stack in an OPTIONAL extra, so the stock image
# cannot read an audio file at all: /v1/audio/transcriptions fails every
# request with a 400 "Invalid or unsupported audio file." (raised from
# entrypoints/speech_to_text/base/serving.py, which swallows the underlying
# ModuleNotFoundError). Caught live -- the container starts, loads the model,
# reports /health 200, and only then rejects every request, which is why this
# is worth an explicit comment rather than a bare install line.
#
# ffmpeg and libsndfile are already present in the base image; only the Python
# packages are missing. Pinned to the versions verified working here rather
# than floating, matching this repo's pin-don't-drift habit for the other
# deploy images (numpy is left alone -- these install cleanly against the
# image's 2.2.6).
RUN pip install --no-cache-dir librosa==0.11.0 soundfile==0.14.0

# Fail the BUILD, not a 3am cold start, if the pins above ever stop importing
# or shift a package vLLM needs -- same guard as deploy/vibevoice/Dockerfile
# and deploy/diarizen/Dockerfile.
RUN python3 -c "import vllm, librosa, soundfile, numpy; \
    print('vllm', vllm.__version__, 'librosa', librosa.__version__, \
          'soundfile', soundfile.__version__, 'numpy', numpy.__version__)"

EXPOSE 8000
