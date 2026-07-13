# VibeVoice-ASR (Microsoft, MIT) -- an 8B decoder-only model that performs
# ASR, speaker diarization, and timestamping jointly in a single
# autoregressive pass over up to 60 minutes of audio. GPU inference runs in
# its own container, outside the API/worker process -- the worker calls it
# over HTTP.
#
# Served through vLLM (Microsoft's official recipe, see
# https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-vllm-asr.md)
# rather than plain transformers.generate(): vLLM's continuous batching and
# CUDA-graph-captured decode are substantially faster than the eager-mode
# transformers path this container used to run, which is what let a 32-minute
# meeting take over 30 minutes to decode.
#
# vllm/vllm-openai publishes a multi-arch (amd64 + arm64) image, so this also
# runs on DGX Spark (Grace, aarch64) without swapping base images the way the
# other deploy/*/Dockerfiles have to (NGC pytorch tag hunting).
#
# This host's GPU is `sm_121a` (GB10/DGX Spark Blackwell Ultra). Any
# Triton/torch.compile kernel -- which vLLM's CUDA-graph capture always
# triggers -- must be assembled by a `ptxas` new enough to know that arch,
# or the engine core dies at startup with `ptxas fatal: Value 'sm_121a' is
# not defined for option 'gpu-name'`. This needs BOTH halves, which is easy
# to get wrong (an earlier pass here thought the base image alone sufficed):
#
#   1. The `-cu130` base (CUDA 13.0, matching this host's driver CUDA
#      version) so the TOOLKIT ptxas at /usr/local/cuda/bin/ptxas (V13.0.88)
#      supports sm_121a. The plain `v0.14.1` tag's CUDA 12.x toolkit ptxas
#      does not. Verified live: `ptxas --gpu-name=sm_121a` succeeds with the
#      cu130 toolkit binary, fails with the 12.x one.
#   2. Pointing Triton AT that toolkit ptxas. vLLM does not use the toolkit
#      ptxas -- Triton ships its OWN bundled `ptxas` (V12.8.93, at
#      triton/backends/nvidia/bin/ptxas) and uses it by default, so the
#      cu130 base is necessary but NOT sufficient. The override
#      `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` (set in
#      docker-compose.vibevoice.yml) makes Triton use the sm_121a-capable
#      toolkit binary. Verified live: an identical Triton kernel that fails
#      with the bundled ptxas compiles clean with the override.
#
# See vllm-project/vllm#33313 and triton-lang/triton#8539.
FROM vllm/vllm-openai:v0.14.1-cu130

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 git && \
    rm -rf /var/lib/apt/lists/*

# Pinned commit, not `main` -- so a rebuild months from now doesn't silently
# pick up upstream changes to the vLLM plugin or the request/response format
# this platform's runner.py depends on.
ARG VIBEVOICE_COMMIT=303b2833e01cff4578ec278bbfe536da54bd19fe
RUN git clone https://github.com/microsoft/VibeVoice /app && \
    git -C /app checkout "$VIBEVOICE_COMMIT"

# Upstream's own launcher (vllm_plugin/scripts/start_server.py) re-runs this
# install on every container start, which is fine for a one-off demo but not
# for a container the supervisor cold-starts repeatedly against a tight
# budget. Do it once, at build time, instead -- see serve.py for the launcher
# this image actually runs. (The `[vllm]` extra that script references
# doesn't exist in VibeVoice's pyproject.toml; plain `-e .` installs the same
# dependencies.)
RUN python3 -m pip install --no-cache-dir -e /app

# Fails the build (not a 3am cold start) if installing VibeVoice's deps moved
# a package vLLM itself needs pinned -- see the Dockerfile comment in
# deploy/diarizen for why this class of transitive-dependency breakage is
# worth guarding against explicitly here.
RUN python3 -c "import vllm, transformers, torch, vibevoice, vllm_plugin; \
    print('vllm', vllm.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__)"

COPY serve.py /opt/vibevoice/serve.py

EXPOSE 8000

ENTRYPOINT ["python3", "/opt/vibevoice/serve.py"]
