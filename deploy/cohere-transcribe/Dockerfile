# Cohere Transcribe Arabic (CohereLabs, Apache-2.0) -- a 2B speech-to-text
# model for Arabic and English, used as the OFFLINE half of this platform's
# live-speech transcript (TRANSCRIPTION_MODE=offline). GPU inference runs in
# this container, outside the API/worker process -- the worker calls it over
# HTTP and the audio never leaves this host.
#
# Almost nothing is needed on top of the stock image, exactly like
# deploy/moss-transcribe/: CohereAsrForConditionalGeneration is registered in
# vLLM upstream, so there is no plugin to install and no launcher to copy. The
# model id and every serve flag live in docker-compose.cohere-transcribe.yml,
# appended to this image's inherited ["vllm", "serve"] entrypoint.
#
# The v0.25.1 base is the same tag deploy/moss-transcribe/ pins, for the same
# two host-specific reasons -- both load-bearing here:
#   1. It is a CUDA 13 build (CUDA_VERSION=13.0.2), so the toolkit ptxas at
#      /usr/local/cuda/bin/ptxas understands this GPU's sm_121a (GB10/DGX
#      Spark Blackwell). A CUDA 12.x base cannot assemble for it at all.
#   2. It publishes a linux/arm64 manifest, so it runs on this Grace host
#      without the NGC tag-hunting the other deploy/*/Dockerfiles need.
# It is also new enough to carry this model: CohereAsrForConditionalGeneration
# is in v0.25.1's model registry (verified against the tag, not just main).
FROM vllm/vllm-openai:v0.25.1

# vLLM keeps its audio decode stack in an OPTIONAL extra, so the stock image
# cannot read an audio file at all: /v1/audio/transcriptions fails every
# request with a 400 "Invalid or unsupported audio file." (raised from
# entrypoints/speech_to_text/base/serving.py, which swallows the underlying
# ModuleNotFoundError). Caught live on moss-transcribe -- the container starts,
# loads the model, reports /health 200, and only then rejects every request,
# which is why this is worth an explicit comment rather than a bare install line.
#
# ffmpeg and libsndfile are already present in the base image; only the Python
# packages are missing. Same pins moss-transcribe verified on this host.
RUN pip install --no-cache-dir librosa==0.11.0 soundfile==0.14.0

# Fail the BUILD, not a 3am cold start, if the pins above stop importing OR if
# a future base-tag bump drops this model's architecture. The registry check is
# the addition over moss-transcribe's guard: without it, an unsupported base
# would only reveal itself as a confusing engine-startup crash after a multi-GB
# weight download.
RUN python3 -c "import vllm, librosa, soundfile, numpy; \
    from vllm.model_executor.models.registry import ModelRegistry; \
    assert 'CohereAsrForConditionalGeneration' in ModelRegistry.get_supported_archs(), \
        'this vLLM base does not support CohereAsrForConditionalGeneration'; \
    print('vllm', vllm.__version__, 'librosa', librosa.__version__, \
          'soundfile', soundfile.__version__, 'numpy', numpy.__version__)"

EXPOSE 8000
