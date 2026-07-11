# VibeVoice-ASR (Microsoft, MIT) -- an 8B decoder-only model that performs
# ASR, speaker diarization, and timestamping jointly in a single
# autoregressive pass over up to 60 minutes of audio. GPU inference runs in
# its own container, outside the API/worker process -- the worker calls it
# over HTTP.
#
# Containerized rather than pip-installed into the worker because it needs
# transformers>=5.3.0, while this platform's `pyannote` model pins
# pyannote.audio>=4.0 with its own transformers floor -- and because a
# resident 16GB BF16 model has no business being loaded and evicted per RQ
# job. Same reasoning as deploy/nemo-clustering and deploy/diarizen.
#
# Uses the NGC PyTorch container (multi-arch: amd64 + arm64) rather than the
# Docker Hub pytorch/pytorch image, which only publishes amd64 -- this
# platform also runs on DGX Spark (Grace, aarch64). 24.10-py3 predates GB10
# (Grace Blackwell) support and fails at runtime with "No supported GPU(s)
# detected"; 26.06-py3 is built against a newer driver (610.43+) than this
# host has (580.159.03) and refuses to run. 26.02-py3 is the tag
# deploy/diarizen already runs successfully against this exact host driver.
FROM nvcr.io/nvidia/pytorch:26.02-py3

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# VibeVoiceAsrForConditionalGeneration landed in transformers 5.3.0; there is
# no earlier version that can load this architecture. --no-deps on torch's
# transitive pull is not needed here (transformers floors torch but doesn't
# pin it), but keep an eye on this if a future base image ships a torch that
# transformers wants to replace: the NGC libtorch is custom-patched and a
# PyPI torch build must never evict it.
RUN pip install --no-cache-dir \
    "transformers>=5.3.0" \
    accelerate \
    soundfile \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.30" \
    "python-multipart>=0.0.9"

# Deliberately NOT installing flash-attn: PyPI ships no aarch64 wheel for it
# (this image is multi-arch for DGX Spark/Grace), and building it from source
# takes the better part of an hour. server.py asks for attn_implementation
# ="sdpa" instead -- PyTorch's built-in scaled_dot_product_attention, which is
# available on every arch and is a supported path for this model.

COPY server.py ./

EXPOSE 9023

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9023"]
