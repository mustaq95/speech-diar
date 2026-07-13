#!/usr/bin/env python3
"""Container entrypoint for the vLLM-served VibeVoice-ASR model.

Mirrors steps 3-5 of upstream's vllm_plugin/scripts/start_server.py (model
download, tokenizer file generation, `vllm serve` exec) but skips the parts
that are hostile to repeated supervisor cold starts: that script re-installs
the VibeVoice package and re-downloads tokenizer files from the Qwen2.5-7B
repo on every single start. Here, the package is installed once at image
build time (see Dockerfile), and the tokenizer files are generated once per
model snapshot and then left on the cache volume.
"""

import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = os.environ.get("VIBEVOICE_MODEL_ID", "microsoft/VibeVoice-ASR")
MAX_NUM_SEQS = os.environ.get("VIBEVOICE_MAX_NUM_SEQS", "2")
MAX_MODEL_LEN = os.environ.get("VIBEVOICE_MAX_MODEL_LEN", "65536")
GPU_MEM_UTIL = os.environ.get("VIBEVOICE_GPU_MEM_UTIL", "0.35")
PORT = os.environ.get("VIBEVOICE_INTERNAL_PORT", "8000")

_TOKENIZER_FILES = (
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
)


def resolve_model_path() -> str:
    try:
        return snapshot_download(MODEL_ID, local_files_only=True)
    except Exception:
        print(f"Model {MODEL_ID} not cached; downloading ...", flush=True)
        return snapshot_download(MODEL_ID)


def ensure_tokenizer_files(model_path: str) -> None:
    if all((Path(model_path) / name).exists() for name in _TOKENIZER_FILES):
        return
    print("Generating tokenizer files ...", flush=True)
    subprocess.run(
        [sys.executable, "-m", "vllm_plugin.tools.generate_tokenizer_files", "--output", model_path],
        check=True,
        cwd="/app",
    )


def main() -> None:
    model_path = resolve_model_path()
    ensure_tokenizer_files(model_path)

    vllm_cmd = [
        "vllm", "serve", model_path,
        "--served-model-name", "vibevoice",
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--max-num-seqs", MAX_NUM_SEQS,
        "--max-model-len", MAX_MODEL_LEN,
        "--gpu-memory-utilization", GPU_MEM_UTIL,
        "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
        "--chat-template-content-format", "openai",
        "--port", PORT,
    ]
    print("Starting:", " ".join(vllm_cmd), flush=True)
    os.execvp("vllm", vllm_cmd)


if __name__ == "__main__":
    main()
