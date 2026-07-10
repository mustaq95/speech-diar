"""NVIDIA Parakeet-Sortformer NIM — streaming profile.

Execution code only; returns the NATIVE payload (untouched word list) —
see `../_nim_shared.py` for the shared gRPC call and native shape, and
`adapter.py` for the only code allowed to understand that shape.

Container: nvcr.io/nim/nvidia/parakeet-1-1b-rnnt-multilingual, profile
`diarizer=sortformer,mode=str,type=default,vad=silero` (deploy/parakeet_nim_up.sh).
This platform diarizes a completed upload, not a live mic — "streaming"
here means the container's RPC shape (`StreamingRecognize` fed the whole
file chunk-by-chunk), not real-time playback.

Configuration (`.env`, read via `packages/config/settings.py`):
  NIM_STR_GRPC — gRPC host:port of the streaming NIM container
  NIM_LANGUAGE, NIM_MAX_SPEAKERS, NIM_GRPC_TIMEOUT_SEC — shared NIM settings
"""

from typing import Any

from packages.config.settings import get_settings

from .. import _nim_shared
from ..base_model import ModelRunner

NimSortformerStrRawOutput = _nim_shared.NimRawOutput


class NimSortformerStrRunner(ModelRunner[NimSortformerStrRawOutput]):
    model_id = "nim-sortformer-str"

    def run(self, audio_path: str, params: dict[str, Any] | None = None) -> NimSortformerStrRawOutput:
        settings = get_settings()
        return _nim_shared.recognize_streaming(audio_path, settings.nim_str_grpc)
