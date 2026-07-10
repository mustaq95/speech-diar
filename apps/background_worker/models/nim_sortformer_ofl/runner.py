"""NVIDIA Parakeet-Sortformer NIM — offline batch profile.

Execution code only; returns the NATIVE payload (untouched word list) —
see `../_nim_shared.py` for the shared gRPC call and native shape, and
`adapter.py` for the only code allowed to understand that shape.

Container: nvcr.io/nim/nvidia/parakeet-1-1b-rnnt-multilingual, profile
`diarizer=sortformer,mode=ofl,type=default,vad=silero` (deploy/parakeet_nim_up.sh).

Configuration (`.env`, read via `packages/config/settings.py`):
  NIM_OFL_GRPC — gRPC host:port of the offline-batch NIM container
  NIM_LANGUAGE, NIM_MAX_SPEAKERS, NIM_GRPC_TIMEOUT_SEC — shared NIM settings
"""

from typing import Any

from packages.config.settings import get_settings

from .. import _nim_shared
from ..base_model import ModelRunner

NimSortformerOflRawOutput = _nim_shared.NimRawOutput


class NimSortformerOflRunner(ModelRunner[NimSortformerOflRawOutput]):
    model_id = "nim-sortformer-ofl"

    def run(self, audio_path: str, params: dict[str, Any] | None = None) -> NimSortformerOflRawOutput:
        settings = get_settings()
        return _nim_shared.recognize_offline(audio_path, settings.nim_ofl_grpc)
