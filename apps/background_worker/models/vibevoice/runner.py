"""VibeVoice-ASR (Microsoft) -- 8B decoder-only model performing ASR, speaker
diarization, and timestamping jointly in a single autoregressive pass.

Execution code only; returns the NATIVE payload (the model's own segment
dicts, including the `Content` transcript this platform's contract has no
field for) untouched -- see `adapter.py` for the only code allowed to
understand that shape.

Container: custom image (NGC PyTorch base + transformers>=5.3.0, which is
where the VibeVoiceAsrForConditionalGeneration architecture lives), HTTP only.
Not pip-installed into the worker: a resident 16GB BF16 model would be loaded
and evicted per RQ job. See deploy/vibevoice/.

Configuration (`.env`, read via `packages/config/settings.py`):
  VIBEVOICE_URL -- base URL of the container (POST {url}/diarize)
  VIBEVOICE_TIMEOUT_SEC -- request timeout
"""

from typing import Any

import httpx

from packages.config.settings import get_settings

from ..base_model import ModelRunner

VibeVoiceRawOutput = dict[str, Any]


class VibeVoiceRunner(ModelRunner[VibeVoiceRawOutput]):
    model_id = "vibevoice"

    def run(self, audio_path: str, params: dict[str, Any] | None = None) -> VibeVoiceRawOutput:
        settings = get_settings()
        with open(audio_path, "rb") as f:
            response = httpx.post(
                f"{settings.vibevoice_url}/diarize",
                files={"file": (audio_path, f, "audio/wav")},
                timeout=settings.vibevoice_timeout_sec,
            )
        response.raise_for_status()
        return response.json()
