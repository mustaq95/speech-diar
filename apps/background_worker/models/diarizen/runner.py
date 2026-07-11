"""DiariZen (WavLM-Large + Conformer local end-to-end diarization, followed
by pyannote-3.1-style global clustering across the whole file).

Execution code only; returns the NATIVE payload (raw RTTM text, untouched) --
see `adapter.py` for the only code allowed to understand that shape.

Container: custom image built by cloning BUTSpeechFIT/DiariZen at a pinned
commit (not pip-installable -- it vendors a modified fork of pyannote-audio
in-tree, which would collide with the real pyannote.audio this platform
already pins for the `pyannote` model if installed in the same process), HTTP
only. See deploy/diarizen/.

Configuration (`.env`, read via `packages/config/settings.py`):
  DIARIZEN_URL -- base URL of the container (POST {url}/diarize)
  DIARIZEN_TIMEOUT_SEC -- request timeout
"""

from typing import Any

import httpx

from packages.config.settings import get_settings

from ..base_model import ModelRunner

DiarizenRawOutput = dict[str, Any]


class DiarizenRunner(ModelRunner[DiarizenRawOutput]):
    model_id = "diarizen"

    def run(self, audio_path: str, params: dict[str, Any] | None = None) -> DiarizenRawOutput:
        settings = get_settings()
        with open(audio_path, "rb") as f:
            response = httpx.post(
                f"{settings.diarizen_url}/diarize",
                files={"file": (audio_path, f, "audio/wav")},
                timeout=settings.diarizen_timeout_sec,
            )
        response.raise_for_status()
        return response.json()
