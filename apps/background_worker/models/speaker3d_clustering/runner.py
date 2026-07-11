"""3D-Speaker CAM++ Clustering Diarizer (FSMN VAD + CAM++ speaker embeddings
+ clustering, no ASR/transcription step at all).

Execution code only; returns the NATIVE payload (raw RTTM text, untouched) --
see `adapter.py` for the only code allowed to understand that shape.

Container: custom image built by cloning modelscope/3D-Speaker at a pinned
commit (no turnkey NIM or vendor image exists for this pipeline), HTTP only.
See deploy/3d-speaker-clustering/.

Configuration (`.env`, read via `packages/config/settings.py`):
  SPEAKER3D_CLUSTERING_URL -- base URL of the container (POST {url}/diarize)
  SPEAKER3D_CLUSTERING_TIMEOUT_SEC -- request timeout
"""

from typing import Any

import httpx

from packages.config.settings import get_settings

from ..base_model import ModelRunner

Speaker3dClusteringRawOutput = dict[str, Any]


class Speaker3dClusteringRunner(ModelRunner[Speaker3dClusteringRawOutput]):
    model_id = "3d-speaker-clustering"

    def run(self, audio_path: str, params: dict[str, Any] | None = None) -> Speaker3dClusteringRawOutput:
        settings = get_settings()
        with open(audio_path, "rb") as f:
            response = httpx.post(
                f"{settings.speaker3d_clustering_url}/diarize",
                files={"file": (audio_path, f, "audio/wav")},
                timeout=settings.speaker3d_clustering_timeout_sec,
            )
        response.raise_for_status()
        return response.json()
