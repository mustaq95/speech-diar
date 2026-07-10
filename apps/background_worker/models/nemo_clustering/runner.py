"""NeMo Clustering Diarizer (MarbleNet VAD + TitaNet + spectral clustering).

Execution code only; returns the NATIVE payload (raw RTTM text, untouched) --
see `adapter.py` for the only code allowed to understand that shape.

Container: custom image built from nvcr.io/nvidia/nemo:26.02 (no turnkey NIM
ships this cascaded pipeline -- NIM only offers `diarizer=sortformer`), HTTP
only. See deploy/nemo-clustering/.

Configuration (`.env`, read via `packages/config/settings.py`):
  NEMO_CLUSTERING_URL -- base URL of the container (POST {url}/diarize)
  NEMO_CLUSTERING_TIMEOUT_SEC -- request timeout
"""

from typing import Any

import httpx

from packages.config.settings import get_settings

from ..base_model import ModelRunner

NemoClusteringRawOutput = dict[str, Any]


class NemoClusteringRunner(ModelRunner[NemoClusteringRawOutput]):
    model_id = "nemo-clustering"

    def run(self, audio_path: str, params: dict[str, Any] | None = None) -> NemoClusteringRawOutput:
        settings = get_settings()
        with open(audio_path, "rb") as f:
            response = httpx.post(
                f"{settings.nemo_clustering_url}/diarize",
                files={"file": (audio_path, f, "audio/wav")},
                timeout=settings.nemo_clustering_timeout_sec,
            )
        response.raise_for_status()
        return response.json()
