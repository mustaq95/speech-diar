"""PyAnnote Community-1 (ADEO) -- execution code.

The same community-1 model as `models/pyannote`, but deployed on-prem by ADEO
(coro `speakrs` backend on a MIG slice) and reached through the LiteLLM
gateway instead of loaded in-process. Two pipelines for one model, so the two
rows compare the deployments rather than the weights.

Speaks the pyannoteAI `/v1/diarize` contract: multipart `file` plus form
fields, NOT OpenAI transcriptions. Jobs are synchronous, so the response is
the finished `DiarizationJob` and there is no polling step. Every other
pyannoteAI field (numSpeakers, minSpeakers, ...) is rejected by the pod with a
400, so none is sent: the engine picks the speaker count itself.

Returns the NATIVE job untouched; only ``adapter.py`` understands its shape.

Configuration (`.env`, read via `packages/config/settings.py`):
  LITELLM_BASE_URL / LITELLM_API_KEY -- the gateway, shared with Inception
  DIARIZE_PATH -- default /v1/diarize
  PYANNOTE_ADEO_MODEL -- default pyannote/community-1
  PYANNOTE_ADEO_TIMEOUT_SEC -- request timeout
  VERIFY_SSL / LITELLM_CA_BUNDLE -- gateway TLS
"""

import os
import wave
from contextlib import suppress
from typing import Any

import httpx

from packages.config.settings import get_settings

from ..base_model import ModelRunner

# Native shape: {"job": <pyannoteAI DiarizationJob JSON>, "audio_duration_sec": float | None}
PyannoteAdeoRawOutput = dict[str, Any]


def _wav_duration_sec(path: str) -> float | None:
    with suppress(Exception):
        with wave.open(path, "rb") as wav:
            return wav.getnframes() / wav.getframerate()
    return None


class PyannoteAdeoRunner(ModelRunner[PyannoteAdeoRawOutput]):
    model_id = "pyannote-adeo"

    def run(self, audio_path: str, params: dict[str, Any] | None = None) -> PyannoteAdeoRawOutput:
        settings = get_settings()
        if not settings.diarize_url or not settings.litellm_api_key:
            raise RuntimeError("LITELLM_BASE_URL and LITELLM_API_KEY must be set")

        with open(audio_path, "rb") as f:
            response = httpx.post(
                settings.diarize_url,
                headers={"Authorization": f"Bearer {settings.litellm_api_key}"},
                files={"file": (os.path.basename(audio_path), f, "audio/wav")},
                data={
                    "model": settings.pyannote_adeo_model,
                    # Fills output.exclusiveDiarization as well. The adapter
                    # renders the overlap-preserving list; this keeps the other
                    # one in raw_output rather than never asking for it.
                    "exclusive": "true",
                },
                timeout=settings.pyannote_adeo_timeout_sec,
                verify=settings.litellm_httpx_verify,
            )
        # Not raise_for_status(): the gateway wraps the pod's per-field
        # validation message in the body, and httpx's message drops it.
        if response.status_code >= 400:
            raise RuntimeError(f"pyannote-adeo {response.status_code}: {response.text[:500]}")
        return {"job": response.json(), "audio_duration_sec": _wav_duration_sec(audio_path)}
