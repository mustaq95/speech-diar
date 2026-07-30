"""Azure Speech BATCH transcription with diarization — execution code.

Submits a batch job to the v3.2 transcriptions API, polls until it finishes,
downloads the result JSON, and returns it untouched; only ``adapter.py``
understands that shape.

Batch mode is URL-based: Azure fetches the audio itself, so the input must be
an http(s) URL Azure can reach (public URL or blob SAS URL from
packages/storage). It is asynchronous and suited to long recordings, unlike
the real-time ``azure`` model which streams a local file. Batch transcription
is available in all standard regions, including those without fast
transcription (e.g. uaenorth).

Configuration: AZURE_SPEECH_KEY + AZURE_SPEECH_REGION (or AZURE_SPEECH_ENDPOINT)
in `.env`, read via `packages/config/settings.py`.
"""

import logging
import time
from typing import Any

import requests

from packages.config.settings import get_settings

from ..base_model import ModelRunner

logger = logging.getLogger(__name__)

# Native shape: the v3.2 batch result JSON —
# {"durationInTicks": ..., "recognizedPhrases": [{"speaker": 1,
#   "offsetInTicks": ..., "durationInTicks": ..., "nBest": [{"display": ...}]}]}
AzureBatchRawOutput = dict[str, Any]


def _endpoint() -> str:
    settings = get_settings()
    if settings.azure_speech_endpoint:
        return settings.azure_speech_endpoint.rstrip("/")
    if not settings.azure_speech_region:
        raise RuntimeError("Set AZURE_SPEECH_ENDPOINT or AZURE_SPEECH_REGION")
    return f"https://{settings.azure_speech_region}.api.cognitive.microsoft.com"


class AzureBatchRunner(ModelRunner[AzureBatchRawOutput]):
    model_id = "azure-batch"

    def run(self, audio_path: str, params: dict[str, Any] | None = None) -> AzureBatchRawOutput:
        if not audio_path.startswith(("http://", "https://")):
            raise RuntimeError(
                "azure-batch needs an http(s) URL Azure can fetch (e.g. a blob "
                "SAS URL). For local files use the 'azure' real-time model."
            )
        settings = get_settings()
        if not settings.azure_speech_key:
            raise RuntimeError("AZURE_SPEECH_KEY is not set")

        params = params or {}
        headers = {"Ocp-Apim-Subscription-Key": settings.azure_speech_key}
        base = f"{_endpoint()}/speechtotext/v3.2/transcriptions"

        submit = requests.post(
            base,
            headers=headers,
            json={
                "displayName": "diarization-platform",
                "locale": params.get("language", settings.locale),
                "contentUrls": [audio_path],
                "properties": {
                    "diarizationEnabled": True,
                    "diarization": {
                        "speakers": {
                            "minCount": 1,
                            "maxCount": params.get("max_speakers", settings.azure_batch_max_speakers),
                        }
                    },
                    "punctuationMode": "DictatedAndAutomatic",
                    "timeToLive": settings.azure_batch_time_to_live,
                },
            },
            timeout=60,
        )
        submit.raise_for_status()
        job_url = submit.json()["self"]

        try:
            started = time.monotonic()
            deadline = started + settings.azure_batch_job_timeout_sec
            while True:
                # Deadline check comes BEFORE the GET: this is a paid API, so
                # once the timeout expires no further calls go out (the
                # `finally` DELETE below is the one exception — it cancels the
                # server-side job so Azure stops billing it).
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Batch job timed out after {settings.azure_batch_job_timeout_sec}s")
                job = requests.get(job_url, headers=headers, timeout=60).json()
                status = job.get("status")
                # Azure queues batch jobs, so a long wait here is normal and
                # says nothing about audio length. Without this line the loop
                # is a black box and any stall looks identical to a hang.
                logger.info("azure-batch job %s: status=%s after %.0fs", job_url.rsplit("/", 1)[-1], status, time.monotonic() - started)
                if status == "Succeeded":
                    break
                if status == "Failed":
                    raise RuntimeError(f"Batch job failed: {job.get('properties', {}).get('error')}")
                time.sleep(settings.azure_batch_poll_interval_sec)

            files = requests.get(f"{job_url}/files", headers=headers, timeout=60).json()
            for entry in files.get("values", []):
                if entry.get("kind") == "Transcription":
                    content_url = entry["links"]["contentUrl"]
                    result = requests.get(content_url, timeout=120)
                    result.raise_for_status()
                    payload = result.json()
                    # Azure's own bracket for this job, used to prefer Azure's
                    # reported duration over our worker-measured wall-clock.
                    payload["_azureJobTiming"] = {
                        "createdDateTime": job.get("createdDateTime"),
                        "lastActionDateTime": job.get("lastActionDateTime"),
                    }
                    return payload
            raise RuntimeError("Batch job succeeded but produced no transcription file")
        finally:
            # Best effort cleanup; jobs also expire via timeToLive.
            try:
                requests.delete(job_url, headers=headers, timeout=30)
            except requests.RequestException:
                pass
