"""Speechmatics -- execution code. A hosted BATCH job API, so audio leaves this
host and this engine is `online`.

Unlike every other engine here it is not one request. It is three:

    POST   /v2/jobs                          -> {"id": ...}      (201)
    GET    /v2/jobs/{id}                     -> poll until status != "running"
    GET    /v2/jobs/{id}/transcript?format=json-v2

Returns the json-v2 transcript untouched; only `adapter.py` reads that shape.

Probed against the live API on 2026-09-01 with a 61 s code-switched
Arabic/English clip. Four findings, each load-bearing:

1. **`ar_en` selects Speechmatics' bilingual CODE-SWITCHING MODEL.** It is not a
   pinned language and must not be "relaxed" to a single code or to auto.
   Speechmatics documents its bilingual packs as transcribing "a selected
   combination of languages in one media file or stream, INCLUDING SPEAKERS WHO
   SWITCH BETWEEN THE LANGUAGES IN THAT PACK", and the API confirms the pack it
   ran as `language_pack_info.language_description == "Arabic and English"`.
   Measured here: Arabic stays in Arabic script, English stays in Latin, in one
   transcript. This IS the automatic code-switching configuration.

2. **`language: "auto"` is a DIFFERENT FEATURE and is not the code-switching
   test.** Language identification detects ONE language and transcribes the whole
   file in it. Measured with `expected_languages: ["ar", "en"]`: the API resolved
   a single pack (`predicted_language: "ar"`, all 90 result words tagged `ar`)
   and then transliterated or DROPPED every English passage -- whole English
   sentences vanished. WER 0.729 with 154 deletions, against 0.589 with 119 for
   `ar_en`.

   So "auto" here is not the neutral choice and does not measure whether this
   engine can code-switch; it measures a feature that by design cannot. Reading
   its poor score as "Speechmatics does not code-switch" is the wrong conclusion
   and was made once already.

3. **Submit-time validation is LAZY.** A language code the service does not know
   still returns `201 Created` with a job id. Nothing about a 201 means the
   config was accepted, so failure has to be read from the job STATUS, not from
   the submit response. `status` is checked against an explicit success value
   rather than "not running" for this reason.

4. The key is region-scoped: this credential is 200 on
   `asr.api.speechmatics.com` and 401 on `eu2.asr.api.speechmatics.com`.
   Repointing SPEECHMATICS_URL at another region is an auth change, not a
   latency tweak.

Live mode costs one full JOB per chunk. `transcribe_bytes` submits, polls and
fetches for each chunk it is handed, so its latency is dominated by queue
turnaround and our own poll granularity rather than by inference. The INPUT is
still fair -- every chunked engine gets byte-identical bytes from one recorder --
but this engine's live latency measures the job pipeline and must be read with
its transport label attached.

Configuration (`.env`, read via `packages/config/settings.py`):
  SPEECHMATICS_API_KEY / SPEECHMATICS_URL   -- credential and regional endpoint
  SPEECHMATICS_LANGUAGE                     -- "ar_en" bilingual pack (see above)
  SPEECHMATICS_OPERATING_POINT              -- "enhanced" | "standard"
  SPEECHMATICS_POLL_INTERVAL_SEC            -- how often the job is polled
  SPEECHMATICS_TIMEOUT_SEC                  -- ceiling on the whole job
"""

import json
import logging
import os
import time
from typing import Any

import httpx

from packages.audio import ensure_canonical_wav
from packages.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

SpeechmaticsRawOutput = dict[str, Any]

ENGINE = "speechmatics"

#: The job status that means a transcript exists. An explicit allow-list, not
#: `!= "running"`: "rejected" and "expired" are also not running, and treating
#: them as done would fetch a transcript that is not there and store the 404
#: body as if it were speech.
_DONE = "done"
_TERMINAL = {"done", "rejected", "expired"}


class SpeechmaticsError(RuntimeError):
    """Base for failures talking to the Speechmatics batch API."""


class SpeechmaticsUpstreamError(SpeechmaticsError):
    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"{ENGINE} {status_code}: {body[:500]}")
        self.status_code = status_code
        self.body = body


def _check(response: httpx.Response) -> Any:
    if response.status_code >= 400:
        raise SpeechmaticsUpstreamError(response.status_code, response.text)
    return response.json()


def transcription_config(settings: Settings) -> dict[str, Any]:
    """The job config posted for a run.

    Split out so a test can assert the language pack without making a request.
    """
    return {
        "type": "transcription",
        "transcription_config": {
            "language": settings.speechmatics_language,
            "operating_point": settings.speechmatics_operating_point,
        },
    }


def _run_job(wav_bytes: bytes, filename: str) -> SpeechmaticsRawOutput:
    """Submit, poll and fetch one transcript for one blob of WAV bytes.

    Shared by the whole-file batch path and the live chunk path, so both spend
    exactly the same request sequence and neither can drift from the other.

    The returned object is the API's own json-v2 body, with the job id and the
    measured turnaround attached beside it under keys the API does not use, so
    nothing we added is mistaken for something the engine said.

    `turnaround_ms` is wall clock from submit to transcript in hand. It is NOT
    comparable to a single-shot engine's latency and must never be printed beside
    one unlabelled: it includes queue time on Speechmatics' side plus up to one
    SPEECHMATICS_POLL_INTERVAL_SEC of our own polling granularity. The surfaces
    label every figure with the transport that produced it, and this engine's
    transport is a job queue.
    """
    settings = get_settings()
    if not settings.speechmatics_api_key:
        raise SpeechmaticsError(
            f"{ENGINE} is not configured -- set SPEECHMATICS_API_KEY in .env"
        )

    base = settings.speechmatics_url.rstrip("/")
    headers = {"Authorization": f"Bearer {settings.speechmatics_api_key}"}
    started = time.perf_counter()

    with httpx.Client(timeout=settings.speechmatics_timeout_sec) as client:
        submitted = _check(
            client.post(
                f"{base}/jobs",
                headers=headers,
                files={"data_file": (filename, wav_bytes, "audio/wav")},
                # Must be sent as an application/json PART, not a plain form
                # field: the API reads it as a typed part.
                data={"config": json.dumps(transcription_config(settings))},
            )
        )
        job_id = submitted.get("id")
        if not job_id:
            raise SpeechmaticsError(f"{ENGINE} submit returned no job id: {submitted}")

        # A 201 above says nothing about whether the config was accepted
        # (finding 3), so the outcome is read from the job status here.
        deadline = started + settings.speechmatics_timeout_sec
        status = "running"
        while time.perf_counter() < deadline:
            job = _check(client.get(f"{base}/jobs/{job_id}", headers=headers))
            status = (job.get("job") or {}).get("status", "running")
            if status in _TERMINAL:
                break
            time.sleep(settings.speechmatics_poll_interval_sec)
        else:
            raise SpeechmaticsError(
                f"{ENGINE} job {job_id} still {status} after "
                f"{settings.speechmatics_timeout_sec}s"
            )

        if status != _DONE:
            raise SpeechmaticsError(f"{ENGINE} job {job_id} ended as {status!r}")

        transcript = _check(
            client.get(
                f"{base}/jobs/{job_id}/transcript",
                headers=headers,
                params={"format": "json-v2"},
            )
        )

    transcript["job_id"] = job_id
    transcript["turnaround_ms"] = int((time.perf_counter() - started) * 1000)
    return transcript


def transcribe_bytes(wav_bytes: bytes, filename: str = "chunk.wav") -> SpeechmaticsRawOutput:
    """Transcribe one already-short WAV -- the live chunk path.

    One chunk is one whole JOB. Nothing is batched across chunks and no session
    is held open, because the batch API has no such concept; see the module
    docstring for what that means for the reported latency.
    """
    return _run_job(wav_bytes, filename)


def run(audio_path: str) -> SpeechmaticsRawOutput:
    """Transcribe `audio_path` in one whole-file job."""
    send_path, cleanup = ensure_canonical_wav(audio_path)
    try:
        with open(send_path, "rb") as fh:
            return _run_job(fh.read(), os.path.basename(send_path))
    finally:
        if cleanup:
            try:
                os.unlink(cleanup)
            except OSError:
                pass
