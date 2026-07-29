"""Cohere Transcribe Arabic — execution code for the OFFLINE transcription mode.

CohereLabs/cohere-transcribe-arabic-07-2026: a 2B Arabic/English speech-to-text
model served on vLLM's OpenAI-compatible transcription API, in a container on
this host (see `deploy/cohere-transcribe/`). Audio never leaves the machine,
which is the point of offline mode.

Returns vLLM's native JSON untouched; only `adapter.py` reads that shape.

Structurally the same as
`apps/background_worker/models/moss_transcribe/runner.py`, which serves a
different model through the same vLLM endpoint — including its hard-won
error handling (see `_post` below).

Unlike every diarization container, this one is NOT GPU-supervisor-managed: it
holds no residency slot and nothing starts it on demand. If it is not running,
the request fails immediately with a message naming the endpoint, which is the
honest outcome — fabricating a cold-start wait for a container this code has no
authority to start would just hide the real problem.

Configuration (`.env`, read via `packages/config/settings.py`):
  COHERE_TRANSCRIBE_URL          — base URL of the container's vLLM server
  COHERE_TRANSCRIBE_TIMEOUT_SEC  — request timeout
  COHERE_TRANSCRIBE_LANGUAGE     — force a language ("ar"/"en"/...); empty = auto-detect
"""

import os
from contextlib import suppress
from typing import Any

import httpx

from packages.audio import ensure_canonical_wav
from packages.config.settings import get_settings

#: vLLM's native transcription response: {"text": ..., "usage": {...}}
CohereRawOutput = dict[str, Any]

#: Matches --served-model-name in
#: deploy/cohere-transcribe/docker-compose.cohere-transcribe.yml, so repointing
#: COHERE_TRANSCRIBE_MODEL_ID at another checkpoint does not break this call.
SERVED_MODEL_NAME = "cohere-transcribe"


def run(audio_path: str) -> CohereRawOutput:
    """Transcribe `audio_path` via the local container; return vLLM's JSON."""
    settings = get_settings()
    send_path, cleanup = ensure_canonical_wav(audio_path)
    data = {
        "model": SERVED_MODEL_NAME,
        "response_format": "json",
        # Deterministic: this is transcription, not generation, and a re-run
        # must reproduce the same transcript for a timing comparison to mean
        # anything.
        "temperature": "0",
    }
    # A forced language becomes a `<|xx|>` decoder prompt the model obeys over
    # the audio: forcing "ar" makes this Arabic-first model hallucinate Arabic
    # filler at the start of English recordings (and forcing "en" would break
    # Arabic ones). Omitting it lets the model self-detect, which is correct for
    # both. Only send it when explicitly set to force a single-language batch.
    if settings.cohere_transcribe_language:
        data["language"] = settings.cohere_transcribe_language
    try:
        with open(send_path, "rb") as fh:
            try:
                response = httpx.post(
                    f"{settings.cohere_transcribe_url}/v1/audio/transcriptions",
                    files={"file": (os.path.basename(send_path), fh, "audio/wav")},
                    data=data,
                    timeout=settings.cohere_transcribe_timeout_sec,
                )
            except httpx.ConnectError as exc:
                raise RuntimeError(
                    f"cohere-transcribe is not reachable at {settings.cohere_transcribe_url} — "
                    "start it with ./deploy/cohere-transcribe/cohere_transcribe_up.sh "
                    "(this container is not auto-started by the GPU supervisor)"
                ) from exc
        # Not raise_for_status(): httpx renders only "Client error '400 Bad
        # Request' for url ...", while vLLM puts the part that actually matters
        # in the body -- "Maximum file size exceeded", "Invalid or unsupported
        # audio file", "Input length (N) exceeds model's maximum context
        # length". Losing that turns a self-explanatory failure into a
        # `docker logs` expedition; it already did once on moss-transcribe. The
        # message ends up in TranscriptResult.error and on the panel.
        if response.status_code >= 400:
            raise RuntimeError(f"cohere-transcribe {response.status_code}: {response.text[:500]}")
        return response.json()
    finally:
        if cleanup:
            with suppress(OSError):
                os.unlink(cleanup)
