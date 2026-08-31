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

Measured 2026-08-31 against the running container, so the numbers below are
traceable rather than assumed:

  * Endpoint tells the truth: `application/json`, body `{"text", "usage"}`,
    `usage` is `{"type": "duration", "seconds": N}` (audio seconds, not tokens).
  * Deterministic at `temperature=0` — the same chunk posted twice came back
    byte-identical, which is what makes a re-run comparable at all.
  * Fast: 130 ms mean for 3 s of audio (RTF ~0.043), 386-707 ms for a 54 s file.
  * `max_model_len` is 1024 and a 54 s file fits, so nothing here splits audio
    the way the inception runner has to.
  * THERE IS NO AUTO-DETECT, and this is the one thing to know before touching
    `language`. `language=auto` is rejected outright:

        400 "Unsupported language: 'auto'. Must be one of
        ['en','fr','de','es','pt','it','nl','pl','el','ar','ko','ja','vi','zh']"

    Omitting the field makes the model settle on ONE language from the content.
    On mostly-Arabic audio it picks Arabic; on speech that code-switches WITHIN
    sentences -- this surface's Mixed 50/50 scripts -- it picks English and
    TRANSLATES the Arabic away, returning zero Arabic characters at every
    duration tested up to 45 s. Sending "ar" is what makes that reliable, and it
    code-switches rather than forcing everything Arabic. See
    `cohere_transcribe_language` in settings for the full measurement, including
    the one case where "ar" costs something.

Configuration (`.env`, read via `packages/config/settings.py`):
  COHERE_TRANSCRIBE_URL          — base URL of the container's vLLM server
  COHERE_TRANSCRIBE_TIMEOUT_SEC  — request timeout
  COHERE_TRANSCRIBE_LANGUAGE     — the decoder language, "ar" by default; empty
                                   is NOT auto-detect, it emits English
"""

import os
import time
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
    # The language becomes a `<|xx|>` decoder prompt the model obeys over the
    # audio. Omitting it is NOT auto-detect -- this endpoint has none, and an
    # absent field makes the model settle on one language from the content, which
    # on code-switched speech is English (translating the Arabic away).
    # `cohere_transcribe_language` therefore defaults to "ar"; the guard stays
    # only so a host CAN blank it deliberately.
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


def transcribe_bytes(wav_bytes: bytes, filename: str = "chunk.wav") -> dict[str, Any]:
    """Transcribe one already-short WAV, with its measured latency attached.

    The LIVE chunk path's entry point, mirroring the inception runner's function
    of the same name and returning the same `{"response", "latency_ms"}` entry
    shape the live session stores.

    Chunk length is NOT what decides whether Arabic works here -- the language
    field is (see the module docstring: this endpoint has no auto-detect, and an
    absent field emits English). With the default "ar" these chunks return proper
    Arabic, measured at 463 Arabic characters over 22 x 3 s pieces.

    The one thing chunking does cost: on PURE-ENGLISH audio, "ar" leaks Arabic
    into 6 of 18 chunks, where the same setting is clean on a whole file. So an
    English-only script read live through this route carries a small self-
    inflicted error rate that its batch counterpart does not.
    """
    settings = get_settings()
    data = {
        "model": SERVED_MODEL_NAME,
        "response_format": "json",
        "temperature": "0",
    }
    if settings.cohere_transcribe_language:
        data["language"] = settings.cohere_transcribe_language

    # Timed around the HTTP call only -- not the settings read, not the caller's
    # WAV framing. The scorecard attributes this number to the engine.
    started = time.perf_counter()
    try:
        response = httpx.post(
            f"{settings.cohere_transcribe_url}/v1/audio/transcriptions",
            files={"file": (filename, wav_bytes, "audio/wav")},
            data=data,
            timeout=settings.cohere_transcribe_timeout_sec,
        )
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"cohere-transcribe is not reachable at {settings.cohere_transcribe_url} — "
            "start it with `docker compose up -d cohere-transcribe`"
        ) from exc
    latency_ms = int((time.perf_counter() - started) * 1000)

    # Same reasoning as `run`: vLLM puts the useful part in the body, and httpx's
    # own message drops it.
    if response.status_code >= 400:
        raise RuntimeError(f"cohere-transcribe {response.status_code}: {response.text[:500]}")
    return {"response": response.json(), "latency_ms": latency_ms}


def text_of(part: dict[str, Any]) -> str:
    """The transcript text of one chunk entry. The one place that knows where
    inside an entry this engine's text lives."""
    return ((part.get("response") or {}).get("text") or "").strip()
