"""TryHamsa TTS (new) — execution code, reached through the LiteLLM gateway.

The same vendor as `../hamsa/`, reached a completely different way, and that
difference is the whole point of running both: `../hamsa/` posts to the
inference pod directly and gets back headerless PCM16 at a rate nothing in the
response states, while this one posts to the gateway's OpenAI-compatible
`/v1/audio/speech` and gets back a real RIFF/WAVE container ffprobe can
measure. Probed: the pod accepts `Zeina` and `Amir` too, so these are not two
voice sets and very likely not two models — what the scorecard compares here
is two PIPELINES, exactly as the STT side compares stream against chunks.

Returns the gateway's native bytes untouched; only `adapter.py` reads their
container.

Everything below was measured against the live gateway on 2026-09-09, and
several findings contradict the vendor's own guide. The probe wins:

  * **The model id is `hamsa-tts-new`, and reaching it is about the KEY.**
    The key this repo shipped with answered 403 `team not allowed to access
    model` for that id and listed 10 models, which first read as "the id is
    wrong" -- it was not. The key now in .env lists 14 including this one and
    is a strict superset of the old one. `hamsa-tts` is the older deployment
    and still resolves, but the cloned customer voices exist ONLY on
    `hamsa-tts-new`, so a 403 here means the credential regressed rather than
    that the model moved.
  * **`Content-Type` is always `audio/mpeg`**, even though the body is a real
    RIFF/WAVE — the identical lie `../inception/` documents on this same
    gateway. Never branch on it; the adapter sniffs magic bytes.
  * **`response_format` is ignored.** Asking for `mp3` returned real WAV
    anyway, so nothing here offers the choice and the adapter requires WAV.
  * **It does not meaningfully stream**, despite being consumed with
    `client.stream()`. First byte landed at 97-99% of total across three input
    sizes (524ms / 8.96s / 36.7s totals; ratios 0.973 / 0.979 / 0.989) — even
    for a 166-second clip, the gateway buffers the whole body upstream and then
    flushes it. That is why this engine is registered `delivery="single"`, and
    why a `first_audio_ms` of ~= `synth_ms` here is the correct measurement
    rather than a bug. Synthesis RTF measured ~0.22.
  * **The RIFF and data size fields are `0xFFFFFFFF`** (a streaming-style
    header from the backend). ffprobe, `file` and browsers read it correctly;
    Python's `wave` reports 2147483647 frames, i.e. 134217 seconds. See the
    sentinel guard in `packages/audio/probe_audio`.
  * **An unknown voice is a 500, not the documented 400.** The gateway
    swallows the vendor's `speaker_not_found` and answers
    `{"error":{"message":"Internal server error"}}`, so a 500 from here cannot
    be distinguished from a real outage. All 125 shipped voices (113 bundled +
    12 cloned) were swept against this model and every one returned real WAV,
    so the dropdown cannot produce it -- but the CLONES only resolve on
    `hamsa-tts-new`, and they 500 on `hamsa-tts` while built-ins on the same
    call succeed.
  * **One bundled voice returns near-silence for short Arabic input.** `Ali`
    answers 200 with a 684-byte body (0.02s of PCM) for a 1-to-5-word Arabic
    phrase, reproducibly, while returning a normal 3.3s render for a 9-word
    one; `Zeina` handles the identical short inputs fine. Nothing here filters
    it: a duration floor would be exactly the kind of heuristic this repo
    keeps out of the audio path, and the 0.02s clip IS what the engine
    produced. It surfaces as a 0:00 clip with a large RTF, which is the honest
    reading of a failed render.
  * **Empty input is a 200 carrying a 44-byte header-only WAV**, not the
    documented validation error. The adapter rejects a frameless body rather
    than store an unplayable clip.
  * Identical input does NOT give identical output — the model samples, so byte
    length and duration vary run to run (52080 vs 62022 bytes for one
    sentence). No test can pin exact bytes.

Configuration (`.env`, read via `packages/config/settings.py`):
  LITELLM_BASE_URL / LITELLM_API_KEY  — gateway endpoint and credential
                                         (shared with inception-stt/-tts)
  TTS_SPEECH_PATH                     — path on that gateway
  HAMSA_TTS_NEW_MODEL                 — gateway model id (see above)
  HAMSA_TTS_NEW_VOICE                 — comma list, entry 0 is the default
  HAMSA_TTS_NEW_TIMEOUT_SEC           — request timeout
  VERIFY_SSL / LITELLM_CA_BUNDLE      — TLS for this gateway only
"""

import threading
import time
from typing import Any

import httpx

from packages.config.settings import Settings, get_settings

#: The container this engine asks for and the only one it will accept back.
#: Not a setting: probed, the gateway ignores the field entirely and returns
#: WAV regardless, so offering the choice would be offering a lie.
RESPONSE_FORMAT = "wav"

#: Pooled client, same reasoning as `../inception/runner.py`'s `_http_client`:
#: a fresh client per call re-pays the TLS handshake, and that cost would be
#: charged to the engine's measured synthesis time.
_client: httpx.Client | None = None
_client_verify: object = object()  # sentinel: no client built yet
_client_lock = threading.Lock()


def _http_client(settings: Settings) -> httpx.Client:
    global _client, _client_verify
    verify = settings.litellm_httpx_verify
    with _client_lock:
        if _client is None or _client_verify != verify:
            if _client is not None:
                _client.close()
            _client = httpx.Client(verify=verify)
            _client_verify = verify
        return _client


class HamsaNewTtsError(RuntimeError):
    """Base for failures talking to the TryHamsa TTS (new) gateway."""


class HamsaNewTtsTimeoutError(HamsaNewTtsError):
    """The gateway did not answer within HAMSA_TTS_NEW_TIMEOUT_SEC."""


class HamsaNewTtsConnectionError(HamsaNewTtsError):
    """The gateway was unreachable (DNS / network / TLS)."""


class HamsaNewTtsUpstreamError(HamsaNewTtsError):
    """The gateway answered with a non-2xx status.

    A 500 here is most often an unknown voice: the gateway turns the vendor's
    `speaker_not_found` 400 into an opaque internal error, so the message is
    worth carrying verbatim to the operator.
    """

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"hamsa-tts-new {status_code}: {body[:500]}")
        self.status_code = status_code
        self.body = body


class HamsaNewTtsResponseError(HamsaNewTtsError):
    """The gateway answered 2xx with a body that was not the expected audio."""


def run(text: str, voice: str) -> dict[str, Any]:
    """Synthesize `text` with `voice`, returning the gateway's native bytes
    and headers untouched, alongside what was measured."""
    settings = get_settings()
    if not settings.tts_speech_url or not settings.litellm_api_key:
        raise HamsaNewTtsError(
            "hamsa-tts-new is not configured — set LITELLM_BASE_URL and LITELLM_API_KEY in .env"
        )

    client = _http_client(settings)
    payload = {
        "model": settings.hamsa_tts_new_model,
        "input": text,
        "voice": voice,
        "response_format": RESPONSE_FORMAT,
        # No "speed" or "expressiveness". Both are accepted by the gateway
        # (probed), but nothing in this UI chooses them and a default sent
        # explicitly is indistinguishable from one the vendor picked — leaving
        # them out keeps the clip the engine's own rendering.
    }

    chunks: list[bytes] = []
    first_audio_ms: int | None = None
    # Brackets the HTTP call ONLY. Nothing here writes to storage or shells out
    # to ffprobe: that IO happens in the route, and charging it to the vendor
    # would make this figure something other than synthesis time.
    started = time.perf_counter()
    try:
        with client.stream(
            "POST",
            settings.tts_speech_url,
            headers={"Authorization": f"Bearer {settings.litellm_api_key}"},
            json=payload,
            timeout=settings.hamsa_tts_new_timeout_sec,
        ) as response:
            if response.status_code >= 400:
                response.read()
                raise HamsaNewTtsUpstreamError(response.status_code, response.text)
            response_headers = dict(response.headers)
            status_code = response.status_code
            for chunk in response.iter_bytes():
                if not chunk:
                    continue
                if first_audio_ms is None:
                    first_audio_ms = int((time.perf_counter() - started) * 1000)
                chunks.append(chunk)
    except httpx.TimeoutException as exc:
        raise HamsaNewTtsTimeoutError(
            f"hamsa-tts-new did not respond within {settings.hamsa_tts_new_timeout_sec}s"
        ) from exc
    except httpx.ConnectError as exc:
        raise HamsaNewTtsConnectionError(
            f"hamsa-tts-new is not reachable at {settings.tts_speech_url}: {exc}"
        ) from exc
    except httpx.HTTPError as exc:
        raise HamsaNewTtsConnectionError(f"hamsa-tts-new request failed: {exc}") from exc

    synth_ms = int((time.perf_counter() - started) * 1000)

    return {
        "audio": b"".join(chunks),
        "status_code": status_code,
        "headers": response_headers,
        "synth_ms": synth_ms,
        "first_audio_ms": first_audio_ms,
        "voice": voice,
    }
