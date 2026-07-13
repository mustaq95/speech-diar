"""VibeVoice-ASR (Microsoft) -- 8B decoder-only model performing ASR, speaker
diarization, and timestamping jointly in a single autoregressive pass.

Execution code only; returns the NATIVE payload (the model's own segment
dicts, including the `Content` transcript this platform's contract has no
field for) untouched -- see `adapter.py` for the only code allowed to
understand that shape.

Two execution paths, selected by configuration (`.env`, read via
`packages/config/settings.py`):

  Local (default):
    VIBEVOICE_URL -- base URL of the container's OpenAI-compatible vLLM
    server (POST {url}/v1/chat/completions); the container is
    GPU-supervisor-managed (residency cap, cold start, idle unload -- see
    apps/background_worker/supervisor/) and runs Microsoft's official vLLM
    deployment (deploy/vibevoice/), not plain transformers.generate().
  Remote (VIBEVOICE_BASEURL set):
    VIBEVOICE_BASEURL -- OpenAI-style chat-completions proxy fronting the
    same model off-host; audio goes up base64-encoded as an `input_audio`
    content part, and the generated text comes back in
    choices[0].message.content. When this is set, vibevoice is EXCLUDED
    from the GPU supervisor entirely (see supervisor/registry.py) -- no
    local container, no GPU slot, exactly like the cloud-lane models.
    VIBEVOICE_API_KEY -- bearer token for the proxy, if it needs one.
    VIBEVOICE_SSL_VERIFY -- TLS verification for the proxy call, default
    true; set false if the proxy sits behind a cert this host doesn't
    trust.

  VIBEVOICE_TIMEOUT_SEC -- request timeout, both paths. vLLM's continuous
  batching decodes long audio well under realtime, but the timeout is kept
  wide as headroom for a cold model or an unusually long meeting.
"""

import base64
import json
import re
from contextlib import suppress
from typing import Any
from wave import open as wave_open

import httpx

from packages.config.settings import get_settings

from ..base_model import ModelRunner

VibeVoiceRawOutput = dict[str, Any]

# Both the local (vLLM container) and remote (proxy) paths hand back the
# model's raw generated text over an OpenAI-compatible chat-completions
# response, so this one parser covers both.
_OBJECT_RE = re.compile(r"\{[^{}]*\}")
_START_RE = re.compile(r'"Start"\s*:\s*(-?\d+(?:\.\d+)?)')
_END_RE = re.compile(r'"End"\s*:\s*(-?\d+(?:\.\d+)?)')
_SPEAKER_RE = re.compile(r'"Speaker"\s*:\s*(\d+)')


def _parse_segments(raw_text: str) -> list[dict] | None:
    """Recover the segment list from the model's generated text. VibeVoice
    emits a string that *resembles* JSON and does not always parse (an
    apostrophe in the transcribed `Content` is enough) -- strict JSON first,
    then a regex fallback limited to the numeric fields the diarization
    contract needs. Nothing is invented: every number returned was emitted
    by the model."""
    body = raw_text.split("<|im_start|>assistant", 1)[-1]
    for marker in ("<|im_end|>", "<|endoftext|>"):
        body = body.split(marker, 1)[0]
    body = body.strip()

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        pass
    else:
        return parsed if isinstance(parsed, list) else None

    segments: list[dict] = []
    for match in _OBJECT_RE.finditer(body):
        chunk = match.group(0)
        start = _START_RE.search(chunk)
        end = _END_RE.search(chunk)
        if not (start and end):
            continue
        segment: dict = {"Start": float(start.group(1)), "End": float(end.group(1))}
        # Absent on non-speech events ([Silence], [Music], ...) -- leave it
        # out rather than inventing a speaker for them.
        speaker = _SPEAKER_RE.search(chunk)
        if speaker:
            segment["Speaker"] = int(speaker.group(1))
        segments.append(segment)

    return segments or None


def _wav_duration_sec(path: str) -> float | None:
    with suppress(Exception):
        with wave_open(path, "rb") as wav:
            return wav.getnframes() / wav.getframerate()
    return None


class VibeVoiceRunner(ModelRunner[VibeVoiceRawOutput]):
    model_id = "vibevoice"

    def run(self, audio_path: str, params: dict[str, Any] | None = None) -> VibeVoiceRawOutput:
        settings = get_settings()
        if settings.vibevoice_baseurl:
            return self._run_remote(audio_path, settings)
        return self._run_local(audio_path, settings)

    def _run_local(self, audio_path: str, settings: Any) -> VibeVoiceRawOutput:
        duration = _wav_duration_sec(audio_path)
        if duration is None:
            raise RuntimeError(f"could not read WAV duration for {audio_path}")
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("ascii")
        response = httpx.post(
            f"{settings.vibevoice_url}/v1/chat/completions",
            json={
                "model": "vibevoice",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that transcribes audio "
                        "input into text output in JSON format.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "audio_url",
                                "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"},
                            },
                            {
                                "type": "text",
                                "text": f"This is a {duration:.2f} seconds audio, please "
                                "transcribe it with these keys: Start time, End time, "
                                "Speaker ID, Content",
                            },
                        ],
                    },
                ],
                "max_tokens": 32768,
                "temperature": 0.0,
                "top_p": 1.0,
            },
            timeout=settings.vibevoice_timeout_sec,
        )
        response.raise_for_status()
        raw_text = response.json()["choices"][0]["message"]["content"]
        parsed = _parse_segments(raw_text)
        if parsed is None:
            raise RuntimeError(f"local vibevoice output did not parse as segments: {raw_text!r}")
        return {"audio_duration_sec": duration, "segments": parsed}

    def _run_remote(self, audio_path: str, settings: Any) -> VibeVoiceRawOutput:
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("ascii")
        headers = {}
        if settings.vibevoice_api_key:
            headers["Authorization"] = f"Bearer {settings.vibevoice_api_key}"
        response = httpx.post(
            settings.vibevoice_baseurl,
            headers=headers,
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                        ],
                    }
                ],
            },
            timeout=settings.vibevoice_timeout_sec,
            verify=settings.vibevoice_ssl_verify,
        )
        response.raise_for_status()
        body = response.json()
        raw_text = body["choices"][0]["message"]["content"]
        parsed = _parse_segments(raw_text)
        if parsed is None:
            raise RuntimeError(f"remote vibevoice output did not parse as segments: {raw_text!r}")
        return {
            "audio_duration_sec": _wav_duration_sec(audio_path),
            "segments": parsed,
        }
