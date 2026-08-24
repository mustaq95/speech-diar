"""Translates TryHamsa TTS's native response into a `TtsRender`.

Native shape: `runner.run`'s dict, `{"audio": <headerless PCM16 bytes>,
"status_code", "headers", "synth_ms", "first_audio_ms", "voice"}`. The audio
carries no container of its own — it is exactly the raw frames the endpoint
streamed, at `settings.hamsa_tts_sample_rate` (an assumption; see the
runner's docstring). This adapter is the only code that wraps those frames
in a real RIFF/WAVE header, built with stdlib `wave` rather than hand-rolled
bytes, so a WAV reader downstream (ffprobe, the browser) sees a valid file.

`status_code`/`headers` are not audio and are kept alongside it in
`raw_meta` for the caller to persist, mirroring how the STT side keeps a
`raw_output` blob distinct from what got reduced into the unified contract.
"""

import io
import wave
from typing import Any

from packages.config.settings import get_settings

from .. import TtsRender


def adapt(raw: dict[str, Any]) -> TtsRender:
    settings = get_settings()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(settings.hamsa_tts_sample_rate)
        wav.writeframes(raw["audio"])

    return TtsRender(
        audio=buffer.getvalue(),
        audio_format="wav",
        synth_ms=raw["synth_ms"],
        first_audio_ms=raw["first_audio_ms"],
        voice=raw["voice"],
        raw_meta={"status_code": raw["status_code"], "headers": raw["headers"]},
    )
