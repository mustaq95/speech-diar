"""Translates Inception-TTS's native response into a `TtsRender`.

Native shape: `runner.run`'s dict, `{"audio": <real container bytes>,
"status_code", "headers", "synth_ms", "first_audio_ms", "voice"}`. Unlike
Hamsa's, this audio is already a real container (confirmed by ffprobe: real
RIFF/WAVE for a "wav" request) and is passed through verbatim — no wrapping.

The gateway's `Content-Type` header is never trusted (see the runner's
docstring: it always claims `audio/mpeg`, even for a real WAV body). This
adapter sniffs the actual container from magic bytes instead, and treats a
mismatch against the requested `response_format` as a real error, not
something to silently relabel — a caller that asked for WAV and quietly got
MP3 back would otherwise persist a WAV extension on non-WAV bytes.
"""

from typing import Any

from packages.config.settings import get_settings

from .. import TtsRender
from .runner import InceptionTtsResponseError


def _sniff_format(audio: bytes) -> str | None:
    """The real container of `audio`, from its magic bytes, or None if
    neither WAV nor MP3 is recognized."""
    if len(audio) >= 12 and audio[:4] == b"RIFF" and audio[8:12] == b"WAVE":
        return "wav"
    if audio[:3] == b"ID3" or audio[:2] in (b"\xff\xfb", b"\xff\xf3"):
        return "mp3"
    return None


def adapt(raw: dict[str, Any]) -> TtsRender:
    settings = get_settings()
    audio = raw["audio"]
    sniffed = _sniff_format(audio)
    expected = settings.inception_tts_response_format
    if sniffed != expected:
        raise InceptionTtsResponseError(
            f"inception-tts requested response_format={expected!r} but returned "
            f"a body sniffed as {sniffed!r} ({len(audio)} bytes)"
        )

    return TtsRender(
        audio=audio,
        audio_format=sniffed,
        synth_ms=raw["synth_ms"],
        first_audio_ms=raw["first_audio_ms"],
        voice=raw["voice"],
        raw_meta={"status_code": raw["status_code"], "headers": raw["headers"]},
    )
