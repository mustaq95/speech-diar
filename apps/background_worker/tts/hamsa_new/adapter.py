"""Translates TryHamsa TTS (new)'s native response into a `TtsRender`.

Native shape: `runner.run`'s dict, `{"audio": <real container bytes>,
"status_code", "headers", "synth_ms", "first_audio_ms", "voice"}`.

Unlike `../hamsa/`'s adapter, which has to BUILD a WAV header around
headerless PCM at a declared rate, this engine already returns a real
container and the audio is passed through verbatim — so its sample rate is a
measurement (ffprobe reads it off these bytes) rather than an assumption, and
nothing downstream needs to label it "(assumed)".

Two checks, both from the probe (see the runner's docstring), and both of
which store nothing rather than store something misleading:

  * The container is sniffed from magic bytes and must be WAV. The gateway's
    `Content-Type` always claims `audio/mpeg` and is never consulted; and
    since the gateway ignores `response_format` entirely, anything that is not
    a RIFF/WAVE here means the contract changed rather than that a different
    format was requested.
  * A body carrying no PCM frames is an error. Empty input returns 200 with a
    44-byte header-only WAV, which would otherwise be stored as a real clip
    whose duration renders as 0 — a figure that reads as a measurement.
"""

from typing import Any

from .. import TtsRender
from .runner import HamsaNewTtsResponseError


def _is_wav(audio: bytes) -> bool:
    """True when `audio` really is a RIFF/WAVE container, by magic bytes.

    Deliberately does not look at the declared sizes at bytes 4-8: this
    backend writes `0xFFFFFFFF` there (measured), so they say nothing about
    the real length.
    """
    return len(audio) >= 12 and audio[:4] == b"RIFF" and audio[8:12] == b"WAVE"


def _has_frames(audio: bytes) -> bool:
    """True when at least one byte of PCM follows the `data` chunk header.

    Found by locating the chunk rather than comparing against 44, because the
    header's own length depends on which chunks the backend emitted, and the
    declared data size is the `0xFFFFFFFF` sentinel and so unusable.
    """
    marker = audio.find(b"data")
    return marker != -1 and len(audio) > marker + 8


def adapt(raw: dict[str, Any]) -> TtsRender:
    audio = raw["audio"]
    if not _is_wav(audio):
        raise HamsaNewTtsResponseError(
            f"hamsa-tts-new returned {len(audio)} bytes that are not a RIFF/WAVE container "
            f"(first bytes: {audio[:12].hex() or 'none'}) — the gateway's contract has changed"
        )
    if not _has_frames(audio):
        raise HamsaNewTtsResponseError(
            f"hamsa-tts-new returned a WAV header with no audio frames ({len(audio)} bytes); "
            "the gateway answers 200 with a header-only body for empty input"
        )

    return TtsRender(
        audio=audio,
        audio_format="wav",
        synth_ms=raw["synth_ms"],
        first_audio_ms=raw["first_audio_ms"],
        voice=raw["voice"],
        raw_meta={"status_code": raw["status_code"], "headers": raw["headers"]},
    )
