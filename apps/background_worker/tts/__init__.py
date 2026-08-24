"""TTS synthesis engines: script text -> read-aloud audio.

A sibling tree to `../transcription/`, not a branch of it: STT turns audio
into text, this turns text into audio, and the two are compared by different
axes. STT engines are labelled by `transport` (`stream` vs `chunks`); these
engines are labelled by `delivery` (`stream` vs `single`) — a deliberately
different name, because neither vocabulary describes the other direction
correctly. Hamsa's TTS stream is a single POST whose BODY streams (not a
persistent socket like its STT side), and Inception-TTS's "single" response
still arrives over `transfer-encoding: chunked` with no meaningful
front-loading — measured ~6.7s wall clock, whole body buffered, before any
byte is visible.

Same runner/adapter split as everywhere else in this repo: each engine's
`runner.py` executes it and returns its NATIVE output untouched; its
`adapter.py` is the only code allowed to understand that shape, and it
reduces to `TtsRender` below. No abstract base class: two single-use engines
do not need one.
"""

from dataclasses import dataclass
from typing import Any, Callable

from packages.config.settings import Settings


@dataclass(frozen=True)
class TtsRender:
    """The unified shape every engine's adapter produces: real, playable audio
    plus the measurements taken around it. `raw_meta` (status code, response
    headers) is not audio and travels separately, mirroring how the STT side
    keeps `raw_output` distinct from what got reduced into a transcript."""

    audio: bytes
    audio_format: str
    synth_ms: int
    first_audio_ms: int | None
    voice: str
    raw_meta: dict[str, Any]


@dataclass(frozen=True)
class TtsEngine:
    """One TTS engine: what it is called, how it delivers audio, and how to
    run it."""

    tts_id: str
    name: str
    #: "stream" | "single" -- see the module docstring for why this is not
    #: the STT side's "transport" vocabulary.
    delivery: str
    #: (text, voice) -> the engine's NATIVE output
    run: Callable[[str, str], Any]
    #: native output -> TtsRender
    adapt: Callable[[Any], TtsRender]
    #: The voice dropdown's options for this engine.
    voices: Callable[[Settings], list[str]]
    #: The voice used when the caller does not choose one.
    default_voice: Callable[[Settings], str]
    #: Is this engine actually configured on this host? Used to decide
    #: whether to offer it at all, so an unconfigured host shows "not
    #: configured" rather than a guaranteed failure on every run.
    configured: Callable[[Settings], bool]
    #: The request-shaping settings this engine will actually use, as plain
    #: data for the UI to display beside the voice picker. Engine-level and
    #: therefore true of EVERY voice this engine offers -- which is why it can
    #: sit under a voice name without claiming to be a property of that voice.
    #: Deliberately not a pre-formatted string: the backend does not choose
    #: the separator or the casing.
    synthesis_params: Callable[[Settings], dict[str, str]]


# Imported here, after TtsRender/TtsEngine are defined: each adapter imports
# TtsRender from this package, so importing them earlier would be circular.
from .hamsa import adapter as hamsa_adapter  # noqa: E402
from .hamsa import runner as hamsa_runner  # noqa: E402
from .inception import adapter as inception_adapter  # noqa: E402
from .inception import runner as inception_runner  # noqa: E402

TTS_ENGINES: dict[str, TtsEngine] = {
    engine.tts_id: engine
    for engine in (
        TtsEngine(
            tts_id="hamsa-tts",
            name="TryHamsa TTS",
            delivery="stream",
            run=hamsa_runner.run,
            adapt=hamsa_adapter.adapt,
            voices=lambda s: s.hamsa_tts_speaker_options,
            default_voice=lambda s: s.hamsa_tts_default_speaker,
            # The bearer is NOT optional: measured, omitting it returns
            # 401 {"detail":"Authorization header is missing"} even with a
            # valid X-API-Key. Leaving it out of this predicate advertised a
            # host as configured and then failed every single run.
            configured=lambda s: bool(
                s.hamsa_tts_api_url and s.hamsa_tts_key and s.hamsa_tts_bearer_token
            ),
            synthesis_params=lambda s: {
                "language": s.hamsa_tts_language_id,
                "dialect": s.hamsa_tts_dialect,
            },
        ),
        TtsEngine(
            tts_id="inception-tts",
            name="Inception-TTS",
            delivery="single",
            run=inception_runner.run,
            adapt=inception_adapter.adapt,
            voices=lambda s: s.inception_tts_voice_options,
            default_voice=lambda s: s.inception_tts_default_voice,
            configured=lambda s: bool(s.litellm_base_url and s.litellm_api_key),
            synthesis_params=lambda s: {
                "model": s.inception_tts_model,
                "format": s.inception_tts_response_format,
            },
        ),
    )
}

#: Engines the TTS comparison surface runs, in display order. An explicit
#: tuple, not "every configured engine" -- mirrors COMPARISON_ASR_IDS's own
#: reasoning: silently gaining a column because someone set a credential
#: would change what the scorecard means.
COMPARISON_TTS_IDS: tuple[str, ...] = ("hamsa-tts", "inception-tts")

#: Which delivery each engine uses. Kept alongside the registry rather than
#: read off `TtsEngine.delivery` everywhere, mirroring `ASR_TRANSPORTS` /
#: `transport_for` on the STT side.
TTS_DELIVERY: dict[str, str] = {"hamsa-tts": "stream", "inception-tts": "single"}


def delivery_for(tts_id: str) -> str | None:
    """The delivery mode an engine uses, or None for an unregistered id."""
    return TTS_DELIVERY.get(tts_id)


def engine_for(tts_id: str) -> TtsEngine | None:
    return TTS_ENGINES.get(tts_id)
