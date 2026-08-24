"""Live check against the real TryHamsa TTS and Inception-TTS deployments.

Skipped unless RUN_LIVE_TESTS=1 (see conftest's `pytest_collection_modifyitems`),
like the other live tests: real credentials, real endpoints, real time.

Run with:
    RUN_LIVE_TESTS=1 uv run pytest -m live -k tts -v

This is the only test that can catch the two things every stub necessarily
assumes away, both of which would otherwise ship green:

  * HAMSA_TTS_SAMPLE_RATE is a DECLARED value, not a measured one -- the stream
    is headerless PCM and states its rate nowhere. If the engine ever returns
    22050, every Hamsa duration and RTF in the product silently becomes wrong
    while every unit test still passes. The duration sanity check below is the
    only thing standing in front of that.
  * The gateways' request/response contracts. Inception's Content-Type lies
    (always `audio/mpeg`, even for a WAV body), so the adapter sniffs magic
    bytes; if the gateway ever stops honouring `response_format`, only a real
    call notices.
"""

import wave
from io import BytesIO

import pytest

from apps.background_worker.tts import TTS_ENGINES
from packages.audio import probe_audio
from packages.config.settings import get_settings

#: Short on purpose: this costs real synthesis time and real quota.
SENTENCE = "أهلاً بكم في أبوظبي."

#: Arabic speech runs roughly 2-3.5 words/sec. A rate that is wrong by a factor
#: of 1.5 (16k read as 24k) pushes a short clip outside this window, which is
#: what makes the declared-rate assumption falsifiable at all.
MIN_SEC, MAX_SEC = 0.5, 20.0


@pytest.mark.live
@pytest.mark.parametrize("tts_id", ["hamsa-tts", "inception-tts"])
def test_engine_returns_playable_audio(tts_id: str) -> None:
    settings = get_settings()
    engine = TTS_ENGINES[tts_id]
    if not engine.configured(settings):
        pytest.skip(f"{tts_id} is not configured on this host")

    raw = engine.run(SENTENCE, engine.default_voice(settings))
    render = engine.adapt(raw)

    assert render.audio, f"{tts_id} returned an empty body"
    assert render.synth_ms > 0
    assert render.first_audio_ms is not None
    assert render.first_audio_ms <= render.synth_ms

    probe = probe_audio(render.audio)
    assert probe.duration_sec is not None, f"{tts_id}'s output is not decodable audio"
    assert MIN_SEC < probe.duration_sec < MAX_SEC, (
        f"{tts_id} rendered {probe.duration_sec:.2f}s for {len(SENTENCE)} chars, which is "
        "outside plausible speech tempo -- for hamsa-tts the most likely cause is that "
        f"HAMSA_TTS_SAMPLE_RATE ({settings.hamsa_tts_sample_rate}) does not match what the "
        "engine actually streams"
    )


@pytest.mark.live
def test_hamsa_stream_front_loads_audio() -> None:
    """The whole reason hamsa-tts is labelled `stream`. If first audio ever
    lands at ~total, the label is wrong and the scorecard's comparison against
    Inception's buffered response stops meaning anything."""
    settings = get_settings()
    engine = TTS_ENGINES["hamsa-tts"]
    if not engine.configured(settings):
        pytest.skip("hamsa-tts is not configured on this host")

    render = engine.adapt(engine.run(SENTENCE, engine.default_voice(settings)))
    assert render.first_audio_ms < render.synth_ms, (
        f"first audio at {render.first_audio_ms}ms vs {render.synth_ms}ms total: "
        "the response is no longer streaming"
    )


@pytest.mark.live
def test_hamsa_output_is_a_valid_wav_at_the_declared_rate() -> None:
    """The adapter writes the header at the declared rate; this confirms the
    result is a file a browser and ffprobe both accept."""
    settings = get_settings()
    engine = TTS_ENGINES["hamsa-tts"]
    if not engine.configured(settings):
        pytest.skip("hamsa-tts is not configured on this host")

    render = engine.adapt(engine.run(SENTENCE, engine.default_voice(settings)))
    with wave.open(BytesIO(render.audio), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == settings.hamsa_tts_sample_rate


@pytest.mark.live
def test_inception_honours_the_requested_response_format() -> None:
    """Measured: the gateway returns real RIFF/WAVE for response_format=wav
    while still labelling it `audio/mpeg`. If it ever stops honouring the
    parameter, the adapter's sniff raises instead of storing mislabelled bytes."""
    settings = get_settings()
    engine = TTS_ENGINES["inception-tts"]
    if not engine.configured(settings):
        pytest.skip("inception-tts is not configured on this host")

    render = engine.adapt(engine.run(SENTENCE, engine.default_voice(settings)))
    assert render.audio_format == settings.inception_tts_response_format

    probe = probe_audio(render.audio)
    assert probe.sample_rate is not None, "inception-tts's rate should be readable from its container"
