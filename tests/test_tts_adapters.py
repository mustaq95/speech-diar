"""Unit coverage for the TTS adapters and the shared audio probe.

The adapters are the only code allowed to understand each engine's native
shape, so these are the tests that pin down what that shape is: Hamsa returns
headerless PCM that must become a real WAV, Inception returns a real container
whose declared Content-Type cannot be trusted (the gateway sends `audio/mpeg`
even for a WAV body -- measured, see the runner docstring), and hamsa-tts-new
returns a real WAV whose DECLARED SIZES are the 0xFFFFFFFF sentinel.
"""

import shutil
import subprocess
import wave
from io import BytesIO

import pytest

from apps.background_worker.tts.hamsa import adapter as hamsa_adapter
from apps.background_worker.tts.hamsa_new import adapter as hamsa_new_adapter
from apps.background_worker.tts.hamsa_new.runner import HamsaNewTtsResponseError
from apps.background_worker.tts.inception import adapter as inception_adapter
from apps.background_worker.tts.inception.runner import InceptionTtsResponseError
from packages.audio import probe_audio
from packages.config.settings import get_settings

MP3_BYTES = b"\xff\xf3\x84\xc4" + b"\x00" * 256
ID3_BYTES = b"ID3\x04\x00\x00" + b"\x00" * 256

#: hamsa-tts-new's REAL 44-byte response header, copied byte for byte off a
#: live call on 2026-09-09: PCM, mono, 16000 Hz, and BOTH size fields set to
#: 0xFFFFFFFF -- the backend writes a streaming-style header and never goes
#: back to fill in the length. A synthetic `wave`-built header would carry
#: real sizes and so would not exercise any of this, which is the whole
#: reason the real bytes are pinned here.
HAMSA_NEW_HEADER = bytes.fromhex(
    "52494646"          # "RIFF"
    "ffffffff"          # riff size: the sentinel, not a length
    "57415645"          # "WAVE"
    "666d7420" "10000000"   # "fmt " chunk, 16 bytes
    "0100" "0100"           # PCM, 1 channel
    "803e0000" "007d0000"   # 16000 Hz, 32000 bytes/sec
    "0200" "1000"           # block align 2, 16 bits
    "64617461" "ffffffff"   # "data" chunk, sentinel size again
)


def _wav_bytes(rate: int = 24000, seconds: float = 0.5) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(b"\x00\x00" * int(rate * seconds))
    return buffer.getvalue()


def _native(audio: bytes, *, first_ms: int = 120, synth_ms: int = 900) -> dict:
    return {
        "audio": audio,
        "status_code": 200,
        "headers": {"content-type": "audio/mpeg"},
        "synth_ms": synth_ms,
        "first_audio_ms": first_ms,
        "voice": "Ruba",
    }


# --- hamsa ---------------------------------------------------------------


def test_hamsa_adapter_wraps_pcm_in_a_real_wav_header() -> None:
    """The stream carries no container at all. Anything downstream (ffprobe, the
    browser, the download) needs a real header, at the DECLARED rate."""
    frames = 8000
    render = hamsa_adapter.adapt(_native(b"\x00\x00" * frames))

    assert render.audio_format == "wav"
    with wave.open(BytesIO(render.audio), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == get_settings().hamsa_tts_sample_rate
        assert wav.getnframes() == frames


def test_hamsa_adapter_changes_not_one_sample() -> None:
    """Wrapping is not transcoding: the frames handed back must be byte-identical
    to what the engine streamed, or the clip is no longer what it produced."""
    pcm = bytes(range(256)) * 8
    render = hamsa_adapter.adapt(_native(pcm))
    with wave.open(BytesIO(render.audio), "rb") as wav:
        assert wav.readframes(wav.getnframes()) == pcm


def test_hamsa_adapter_keeps_metadata_out_of_the_audio() -> None:
    """`raw_meta` is what lands in the JSON column; the audio must never be in it."""
    render = hamsa_adapter.adapt(_native(b"\x00\x00" * 100))
    assert render.raw_meta == {"status_code": 200, "headers": {"content-type": "audio/mpeg"}}
    assert "audio" not in render.raw_meta


# --- inception -----------------------------------------------------------


def test_inception_adapter_sniffs_wav_regardless_of_content_type(monkeypatch) -> None:
    """The gateway labels a real WAV `audio/mpeg`. Trusting that header would
    store WAV bytes under an .mp3 key and serve them mislabelled."""
    monkeypatch.setattr(get_settings(), "inception_tts_response_format", "wav")
    render = inception_adapter.adapt(_native(_wav_bytes()))
    assert render.audio_format == "wav"


@pytest.mark.parametrize("payload", [MP3_BYTES, ID3_BYTES], ids=["frame-sync", "id3"])
def test_inception_adapter_sniffs_mp3(monkeypatch, payload: bytes) -> None:
    monkeypatch.setattr(get_settings(), "inception_tts_response_format", "mp3")
    assert inception_adapter.adapt(_native(payload)).audio_format == "mp3"


def test_inception_adapter_passes_the_container_through_untouched(monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "inception_tts_response_format", "wav")
    payload = _wav_bytes()
    assert inception_adapter.adapt(_native(payload)).audio == payload


def test_inception_adapter_rejects_a_format_mismatch(monkeypatch) -> None:
    """Asking for WAV and quietly getting MP3 would persist a .wav key over mp3
    bytes. Loud failure beats a mislabelled clip."""
    monkeypatch.setattr(get_settings(), "inception_tts_response_format", "wav")
    with pytest.raises(InceptionTtsResponseError):
        inception_adapter.adapt(_native(MP3_BYTES))


def test_inception_adapter_rejects_an_unrecognizable_body(monkeypatch) -> None:
    """A gateway error page returned with a 200 is not audio."""
    monkeypatch.setattr(get_settings(), "inception_tts_response_format", "wav")
    with pytest.raises(InceptionTtsResponseError):
        inception_adapter.adapt(_native(b"<html>gateway error</html>"))


# --- hamsa-tts-new -------------------------------------------------------
#
# Every payload below is the shape the LIVE gateway actually returned on
# 2026-09-09, not what its vendor guide documents. The guide was wrong about
# the status codes, the Content-Type and the honoured response_format, so the
# probe is what these tests encode.


def _hamsa_new_native(audio: bytes) -> dict:
    """The runner's native dict, carrying the gateway's real lying header."""
    return _native(audio) | {"voice": "Zeina"}


def test_hamsa_new_adapter_accepts_the_sentinel_header_wav() -> None:
    """The real body: a RIFF/WAVE whose declared sizes are 0xFFFFFFFF. An
    adapter that validated those fields would reject every real response."""
    render = hamsa_new_adapter.adapt(_hamsa_new_native(HAMSA_NEW_HEADER + b"\x00\x00" * 8000))
    assert render.audio_format == "wav"
    assert render.voice == "Zeina"


def test_hamsa_new_adapter_ignores_the_lying_content_type() -> None:
    """The gateway always says `audio/mpeg`, even for this real WAV -- the same
    lie inception-tts tells on the same gateway. Branching on it would store
    WAV bytes under an .mp3 key."""
    native = _hamsa_new_native(HAMSA_NEW_HEADER + b"\x00\x00" * 100)
    assert native["headers"]["content-type"] == "audio/mpeg"
    assert hamsa_new_adapter.adapt(native).audio_format == "wav"


def test_hamsa_new_adapter_passes_the_container_through_untouched() -> None:
    """Stored verbatim, never canonicalized: the download has to be the bytes
    the engine sent, sentinel header included."""
    payload = HAMSA_NEW_HEADER + bytes(range(256)) * 8
    assert hamsa_new_adapter.adapt(_hamsa_new_native(payload)).audio == payload


def test_hamsa_new_adapter_rejects_a_header_with_no_frames() -> None:
    """Empty input returns 200 and EXACTLY this 44-byte header -- probed, not
    hypothetical. Storing it would put a clip in the UI whose duration renders
    as 0, which reads as a measurement of silence rather than a refusal."""
    with pytest.raises(HamsaNewTtsResponseError, match="no audio frames"):
        hamsa_new_adapter.adapt(_hamsa_new_native(HAMSA_NEW_HEADER))


@pytest.mark.parametrize(
    "payload, label",
    [
        (MP3_BYTES, "mp3"),
        (b"<html>gateway error</html>", "an error page"),
        (b"", "an empty body"),
    ],
)
def test_hamsa_new_adapter_rejects_anything_that_is_not_wav(payload: bytes, label: str) -> None:
    """This gateway IGNORES response_format and returns WAV whatever is asked
    for (probed: asking mp3 got real WAV back). So a non-WAV body here does not
    mean a different format was requested -- it means the contract changed, and
    guessing at it would mislabel the stored clip."""
    with pytest.raises(HamsaNewTtsResponseError):
        hamsa_new_adapter.adapt(_hamsa_new_native(payload))


def test_hamsa_new_adapter_keeps_the_audio_out_of_raw_meta() -> None:
    render = hamsa_new_adapter.adapt(_hamsa_new_native(HAMSA_NEW_HEADER + b"\x00\x00" * 100))
    assert render.raw_meta == {"status_code": 200, "headers": {"content-type": "audio/mpeg"}}
    assert "audio" not in render.raw_meta


# --- the 0xFFFFFFFF sentinel, end to end through probe_audio -------------


def test_probe_reads_the_sentinel_header_wav_via_ffprobe() -> None:
    """ffprobe reads these bytes correctly where `wave` cannot, which is why
    probe_audio tries it first. 8000 frames at 16 kHz is 0.5s."""
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe not available")
    probe = probe_audio(HAMSA_NEW_HEADER + b"\x00\x00" * 8000)
    assert probe.sample_rate == 16000
    assert probe.duration_sec == pytest.approx(0.5, abs=0.05)
    assert probe.channels == 1


def test_wave_fallback_reports_unknown_for_a_sentinel_length() -> None:
    """The guard for a host with no ffprobe installed.

    Python's `wave` reads the 0xFFFFFFFF data size as 2147483647 frames, i.e.
    134217 seconds -- a 37-hour duration for a half-second clip, and an RTF
    derived from it that would look like a real measurement. `duration_sec`
    must come back None so the UI renders the absence instead.

    Forced down the fallback by hiding ffprobe, because on this host the
    ffprobe path above succeeds and this branch would never run.
    """
    payload = HAMSA_NEW_HEADER + b"\x00\x00" * 8000

    # Sanity: this is genuinely what `wave` does with these bytes.
    with wave.open(BytesIO(payload), "rb") as wav:
        assert wav.getnframes() == 0x7FFFFFFF

    import packages.audio as audio_module

    original = audio_module.shutil.which
    try:
        audio_module.shutil.which = lambda name: None if name == "ffprobe" else original(name)
        probe = probe_audio(payload)
    finally:
        audio_module.shutil.which = original

    assert probe.duration_sec is None, "a sentinel length must never be reported as a duration"
    # The fields that ARE readable off the header must survive: the point is to
    # drop the unknowable one, not to discard the whole probe.
    assert probe.sample_rate == 16000
    assert probe.channels == 1
    assert probe.bit_depth == 16


# --- timing invariant ----------------------------------------------------


@pytest.mark.parametrize(
    "adapt, payload",
    [
        (hamsa_adapter.adapt, b"\x00\x00" * 4000),
        (hamsa_new_adapter.adapt, HAMSA_NEW_HEADER + b"\x00\x00" * 4000),
        (inception_adapter.adapt, _wav_bytes()),
    ],
    ids=["hamsa", "hamsa-new", "inception"],
)
def test_first_audio_never_exceeds_total_synthesis(monkeypatch, adapt, payload: bytes) -> None:
    """First audio is measured inside the same wall-clock bracket as the total,
    so one cannot outrun the other. A violation means the two are being timed
    against different clocks."""
    monkeypatch.setattr(get_settings(), "inception_tts_response_format", "wav")
    render = adapt(_native(payload, first_ms=120, synth_ms=900))
    assert render.first_audio_ms is not None
    assert render.first_audio_ms <= render.synth_ms


# --- probe_audio ---------------------------------------------------------


def test_probe_reads_a_real_wav() -> None:
    probe = probe_audio(_wav_bytes(rate=24000, seconds=0.5))
    assert probe.sample_rate == 24000
    assert probe.duration_sec == pytest.approx(0.5, abs=0.05)
    assert probe.channels == 1
    assert probe.bit_depth == 16


def test_probe_reports_unknown_rather_than_guessing() -> None:
    """An unreadable payload yields None, never a default that would render as
    a measured figure."""
    probe = probe_audio(b"not audio at all")
    assert probe.sample_rate is None
    assert probe.duration_sec is None


def test_probe_does_not_raise_on_empty_bytes() -> None:
    """A failed engine can leave an empty body; probing it must not 500 the route."""
    assert probe_audio(b"").duration_sec is None


def _real_mp3() -> bytes | None:
    """A genuine MP3, encoded on the fly. None when ffmpeg is unavailable.

    Synthetic frame-sync bytes are not enough here: the bug this guards was in
    how ffprobe's OUTPUT is parsed, so the payload has to be something ffprobe
    can actually read.
    """
    if shutil.which("ffmpeg") is None:
        return None
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=1", "-ac", "1", "-ar", "24000",
         "-f", "mp3", "pipe:1"],
        capture_output=True,
    )
    return proc.stdout if proc.returncode == 0 and proc.stdout else None


def test_probe_reads_an_mp3_that_carries_no_bit_depth() -> None:
    """MP3 reports no `bits_per_raw_sample` at all, and ffprobe OMITS a key it
    has no value for rather than emitting null.

    Indexing that key threw a KeyError that escaped to the outer suppress and
    discarded the rate and duration too, so every mp3 clip stored null
    everything -- while WAV kept working via the `wave` fallback, which is
    exactly why no existing test noticed. A real rate beside a null bit depth
    is the correct answer; all-None is the bug.
    """
    payload = _real_mp3()
    if payload is None:
        pytest.skip("ffmpeg not available to encode a real mp3")

    probe = probe_audio(payload)
    assert probe.sample_rate == 24000, "the rate was discarded along with the missing key"
    assert probe.duration_sec is not None and probe.duration_sec > 0.5
    assert probe.channels == 1
    # Genuinely unknowable for this codec, so null rather than a guessed 16.
    assert probe.bit_depth is None
