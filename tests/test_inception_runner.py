"""The inception-stt runner's HTTP contract with the LiteLLM gateway.

Mirrors `test_cohere_transcribe_runner.py`, which covers a different engine on a
structurally similar endpoint.

Two things here are not incidental cover:

**The language sentinel.** `STT_DEFAULT_LANGUAGE` is written as "auto" in `.env`,
which is not a language code the gateway understands. Sent verbatim it becomes a
decoder prompt the model obeys over the audio — the same failure
`cohere_transcribe_language` documents, where forcing "ar" makes English
recordings hallucinate Arabic. Since the whole comparison is about mixed ar/en
speech, the request must carry no `language` field at all unless someone sets a
real code. That is asserted on the request, not just on the helper.

**Splitting has to recover content.** The gateway drops audio
non-monotonically (see the runner's module docstring for the measurements), so
segmenting is what makes a long transcript exist. The live test proves recovery
against the real gateway; a short clip would prove nothing, because a clip that
needs no splitting exercises none of this.
"""

import os
import wave
from types import SimpleNamespace

import httpx
import pytest

from apps.background_worker.transcription.inception import adapter, runner
from tests.conftest import make_wav_bytes


def _fake_settings(**overrides):
    defaults = dict(
        stt_endpoint_url="https://gateway.example/v1/audio/transcriptions",
        litellm_api_key="k-test",
        litellm_base_url="https://gateway.example",
        stt_model="inception-stt",
        stt_default_language="auto",
        stt_timeout_seconds=120,
        batch_segment_seconds=3,
        batch_max_concurrency=4,
        litellm_httpx_verify=True,
        stt_keepalive_expiry_sec=30,
        stt_max_connections=16,
        live_chunk_default_sec=3,
        # These tests cover the SPLIT regime, which is what most of them are
        # about (ordering, per-segment latency, one-call-for-short-audio). Pinned
        # explicitly rather than inherited: production defaults this to True, and
        # a fixture that followed it would silently stop exercising the splitting
        # the moment the default flipped. The whole-file branch has its own test.
        inception_batch_whole_file=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _stub_post(monkeypatch, *, status_code=200, text="", payload=None, calls=None,
               raises=None):
    """Stub the POOLED client, not httpx.post.

    The runner reuses one `httpx.Client` so the TCP+TLS handshake is paid once
    rather than per call (73 ms of a 107 ms round trip, measured). Patching
    `httpx.post` would leave that untested and would silently pass even if the
    runner went back to a fresh connection per call, so the seam here is
    `_http_client` — the same object the real code posts through.
    """
    def fake_post(url, headers=None, data=None, files=None, timeout=None, **kwargs):
        if calls is not None:
            calls.append(dict(url=url, headers=headers, data=data, files=files,
                              timeout=timeout))
        if raises is not None:
            raise raises
        return SimpleNamespace(status_code=status_code, text=text,
                               json=lambda: dict(payload or {}))

    client = SimpleNamespace(post=fake_post)
    monkeypatch.setattr(runner, "_http_client", lambda settings: client)
    return client


def _write_wav(path, duration_sec: float, framerate: int = 16000) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(framerate)
        wav.writeframes(b"\x00" * (int(duration_sec * framerate) * 2))


# --- the language sentinel -------------------------------------------------

@pytest.mark.parametrize("configured", ["", "auto", "AUTO", " auto ", "detect"])
def test_auto_language_is_omitted_from_the_request(monkeypatch, tmp_path, configured) -> None:
    """Every spelling of "let the model decide" must send NO language field."""
    wav = tmp_path / "clip.wav"
    wav.write_bytes(make_wav_bytes(duration_sec=0.5))
    monkeypatch.setattr(runner, "get_settings",
                        lambda: _fake_settings(stt_default_language=configured))
    calls: list = []
    _stub_post(monkeypatch, payload={"text": "hello"}, calls=calls)

    runner.run(str(wav))

    assert len(calls) == 1
    assert "language" not in calls[0]["data"], (
        f"{configured!r} leaked into the request as a forced language"
    )


def test_a_real_language_code_is_sent(monkeypatch, tmp_path) -> None:
    wav = tmp_path / "clip.wav"
    wav.write_bytes(make_wav_bytes(duration_sec=0.5))
    monkeypatch.setattr(runner, "get_settings", lambda: _fake_settings(stt_default_language="ar"))
    calls: list = []
    _stub_post(monkeypatch, payload={"text": "مرحبا"}, calls=calls)

    runner.run(str(wav))

    assert calls[0]["data"]["language"] == "ar"


def test_requested_language_helper() -> None:
    assert runner.requested_language(_fake_settings(stt_default_language="auto")) is None
    assert runner.requested_language(_fake_settings(stt_default_language="")) is None
    assert runner.requested_language(_fake_settings(stt_default_language="en")) == "en"


# --- the request shape -----------------------------------------------------

def test_runner_posts_model_auth_and_timeout(monkeypatch, tmp_path) -> None:
    wav = tmp_path / "clip.wav"
    wav.write_bytes(make_wav_bytes(duration_sec=0.5))
    monkeypatch.setattr(runner, "get_settings", lambda: _fake_settings())
    calls: list = []
    _stub_post(monkeypatch, payload={"text": "hi"}, calls=calls)

    runner.run(str(wav))

    call = calls[0]
    assert call["url"] == "https://gateway.example/v1/audio/transcriptions"
    assert call["headers"]["Authorization"] == "Bearer k-test"
    assert call["data"]["model"] == "inception-stt"
    assert call["timeout"] == 120


def test_the_client_is_pooled(monkeypatch) -> None:
    """One client per process. Pooling is the point: a fresh connection per call
    cost 73 ms of handshake on every chunk, reported as the engine's latency."""
    runner._client = None
    settings = _fake_settings()
    first = runner._http_client(settings)
    second = runner._http_client(settings)

    assert first is second, "a new client per call would re-handshake every time"
    runner._client = None


def test_keepalive_outlasts_the_chunk_interval() -> None:
    """A pool whose connections expire between chunks pays the handshake anyway.
    Asserted against the shipped defaults, not a fixture, because this is a
    relationship between two settings someone could tune apart."""
    from packages.config.settings import Settings

    defaults = Settings.model_fields
    assert (defaults["stt_keepalive_expiry_sec"].default
            > defaults["live_chunk_max_sec"].default), (
        "STT_KEEPALIVE_EXPIRY_SEC must exceed the largest selectable chunk interval"
    )


def test_the_tls_setting_reaches_httpx(monkeypatch) -> None:
    """A CA bundle path must arrive as `verify`, not be reduced to a bool.

    Captured at the httpx.Client boundary rather than by inspecting a built
    client: the internals are private and a real bundle path would have to exist
    on disk for the constructor to succeed.
    """
    runner._client = None
    seen: dict = {}

    class FakeClient:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def close(self):
            pass

    monkeypatch.setattr(runner.httpx, "Client", FakeClient)
    runner._http_client(_fake_settings(litellm_httpx_verify="/etc/ca.pem"))

    assert seen["verify"] == "/etc/ca.pem"
    assert seen["limits"].keepalive_expiry == 30
    runner._client = None


def test_the_client_is_rebuilt_when_tls_changes(monkeypatch) -> None:
    runner._client = None
    first = runner._http_client(_fake_settings(litellm_httpx_verify=True))
    second = runner._http_client(_fake_settings(litellm_httpx_verify=False))
    assert first is not second
    runner._client = None


def test_latency_is_measured_per_segment(monkeypatch, tmp_path) -> None:
    wav = tmp_path / "clip.wav"
    wav.write_bytes(make_wav_bytes(duration_sec=0.5))
    monkeypatch.setattr(runner, "get_settings", lambda: _fake_settings())
    _stub_post(monkeypatch, payload={"text": "hi"})

    raw = runner.run(str(wav))

    assert isinstance(raw[0]["latency_ms"], int)
    assert raw[0]["latency_ms"] >= 0
    # The engine's own JSON stays intact under "response", with our measurements
    # beside it rather than merged in.
    assert raw[0]["response"] == {"text": "hi"}


# --- splitting -------------------------------------------------------------

def test_long_audio_is_split_and_kept_in_audio_order(monkeypatch, tmp_path) -> None:
    """10 s at a 3 s segment length is 4 calls, and the returned list must be in
    audio order — the adapter joins it into prose, so completion order would
    scramble the transcript."""
    wav = tmp_path / "long.wav"
    _write_wav(wav, duration_sec=10)
    monkeypatch.setattr(runner, "get_settings", lambda: _fake_settings(batch_segment_seconds=3))
    calls: list = []
    _stub_post(monkeypatch, payload={"text": "piece"}, calls=calls)

    raw = runner.run(str(wav))

    assert len(calls) == 4
    assert [part["segment_index"] for part in raw] == [0, 1, 2, 3]
    starts = [part["segment_start"] for part in raw]
    assert starts == sorted(starts)
    assert raw[0]["segment_start"] == 0.0
    assert raw[-1]["segment_end"] == pytest.approx(10.0, abs=0.01)


def test_short_audio_is_one_call(monkeypatch, tmp_path) -> None:
    wav = tmp_path / "short.wav"
    _write_wav(wav, duration_sec=2)
    monkeypatch.setattr(runner, "get_settings", lambda: _fake_settings(batch_segment_seconds=3))
    calls: list = []
    _stub_post(monkeypatch, payload={"text": "one"}, calls=calls)

    raw = runner.run(str(wav))

    assert len(calls) == 1
    assert len(raw) == 1


# --- error mapping ---------------------------------------------------------

def test_upstream_body_survives_in_the_error(monkeypatch, tmp_path) -> None:
    """The gateway puts the actionable reason in the body; httpx's own message
    throws it away. That text ends up in TranscriptResult.error and on the panel."""
    wav = tmp_path / "clip.wav"
    wav.write_bytes(make_wav_bytes(duration_sec=0.5))
    monkeypatch.setattr(runner, "get_settings", lambda: _fake_settings())
    _stub_post(monkeypatch, status_code=400, text="Invalid or unsupported audio file")

    with pytest.raises(runner.InceptionUpstreamError) as excinfo:
        runner.run(str(wav))
    assert "Invalid or unsupported audio file" in str(excinfo.value)
    assert excinfo.value.status_code == 400


@pytest.mark.parametrize(
    "raised, expected",
    [
        (httpx.TimeoutException("slow"), runner.InceptionTimeoutError),
        (httpx.ConnectError("refused"), runner.InceptionConnectionError),
    ],
)
def test_transport_failures_map_to_typed_errors(monkeypatch, tmp_path, raised, expected) -> None:
    wav = tmp_path / "clip.wav"
    wav.write_bytes(make_wav_bytes(duration_sec=0.5))
    monkeypatch.setattr(runner, "get_settings", lambda: _fake_settings())
    _stub_post(monkeypatch, raises=raised)
    with pytest.raises(expected):
        runner.run(str(wav))


def test_non_string_text_is_a_response_error(monkeypatch, tmp_path) -> None:
    wav = tmp_path / "clip.wav"
    wav.write_bytes(make_wav_bytes(duration_sec=0.5))
    monkeypatch.setattr(runner, "get_settings", lambda: _fake_settings())
    _stub_post(monkeypatch, payload={"text": {"unexpected": "shape"}})

    with pytest.raises(runner.InceptionResponseError):
        runner.run(str(wav))


def test_unconfigured_gateway_fails_with_a_named_reason(monkeypatch, tmp_path) -> None:
    wav = tmp_path / "clip.wav"
    wav.write_bytes(make_wav_bytes(duration_sec=0.5))
    monkeypatch.setattr(runner, "get_settings",
                        lambda: _fake_settings(stt_endpoint_url=None, litellm_base_url=None))

    with pytest.raises(runner.InceptionError, match="LITELLM_BASE_URL"):
        runner.run(str(wav))


# --- adapter ---------------------------------------------------------------

def test_adapter_joins_segments_in_index_order() -> None:
    raw = [
        {"response": {"text": " world "}, "segment_index": 1},
        {"response": {"text": "hello"}, "segment_index": 0},
        {"response": {"text": ""}, "segment_index": 2},
        {"response": {"text": "again"}, "segment_index": 3},
    ]
    assert adapter.adapt(raw) == "hello world again"


def test_adapter_tolerates_missing_text() -> None:
    assert adapter.adapt([{"segment_index": 0},
                          {"response": {"text": None}, "segment_index": 1}]) == ""


def test_adapter_ignores_non_text_native_fields() -> None:
    """usage and word_timestamps are dropped from the transcript but must not
    break it — and they still reach raw_output untouched."""
    raw = [{"response": {"text": "kept", "usage": {"tokens": 9},
                         "word_timestamps": None, "audio_duration": 3.0},
            "segment_index": 0, "latency_ms": 12}]
    assert adapter.adapt(raw) == "kept"


# --- live -----------------------------------------------------------------

@pytest.mark.live
def test_splitting_recovers_content_a_single_call_drops() -> None:
    """Against the real gateway: splitting must beat one call on the same audio.

    The clip has to be long enough to be split — the whole defect is
    length-dependent, so a clip under the segment length exercises nothing.
    """
    sample = "tests/samples/3-two-speakers-en.wav"
    if not os.path.exists(sample):
        pytest.skip(f"{sample} not present")

    split_words = len(adapter.adapt(runner.run(sample)).split())

    with wave.open(sample, "rb") as wav:
        duration = wav.getnframes() / float(wav.getframerate())
    with open(sample, "rb") as fh:
        whole = runner.transcribe_bytes(fh.read(), "whole.wav")
    single_words = len(runner.text_of(whole).split())

    assert duration > 30, "fixture too short to exercise splitting"
    assert split_words > single_words, (
        f"splitting recovered {split_words} words, one call recovered {single_words} — "
        "segmenting is supposed to recover content the gateway drops"
    )


# --- INCEPTION_BATCH_WHOLE_FILE ----------------------------------------------
#
# The comparison surface gives both engines the same input in batch mode, so this
# engine sends the recording whole rather than split. It costs transcript: on one
# 65s recording the split returns 140 words and the whole file returns 65, at
# HTTP 200 with no error. That is an accepted trade, not a bug, and these tests
# pin the mechanism so the trade stays deliberate.


def test_whole_file_mode_makes_exactly_one_call_for_long_audio(monkeypatch, tmp_path) -> None:
    """10s at a 3s segment length is 4 calls when splitting, and must be 1 here.

    The count is the whole point: it is what makes the input identical to
    cohere-transcribe's, and it is also what discards the content.
    """
    wav = tmp_path / "long.wav"
    _write_wav(wav, duration_sec=10)
    monkeypatch.setattr(
        runner, "get_settings",
        lambda: _fake_settings(inception_batch_whole_file=True, batch_segment_seconds=3),
    )
    calls: list = []
    _stub_post(monkeypatch, payload={"text": "whole"}, calls=calls)

    raw = runner.run(str(wav))

    assert len(calls) == 1, "whole-file mode must not split, whatever the segment length says"
    assert len(raw) == 1
    # Still a LIST of entries, so the adapter and raw_output are unchanged by the
    # branch -- an engine returning a bare dict here would break both.
    assert isinstance(raw, list)
    assert raw[0]["segment_index"] == 0


def test_whole_file_mode_carries_no_segment_offsets(monkeypatch, tmp_path) -> None:
    """There is one piece spanning the recording, so a start/end offset would be
    describing a split that did not happen."""
    wav = tmp_path / "long.wav"
    _write_wav(wav, duration_sec=10)
    monkeypatch.setattr(
        runner, "get_settings", lambda: _fake_settings(inception_batch_whole_file=True)
    )
    _stub_post(monkeypatch, payload={"text": "whole"})

    raw = runner.run(str(wav))

    assert "segment_start" not in raw[0]
    assert "segment_end" not in raw[0]


def test_the_flag_is_what_decides_not_the_audio_length(monkeypatch, tmp_path) -> None:
    """Same audio, same segment length, both regimes — so a future change cannot
    make the branch depend on duration by accident."""
    wav = tmp_path / "long.wav"
    _write_wav(wav, duration_sec=10)

    monkeypatch.setattr(
        runner, "get_settings",
        lambda: _fake_settings(inception_batch_whole_file=False, batch_segment_seconds=3),
    )
    split_calls: list = []
    _stub_post(monkeypatch, payload={"text": "piece"}, calls=split_calls)
    assert len(runner.run(str(wav))) == 4

    monkeypatch.setattr(
        runner, "get_settings",
        lambda: _fake_settings(inception_batch_whole_file=True, batch_segment_seconds=3),
    )
    whole_calls: list = []
    _stub_post(monkeypatch, payload={"text": "whole"}, calls=whole_calls)
    assert len(runner.run(str(wav))) == 1
