"""The cohere-transcribe runner's HTTP contract with its vLLM container.

Mirrors `test_moss_transcribe_runner.py`, which covers the same endpoint for a
different model, including the error path that made a real incident hard to
read: vLLM puts the actionable reason in the response body, and
`raise_for_status()` throws it away.

The canonicalization cover is not incidental. The 32-minute Arabic sample in
`tests/samples/` is 44.1 kHz stereo (326 MB) and is rejected outright by the
container's file-size gate; converted to the platform's canonical 16 kHz mono
it is 59 MB and transcribes fine. Every recording therefore has to go through
`ensure_canonical_wav` before it is sent.
"""

import os
import shutil
import wave
from types import SimpleNamespace

import pytest

import packages.audio as audio
from apps.background_worker.transcription.cohere import runner as cohere_runner
from tests.conftest import make_wav_bytes


def _write_wav(path, *, framerate: int, nchannels: int, sampwidth: int = 2, duration_sec: float = 0.5) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(nchannels)
        wav.setsampwidth(sampwidth)
        wav.setframerate(framerate)
        wav.writeframes(b"\x00" * (int(duration_sec * framerate) * nchannels * sampwidth))


def _fake_settings(**overrides):
    defaults = dict(
        cohere_transcribe_url="http://localhost:9025",
        cohere_transcribe_timeout_sec=1800,
        cohere_transcribe_language="",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _stub_post(monkeypatch, *, status_code=200, text="", payload=None, seen=None):
    def fake_post(url, files=None, data=None, timeout=None, **kwargs):
        if seen is not None:
            seen.update(url=url, data=data, timeout=timeout, files=files)
        return SimpleNamespace(status_code=status_code, text=text, json=lambda: payload or {})

    monkeypatch.setattr(cohere_runner.httpx, "post", fake_post)


def test_runner_posts_the_expected_request(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    wav = tmp_path / "clip.wav"
    wav.write_bytes(make_wav_bytes(duration_sec=0.5))
    monkeypatch.setattr(cohere_runner, "get_settings", lambda: _fake_settings())
    seen: dict = {}
    _stub_post(monkeypatch, payload={"text": "مرحبا"}, seen=seen)

    raw = cohere_runner.run(str(wav))

    assert seen["url"] == "http://localhost:9025/v1/audio/transcriptions"
    assert seen["timeout"] == 1800
    # Addressed by --served-model-name, not the HF repo id, so repointing
    # COHERE_TRANSCRIBE_MODEL_ID does not break the runner.
    assert seen["data"]["model"] == "cohere-transcribe"
    assert seen["data"]["response_format"] == "json"
    # Transcription, not generation: a re-run must reproduce the same text or
    # an online-vs-offline timing comparison means nothing.
    assert seen["data"]["temperature"] == "0"
    # Default is auto-detect: no forced language is sent, so the Arabic-first
    # model does not hallucinate Arabic on English audio.
    assert "language" not in seen["data"]
    # Native payload handed back untouched for the adapter to read.
    assert raw == {"text": "مرحبا"}


def test_runner_forces_language_only_when_set(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A non-empty setting forces that language; empty omits it entirely. The
    forced value becomes a decoder prompt the model obeys, so it must reach the
    request verbatim when set and be absent when not."""
    wav = tmp_path / "clip.wav"
    wav.write_bytes(make_wav_bytes(duration_sec=0.5))

    monkeypatch.setattr(cohere_runner, "get_settings", lambda: _fake_settings(cohere_transcribe_language="ar"))
    seen: dict = {}
    _stub_post(monkeypatch, payload={"text": "مرحبا"}, seen=seen)
    cohere_runner.run(str(wav))
    assert seen["data"]["language"] == "ar"

    monkeypatch.setattr(cohere_runner, "get_settings", lambda: _fake_settings(cohere_transcribe_language=""))
    seen.clear()
    cohere_runner.run(str(wav))
    assert "language" not in seen["data"]


def test_runner_surfaces_the_servers_reason_not_just_the_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A bare '400 Bad Request' on the row turns a one-line config problem into
    a `docker logs` expedition. The body says which gate was hit."""
    wav = tmp_path / "clip.wav"
    wav.write_bytes(make_wav_bytes(duration_sec=0.5))
    monkeypatch.setattr(cohere_runner, "get_settings", lambda: _fake_settings())
    _stub_post(
        monkeypatch,
        status_code=400,
        text='{"error":{"message":"Maximum file size exceeded (parameter=audio_filesize_mb)"}}',
    )

    with pytest.raises(RuntimeError) as excinfo:
        cohere_runner.run(str(wav))

    assert "400" in str(excinfo.value)
    assert "Maximum file size exceeded" in str(excinfo.value)


def test_runner_explains_a_container_that_is_not_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """This container is exempt from the GPU supervisor, so nothing starts it on
    demand. A connection refusal must say so and name the script, rather than
    surfacing a bare httpx ConnectError."""
    import httpx

    wav = tmp_path / "clip.wav"
    wav.write_bytes(make_wav_bytes(duration_sec=0.5))
    monkeypatch.setattr(cohere_runner, "get_settings", lambda: _fake_settings())

    def refuse(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(cohere_runner.httpx, "post", refuse)

    with pytest.raises(RuntimeError) as excinfo:
        cohere_runner.run(str(wav))

    message = str(excinfo.value)
    assert "not reachable" in message
    assert "http://localhost:9025" in message
    assert "cohere_transcribe_up.sh" in message


_needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@_needs_ffmpeg
def test_run_transcodes_and_cleans_up_a_non_canonical_recording(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """44.1 kHz stereo is the shape that blows past the container's file-size
    gate; it must be downmixed before sending, and the temp file must not leak."""
    wav = tmp_path / "clip.wav"
    _write_wav(wav, framerate=44100, nchannels=2)
    monkeypatch.setattr(cohere_runner, "get_settings", lambda: _fake_settings())
    seen: dict = {}
    _stub_post(monkeypatch, payload={"text": "hi"}, seen=seen)

    created: list[str] = []
    real_mkstemp = audio.tempfile.mkstemp

    def spy_mkstemp(*args, **kwargs):
        fd, path = real_mkstemp(*args, **kwargs)
        created.append(path)
        return fd, path

    monkeypatch.setattr(audio.tempfile, "mkstemp", spy_mkstemp)

    cohere_runner.run(str(wav))

    assert created, "expected a transcoded temp file for a non-canonical recording"
    # What was actually sent is the 16kHz mono conversion, not the 44.1kHz source.
    assert seen["files"]["file"][0] == os.path.basename(created[0])
    with wave.open(str(wav), "rb") as original:
        assert original.getframerate() == 44100  # the source is left untouched
    assert not os.path.exists(created[0]), "the transcoded temp file leaked"
