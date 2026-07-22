"""The moss-transcribe runner's HTTP contract with its vLLM container.

Regression cover for a real incident: a 32-minute upload failed with the model's
row reading only `Client error '400 Bad Request' for url ...`. vLLM had said
exactly what was wrong ("Maximum file size exceeded") in the response body, and
`raise_for_status()` discarded it, so diagnosing a one-line config problem meant
reading vLLM's source. These tests pin the two request params that are silently
load-bearing, and the error path that made the incident hard to read.
"""

import os
import shutil
import wave
from types import SimpleNamespace

import pytest

from apps.background_worker.models.moss_transcribe import runner as moss_runner
from tests.conftest import make_wav_bytes


def _write_wav(path, *, framerate: int, nchannels: int, sampwidth: int = 2, duration_sec: float = 0.5) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(nchannels)
        wav.setsampwidth(sampwidth)
        wav.setframerate(framerate)
        wav.writeframes(b"\x00" * (int(duration_sec * framerate) * nchannels * sampwidth))


def _fake_settings(**overrides):
    defaults = dict(
        moss_transcribe_url="http://localhost:9024",
        moss_transcribe_timeout_sec=1800,
        moss_transcribe_cold_start_timeout_sec=600,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _stub_post(monkeypatch, *, status_code=200, text="", payload=None, seen=None):
    def fake_post(url, files=None, data=None, timeout=None, **kwargs):
        if seen is not None:
            seen.update(url=url, data=data, timeout=timeout, files=files)
        return SimpleNamespace(
            status_code=status_code,
            text=text,
            json=lambda: payload or {},
        )

    monkeypatch.setattr(moss_runner.httpx, "post", fake_post)


def test_runner_posts_the_params_vllm_actually_honors(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    wav = tmp_path / "clip.wav"
    wav.write_bytes(make_wav_bytes(duration_sec=0.5))
    monkeypatch.setattr(moss_runner, "get_settings", lambda: _fake_settings())
    seen: dict = {}
    _stub_post(monkeypatch, payload={"text": "[0.0][S01]hi[1.0]"}, seen=seen)

    raw = moss_runner.MossTranscribeRunner().run(str(wav))

    assert seen["url"] == "http://localhost:9024/v1/audio/transcriptions"
    assert seen["timeout"] == 1800
    # Addressed by --served-model-name, not the HF repo id, so repointing
    # MOSS_TRANSCRIBE_MODEL_ID does not break the runner.
    assert seen["data"]["model"] == "moss-transcribe"
    # `json` not `verbose_json`: vLLM sets supports_segment_timestamp=False for
    # this model, so there is no parsed-segment path to ask for.
    assert seen["data"]["response_format"] == "json"
    # The pin that keeps long transcripts intact. Without it vLLM falls back to
    # the model's generation_config.json (max_tokens=5120) and truncates.
    # NOT `max_new_tokens` -- that is SGLang's name and vLLM ignores it.
    # Matches the model's real 128k context (max_position_embeddings=131072);
    # vLLM clamps it down to whatever the audio leaves free.
    assert seen["data"]["max_completion_tokens"] == "131072"
    assert "max_new_tokens" not in seen["data"]
    # No prompt override: vLLM's built-in default is the Chinese instruction the
    # model was trained against.
    assert "prompt" not in seen["data"]
    # Native payload handed back untouched for the adapter to parse.
    assert raw["text"] == "[0.0][S01]hi[1.0]"


def test_runner_surfaces_the_servers_reason_not_just_the_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The incident: the row said '400 Bad Request' and nothing else. vLLM's
    body carries the actionable reason, so it must reach the error message."""
    wav = tmp_path / "clip.wav"
    wav.write_bytes(make_wav_bytes(duration_sec=0.5))
    monkeypatch.setattr(moss_runner, "get_settings", lambda: _fake_settings())
    _stub_post(
        monkeypatch,
        status_code=400,
        text='{"error":{"message":"Maximum file size exceeded","param":"audio_filesize_mb"}}',
    )

    with pytest.raises(RuntimeError) as excinfo:
        moss_runner.MossTranscribeRunner().run(str(wav))

    assert "400" in str(excinfo.value)
    assert "Maximum file size exceeded" in str(excinfo.value)


def test_runner_surfaces_context_length_overflow(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Audio longer than the context window is the honest failure the raised
    audio-size limits deliberately expose rather than mask -- it must be
    readable on the model's row."""
    wav = tmp_path / "clip.wav"
    wav.write_bytes(make_wav_bytes(duration_sec=0.5))
    monkeypatch.setattr(moss_runner, "get_settings", lambda: _fake_settings())
    _stub_post(
        monkeypatch,
        status_code=400,
        text="Input length (90000) exceeds model's maximum context length (65536).",
    )

    with pytest.raises(RuntimeError, match="exceeds model's maximum context length"):
        moss_runner.MossTranscribeRunner().run(str(wav))


_needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_ensure_canonical_passes_canonical_wav_through_untouched(tmp_path) -> None:
    """A 16 kHz mono s16 file is already what MOSS expects -- no transcode, no
    temp file, no ffmpeg."""
    wav = tmp_path / "clip.wav"
    _write_wav(wav, framerate=16000, nchannels=1)

    send_path, cleanup = moss_runner._ensure_canonical(str(wav))

    assert send_path == str(wav)
    assert cleanup is None


@_needs_ffmpeg
def test_ensure_canonical_transcodes_non_canonical_wav(tmp_path) -> None:
    """A pre-canonicalization recording (44.1 kHz stereo -- the shape that blew
    past vLLM's file-size gate) is downmixed to 16 kHz mono s16 before sending."""
    wav = tmp_path / "clip.wav"
    _write_wav(wav, framerate=44100, nchannels=2)

    send_path, cleanup = moss_runner._ensure_canonical(str(wav))
    try:
        assert send_path != str(wav)
        assert cleanup == send_path
        with wave.open(send_path, "rb") as out:
            assert out.getframerate() == 16000
            assert out.getnchannels() == 1
            assert out.getsampwidth() == 2
    finally:
        if os.path.exists(send_path):
            os.unlink(send_path)


@_needs_ffmpeg
def test_run_deletes_the_transcoded_temp_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The temp WAV created for a non-canonical upload must not leak once the
    request finishes."""
    wav = tmp_path / "clip.wav"
    _write_wav(wav, framerate=44100, nchannels=2)
    monkeypatch.setattr(moss_runner, "get_settings", lambda: _fake_settings())
    _stub_post(monkeypatch, payload={"text": "[0.0][S01]hi[1.0]"})

    created: list[str] = []
    real_mkstemp = moss_runner.tempfile.mkstemp

    def spy_mkstemp(*args, **kwargs):
        fd, path = real_mkstemp(*args, **kwargs)
        created.append(path)
        return fd, path

    monkeypatch.setattr(moss_runner.tempfile, "mkstemp", spy_mkstemp)

    moss_runner.MossTranscribeRunner().run(str(wav))

    assert created, "expected a temp file to be created for a non-canonical upload"
    assert not os.path.exists(created[0])
