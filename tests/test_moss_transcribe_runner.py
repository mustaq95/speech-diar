"""The moss-transcribe runner's HTTP contract with its vLLM container.

Regression cover for a real incident: a 32-minute upload failed with the model's
row reading only `Client error '400 Bad Request' for url ...`. vLLM had said
exactly what was wrong ("Maximum file size exceeded") in the response body, and
`raise_for_status()` discarded it, so diagnosing a one-line config problem meant
reading vLLM's source. These tests pin the two request params that are silently
load-bearing, and the error path that made the incident hard to read.
"""

from types import SimpleNamespace

import pytest

from apps.background_worker.models.moss_transcribe import runner as moss_runner


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
    wav.write_bytes(b"RIFFfakewavbytes")
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
    wav.write_bytes(b"RIFFfakewavbytes")
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
    wav.write_bytes(b"RIFFfakewavbytes")
    monkeypatch.setattr(moss_runner, "get_settings", lambda: _fake_settings())
    _stub_post(
        monkeypatch,
        status_code=400,
        text="Input length (90000) exceeds model's maximum context length (65536).",
    )

    with pytest.raises(RuntimeError, match="exceeds model's maximum context length"):
        moss_runner.MossTranscribeRunner().run(str(wav))
