"""The remote-vibevoice path: VIBEVOICE_BASEURL set => the runner calls the
OpenAI-style chat-completions proxy instead of the local container, and
vibevoice drops out of the GPU supervisor entirely (no managed container, no
residency slot, absent from /models/status so the UI renders it as
In-process).

The remote endpoint itself is stubbed everywhere here — at the time this
landed the proxy's hostname did not even resolve from the DGX host, so the
wire format (input_audio content part in, generated text in
choices[0].message.content out) is the standard OpenAI shape, verified
against these tests but NOT yet against the real service.
"""

import json
from types import SimpleNamespace

import pytest

from apps.background_worker.models.vibevoice import runner as vibevoice_runner
from apps.background_worker.supervisor import registry


def _fake_settings(**overrides):
    defaults = dict(
        vibevoice_url="http://localhost:9023",
        vibevoice_timeout_sec=5400,
        vibevoice_cold_start_timeout_sec=600,
        vibevoice_baseurl=None,
        vibevoice_api_key=None,
        vibevoice_ssl_verify=True,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_remote_path_posts_audio_and_parses_generated_text(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFFfakewavbytes")
    settings = _fake_settings(vibevoice_baseurl="https://proxy.example/v1/chat/completions", vibevoice_api_key="sk-test")
    monkeypatch.setattr(vibevoice_runner, "get_settings", lambda: settings)

    seen: dict = {}

    def fake_post(url, headers=None, json=None, timeout=None, verify=None, **kwargs):
        seen.update(url=url, headers=headers, body=json, timeout=timeout, verify=verify)
        content = '[{"Start": 0.0, "End": 1.5, "Speaker": 1}, {"Start": 1.5, "End": 2.0}]'
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": content}}]},
        )

    monkeypatch.setattr(vibevoice_runner.httpx, "post", fake_post)

    raw = vibevoice_runner.VibeVoiceRunner().run(str(wav))

    assert seen["url"] == "https://proxy.example/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer sk-test"
    assert seen["timeout"] == 5400
    assert seen["verify"] is True
    part = seen["body"]["messages"][0]["content"][0]
    assert part["type"] == "input_audio"
    assert part["input_audio"]["format"] == "wav"
    assert part["input_audio"]["data"]  # base64 of the file
    # Native shape matches what the local container returns, so the adapter
    # is untouched: speakerless entries preserved, nothing merged.
    assert raw["segments"] == [{"Start": 0.0, "End": 1.5, "Speaker": 1}, {"Start": 1.5, "End": 2.0}]


def test_remote_path_honors_ssl_verify_toggle(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """VIBEVOICE_SSL_VERIFY=false must reach httpx as verify=False, for
    proxies behind a cert the worker host doesn't trust."""
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFFfakewavbytes")
    settings = _fake_settings(
        vibevoice_baseurl="https://proxy.example/v1/chat/completions",
        vibevoice_ssl_verify=False,
    )
    monkeypatch.setattr(vibevoice_runner, "get_settings", lambda: settings)

    seen: dict = {}

    def fake_post(url, headers=None, json=None, timeout=None, verify=None, **kwargs):
        seen.update(verify=verify)
        content = '[{"Start": 0.0, "End": 1.5, "Speaker": 1}]'
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": content}}]},
        )

    monkeypatch.setattr(vibevoice_runner.httpx, "post", fake_post)

    vibevoice_runner.VibeVoiceRunner().run(str(wav))

    assert seen["verify"] is False


def test_remote_path_recovers_segments_from_malformed_json(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Same tolerance as the container server: the model emits text that
    only resembles JSON, and a quote inside a transcript breaks strict
    parsing — the numeric fields must still come through."""
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFFfakewavbytes")
    settings = _fake_settings(vibevoice_baseurl="https://proxy.example/v1/chat/completions")
    monkeypatch.setattr(vibevoice_runner, "get_settings", lambda: settings)

    broken = '[{"Start": 0.0, "End": 2.5, "Speaker": 3, "Content": "it"s broken"}]'
    with pytest.raises(json.JSONDecodeError):
        json.loads(broken)  # prove the fixture is genuinely malformed

    monkeypatch.setattr(
        vibevoice_runner.httpx,
        "post",
        lambda *a, **k: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": broken}}]},
        ),
    )

    raw = vibevoice_runner.VibeVoiceRunner().run(str(wav))

    assert raw["segments"] == [{"Start": 0.0, "End": 2.5, "Speaker": 3}]


def test_remote_path_raises_clearly_when_output_is_unparseable(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFFfakewavbytes")
    settings = _fake_settings(vibevoice_baseurl="https://proxy.example/v1/chat/completions")
    monkeypatch.setattr(vibevoice_runner, "get_settings", lambda: settings)
    monkeypatch.setattr(
        vibevoice_runner.httpx,
        "post",
        lambda *a, **k: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": "I could not process this audio."}}]},
        ),
    )

    with pytest.raises(RuntimeError, match="did not parse as segments"):
        vibevoice_runner.VibeVoiceRunner().run(str(wav))


def test_local_path_is_untouched_when_no_baseurl(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFFfakewavbytes")
    settings = _fake_settings()  # baseurl None
    monkeypatch.setattr(vibevoice_runner, "get_settings", lambda: settings)

    seen: dict = {}

    def fake_post(url, files=None, timeout=None, **kwargs):
        seen.update(url=url, timeout=timeout)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"segments": []})

    monkeypatch.setattr(vibevoice_runner.httpx, "post", fake_post)

    vibevoice_runner.VibeVoiceRunner().run(str(wav))

    assert seen["url"] == "http://localhost:9023/diarize"
    assert seen["timeout"] == 5400


def test_vibevoice_leaves_the_managed_set_when_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a remote proxy configured, vibevoice must not be
    supervisor-managed: no container, no GPU residency slot, and
    local_pipeline takes the unmanaged fast path via is_managed()."""
    remote = registry.get_settings().model_copy(update={"vibevoice_baseurl": "https://proxy.example/v1"})
    monkeypatch.setattr(registry, "get_settings", lambda: remote)

    assert registry.managed_container("vibevoice") is None
    assert registry.is_managed("vibevoice") is False
    assert "vibevoice" not in registry.all_managed_model_ids()
    # The other five managed models are unaffected.
    assert registry.is_managed("nemo-clustering") is True


def test_status_derivation_skips_rows_for_unmanaged_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """A leftover model_container_state row for a model that moved off-host
    must be omitted from /models/status, not reported as a fabricated
    'unloaded' — its absence is what makes the frontend render the
    In-process chip."""
    from apps.background_worker.supervisor import state
    from packages.database.models import ModelContainerState

    monkeypatch.setattr(state, "live_busy_model_ids", lambda: set())
    monkeypatch.setattr(state.containers, "running_containers", lambda: set())
    monkeypatch.setattr(state, "queued_counts", lambda: {})
    nemo_cfg = SimpleNamespace(container_name="nemo-clustering")
    monkeypatch.setattr(state, "managed_container", lambda model_id: None if model_id == "vibevoice" else nemo_cfg)

    rows = [
        ModelContainerState(model_id="vibevoice", active_job_count=0),
        ModelContainerState(model_id="nemo-clustering", active_job_count=0),
    ]
    statuses = state.derive_statuses(rows)

    assert [s.model_id for s in statuses] == ["nemo-clustering"]
