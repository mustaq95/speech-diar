"""The pyannote-adeo runner's HTTP contract with the LiteLLM gateway.

The gateway silently drops unknown form fields and 400s on every pyannoteAI
field the pod does not support, so the exact fields sent are pinned here, as
is the error path: the pod's per-field message lives in the response body and
must reach the model's row.
"""

from types import SimpleNamespace

import pytest

from apps.background_worker.models.pyannote_adeo import runner as adeo_runner
from tests.conftest import make_wav_bytes


def _fake_settings(**overrides):
    defaults = dict(
        diarize_url="https://gw.example/v1/diarize",
        litellm_api_key="k",
        pyannote_adeo_model="pyannote/community-1",
        pyannote_adeo_timeout_sec=600,
        litellm_httpx_verify=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _stub_post(monkeypatch, *, status_code=200, text="", payload=None, seen=None):
    def fake_post(url, headers=None, files=None, data=None, timeout=None, verify=None):
        if seen is not None:
            seen.update(url=url, headers=headers, files=files, data=data, timeout=timeout, verify=verify)
        return SimpleNamespace(status_code=status_code, text=text, json=lambda: payload or {})

    monkeypatch.setattr(adeo_runner.httpx, "post", fake_post)


def test_runner_posts_the_pyannoteai_contract(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    wav = tmp_path / "clip.wav"
    wav.write_bytes(make_wav_bytes(duration_sec=0.5))
    monkeypatch.setattr(adeo_runner, "get_settings", lambda: _fake_settings())
    job = {"jobId": "j", "status": "succeeded", "output": {"diarization": []}}
    seen: dict = {}
    _stub_post(monkeypatch, payload=job, seen=seen)

    raw = adeo_runner.PyannoteAdeoRunner().run(str(wav))

    assert seen["url"] == "https://gw.example/v1/diarize"
    assert seen["headers"] == {"Authorization": "Bearer k"}
    assert set(seen["files"]) == {"file"}
    assert seen["data"] == {"model": "pyannote/community-1", "exclusive": "true"}
    assert seen["timeout"] == 600
    assert seen["verify"] is False
    assert raw["job"] == job  # native job kept verbatim
    assert raw["audio_duration_sec"] == pytest.approx(0.5)


def test_runner_surfaces_the_gateway_error_body(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    wav = tmp_path / "clip.wav"
    wav.write_bytes(make_wav_bytes(duration_sec=0.5))
    monkeypatch.setattr(adeo_runner, "get_settings", lambda: _fake_settings())
    body = '{"error":{"message":"litellm.BadRequestError: PyannoteException numSpeakers"}}'
    _stub_post(monkeypatch, status_code=400, text=body)

    with pytest.raises(RuntimeError, match="pyannote-adeo 400: .*PyannoteException"):
        adeo_runner.PyannoteAdeoRunner().run(str(wav))


def test_runner_refuses_without_gateway_config(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(adeo_runner, "get_settings", lambda: _fake_settings(diarize_url=None))
    with pytest.raises(RuntimeError, match="LITELLM_BASE_URL"):
        adeo_runner.PyannoteAdeoRunner().run(str(tmp_path / "missing.wav"))
