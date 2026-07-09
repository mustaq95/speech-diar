"""Unit tests for apps/background_worker/worker.py — the RQ task must
dispatch purely by lane and never touch storage/DB itself."""

import pytest

from apps.background_worker import worker


def test_run_model_dispatches_local_lane_to_local_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, str]] = []
    monkeypatch.setattr(worker, "run_local_model", lambda audio_file_id, model_id: calls.append((audio_file_id, model_id)))
    monkeypatch.setattr(worker, "run_azure_model", lambda audio_file_id, model_id: pytest.fail("should not call azure pipeline"))

    worker.run_model(7, "pyannote")

    assert calls == [(7, "pyannote")]


def test_run_model_dispatches_azure_lane_to_azure_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, str]] = []
    monkeypatch.setattr(worker, "run_azure_model", lambda audio_file_id, model_id: calls.append((audio_file_id, model_id)))
    monkeypatch.setattr(worker, "run_local_model", lambda audio_file_id, model_id: pytest.fail("should not call local pipeline"))

    worker.run_model(3, "azure-batch")

    assert calls == [(3, "azure-batch")]


def test_run_model_unknown_model_id_calls_neither_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker, "run_local_model", lambda *a: pytest.fail("should not call local pipeline"))
    monkeypatch.setattr(worker, "run_azure_model", lambda *a: pytest.fail("should not call azure pipeline"))

    worker.run_model(1, "no-such-model")  # must not raise
