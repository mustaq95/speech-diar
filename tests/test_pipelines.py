"""Integration tests for the two worker pipelines (local_pipeline.py /
azure_pipeline.py): each must take one `EvaluationResult` row from
queued -> running -> done|failed, writing real timing and payload — no
lane ever touches the other lane's storage function.
"""

from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy.orm import Session, sessionmaker

from apps.background_worker.pipelines import azure_pipeline, local_pipeline
from packages.database.models import AudioFile, EvaluationResult
from packages.shared_contracts.schemas import DiarizationModelRun, DiarizationSegment


class FakeModel:
    """A minimal stand-in for `DiarizationModel` (runner + adapter)."""

    def __init__(self, *, error: Exception | None = None, processing_ms: int | None = None) -> None:
        self._error = error
        self._processing_ms = processing_ms
        self.received_input: str | None = None
        self.runner = SimpleNamespace(run=self._run)
        self.adapter = SimpleNamespace(adapt=self._adapt, processing_ms=lambda raw: self._processing_ms)

    def _run(self, audio_input: str, params: dict[str, Any] | None = None) -> dict:
        self.received_input = audio_input
        if self._error:
            raise self._error
        return {"native": "payload"}

    def _adapt(self, raw: dict) -> DiarizationModelRun:
        return DiarizationModelRun(
            id="fake",
            name="Fake Model",
            short="Fake",
            description="test double",
            segs=[DiarizationSegment(spk=0, s=0.0, e=1.0)],
        )


def _seed(db_session_factory: sessionmaker[Session], *, s3_key: str | None, blob_key: str | None, model_id: str = "fake") -> int:
    with db_session_factory() as session:
        audio_file = AudioFile(owner_id=1, filename="clip.wav", duration_sec=10.0, s3_key=s3_key, blob_key=blob_key)
        session.add(audio_file)
        session.commit()
        session.add(EvaluationResult(audio_file_id=audio_file.id, model_id=model_id, status="queued"))
        session.commit()
        return audio_file.id


def _status(db_session_factory: sessionmaker[Session], audio_file_id: int, model_id: str) -> EvaluationResult:
    with db_session_factory() as session:
        row = session.query(EvaluationResult).filter_by(audio_file_id=audio_file_id, model_id=model_id).one()
        session.expunge(row)
        return row


# --- local_pipeline (MinIO) ---------------------------------------------


def test_local_pipeline_success_marks_done_with_payload_and_timing(
    db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_file_id = _seed(db_session_factory, s3_key="audio/1.wav", blob_key=None)
    fake_model = FakeModel()
    downloaded: list[str] = []

    monkeypatch.setattr(local_pipeline, "SessionLocal", db_session_factory)
    monkeypatch.setattr(local_pipeline, "REGISTRY", {"fake": fake_model})
    monkeypatch.setattr(local_pipeline, "download_to", lambda key, path: downloaded.append(key))

    local_pipeline.run_local_model(audio_file_id, "fake")

    assert downloaded == ["audio/1.wav"]
    result = _status(db_session_factory, audio_file_id, "fake")
    assert result.status == "done"
    assert result.error is None
    assert result.started_at is not None and result.finished_at is not None
    assert result.processing_ms is not None and result.processing_ms >= 0
    assert result.payload["numSpk"] == 1


def test_local_pipeline_engine_failure_marks_failed_with_error(
    db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_file_id = _seed(db_session_factory, s3_key="audio/2.wav", blob_key=None)
    fake_model = FakeModel(error=RuntimeError("boom"))

    monkeypatch.setattr(local_pipeline, "SessionLocal", db_session_factory)
    monkeypatch.setattr(local_pipeline, "REGISTRY", {"fake": fake_model})
    monkeypatch.setattr(local_pipeline, "download_to", lambda key, path: None)

    local_pipeline.run_local_model(audio_file_id, "fake")

    result = _status(db_session_factory, audio_file_id, "fake")
    assert result.status == "failed"
    assert "boom" in result.error


def test_local_pipeline_unimplemented_runner_marks_failed(
    db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_file_id = _seed(db_session_factory, s3_key="audio/3.wav", blob_key=None)
    fake_model = FakeModel(error=NotImplementedError("not built yet"))

    monkeypatch.setattr(local_pipeline, "SessionLocal", db_session_factory)
    monkeypatch.setattr(local_pipeline, "REGISTRY", {"fake": fake_model})
    monkeypatch.setattr(local_pipeline, "download_to", lambda key, path: None)

    local_pipeline.run_local_model(audio_file_id, "fake")

    result = _status(db_session_factory, audio_file_id, "fake")
    assert result.status == "failed"
    assert "no runner implementation" in result.error


def test_local_pipeline_missing_s3_key_marks_failed_without_running_model(
    db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_file_id = _seed(db_session_factory, s3_key=None, blob_key=None)
    fake_model = FakeModel()

    monkeypatch.setattr(local_pipeline, "SessionLocal", db_session_factory)
    monkeypatch.setattr(local_pipeline, "REGISTRY", {"fake": fake_model})
    monkeypatch.setattr(local_pipeline, "download_to", lambda key, path: pytest.fail("must not download"))

    local_pipeline.run_local_model(audio_file_id, "fake")

    result = _status(db_session_factory, audio_file_id, "fake")
    assert result.status == "failed"
    assert "MinIO" in result.error
    assert fake_model.received_input is None


def test_local_pipeline_missing_result_row_logs_and_returns_without_crashing(
    db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale/orphaned job (its EvaluationResult row already gone) must not
    raise — RQ would otherwise log a raw traceback for something that isn't
    an application error."""
    fake_model = FakeModel()
    monkeypatch.setattr(local_pipeline, "SessionLocal", db_session_factory)
    monkeypatch.setattr(local_pipeline, "REGISTRY", {"fake": fake_model})
    monkeypatch.setattr(local_pipeline, "download_to", lambda key, path: pytest.fail("must not download"))

    local_pipeline.run_local_model(9999, "fake")  # no such audio_file_id/result row

    assert fake_model.received_input is None


def test_local_pipeline_unknown_model_id_marks_failed(db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch) -> None:
    audio_file_id = _seed(db_session_factory, s3_key="audio/4.wav", blob_key=None, model_id="ghost")
    monkeypatch.setattr(local_pipeline, "SessionLocal", db_session_factory)
    monkeypatch.setattr(local_pipeline, "REGISTRY", {})

    local_pipeline.run_local_model(audio_file_id, "ghost")

    result = _status(db_session_factory, audio_file_id, "ghost")
    assert result.status == "failed"
    assert "Unknown model id" in result.error


# --- azure_pipeline (Blob) -----------------------------------------------


def test_azure_pipeline_success_prefers_adapter_reported_processing_ms(
    db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_file_id = _seed(db_session_factory, s3_key=None, blob_key="uploads/abc.wav")
    fake_model = FakeModel(processing_ms=4242)

    monkeypatch.setattr(azure_pipeline, "SessionLocal", db_session_factory)
    monkeypatch.setattr(azure_pipeline, "REGISTRY", {"fake": fake_model})
    monkeypatch.setattr(azure_pipeline, "read_sas_url", lambda blob_key: f"https://example.test/{blob_key}?sas=1")

    azure_pipeline.run_azure_model(audio_file_id, "fake")

    assert fake_model.received_input == "https://example.test/uploads/abc.wav?sas=1"
    result = _status(db_session_factory, audio_file_id, "fake")
    assert result.status == "done"
    assert result.processing_ms == 4242  # Azure's own reported duration, not wall-clock


def test_azure_pipeline_falls_back_to_wall_clock_when_adapter_reports_none(
    db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_file_id = _seed(db_session_factory, s3_key=None, blob_key="uploads/def.wav")
    fake_model = FakeModel(processing_ms=None)

    monkeypatch.setattr(azure_pipeline, "SessionLocal", db_session_factory)
    monkeypatch.setattr(azure_pipeline, "REGISTRY", {"fake": fake_model})
    monkeypatch.setattr(azure_pipeline, "read_sas_url", lambda blob_key: "https://example.test/x")

    azure_pipeline.run_azure_model(audio_file_id, "fake")

    result = _status(db_session_factory, audio_file_id, "fake")
    assert result.status == "done"
    assert result.processing_ms is not None  # worker wall-clock fallback


def test_azure_pipeline_missing_blob_key_marks_failed_without_calling_azure(
    db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_file_id = _seed(db_session_factory, s3_key=None, blob_key=None)
    fake_model = FakeModel()

    monkeypatch.setattr(azure_pipeline, "SessionLocal", db_session_factory)
    monkeypatch.setattr(azure_pipeline, "REGISTRY", {"fake": fake_model})
    monkeypatch.setattr(azure_pipeline, "read_sas_url", lambda blob_key: pytest.fail("must not touch Azure Blob"))

    azure_pipeline.run_azure_model(audio_file_id, "fake")

    result = _status(db_session_factory, audio_file_id, "fake")
    assert result.status == "failed"
    assert "Azure Blob" in result.error
    assert fake_model.received_input is None


def test_azure_pipeline_missing_result_row_logs_and_returns_without_crashing(
    db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_model = FakeModel()
    monkeypatch.setattr(azure_pipeline, "SessionLocal", db_session_factory)
    monkeypatch.setattr(azure_pipeline, "REGISTRY", {"fake": fake_model})
    monkeypatch.setattr(azure_pipeline, "read_sas_url", lambda blob_key: pytest.fail("must not touch Azure Blob"))

    azure_pipeline.run_azure_model(9999, "fake")  # no such audio_file_id/result row

    assert fake_model.received_input is None


def test_azure_pipeline_engine_failure_marks_failed(db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch) -> None:
    audio_file_id = _seed(db_session_factory, s3_key=None, blob_key="uploads/err.wav")
    fake_model = FakeModel(error=RuntimeError("Azure said no"))

    monkeypatch.setattr(azure_pipeline, "SessionLocal", db_session_factory)
    monkeypatch.setattr(azure_pipeline, "REGISTRY", {"fake": fake_model})
    monkeypatch.setattr(azure_pipeline, "read_sas_url", lambda blob_key: "https://example.test/x")

    azure_pipeline.run_azure_model(audio_file_id, "fake")

    result = _status(db_session_factory, audio_file_id, "fake")
    assert result.status == "failed"
    assert "Azure said no" in result.error
