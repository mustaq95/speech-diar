"""Integration tests for the two worker pipelines (local_pipeline.py /
azure_pipeline.py): each must take one `EvaluationResult` row from
queued -> running -> done|failed, writing real timing, the adapted payload and
the engine's raw native output — no lane ever touches the other lane's storage
function.
"""

import io
import wave
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy.orm import Session, sessionmaker, undefer

from apps.background_worker.pipelines import azure_pipeline, local_pipeline
from packages.database.models import AudioFile, EvaluationResult
from packages.shared_contracts.schemas import DiarizationModelRun, DiarizationSegment
from tests.conftest import make_wav_bytes
from tests.test_wav_duration import _corrupt_data_chunk_size


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
        row = (
            session.query(EvaluationResult)
            # raw_output is deferred on the model, so it is NOT loaded by
            # default -- and the expunge below detaches the row, after which
            # loading it is impossible (DetachedInstanceError). Undefer here so
            # callers can assert on it.
            .options(undefer(EvaluationResult.raw_output))
            .filter_by(audio_file_id=audio_file_id, model_id=model_id)
            .one()
        )
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
    # The engine's native output, verbatim beside the adapted payload.
    assert result.raw_output == {"native": "payload"}


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


def test_local_pipeline_segment_far_past_duration_marks_failed(
    db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model that reports a segment ending far past the audio's real
    duration (e.g. an autoregressive engine that hallucinates past the end
    of the input, as observed with VibeVoice) must not be shown as a valid
    'done' result -- the run is discarded and marked failed instead."""
    audio_file_id = _seed(db_session_factory, s3_key="audio/overrun.wav", blob_key=None)
    fake_model = FakeModel()
    fake_model.adapter.adapt = lambda raw: DiarizationModelRun(
        id="fake",
        name="Fake Model",
        short="Fake",
        description="test double",
        segs=[DiarizationSegment(spk=0, s=0.0, e=100.0)],  # duration_sec=10.0 in _seed
    )

    monkeypatch.setattr(local_pipeline, "SessionLocal", db_session_factory)
    monkeypatch.setattr(local_pipeline, "REGISTRY", {"fake": fake_model})
    monkeypatch.setattr(local_pipeline, "download_to", lambda key, path: None)

    local_pipeline.run_local_model(audio_file_id, "fake")

    result = _status(db_session_factory, audio_file_id, "fake")
    assert result.status == "failed"
    assert "100.00s" in result.error
    assert "10.00s" in result.error


def test_local_pipeline_unknown_model_id_marks_failed(db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch) -> None:
    audio_file_id = _seed(db_session_factory, s3_key="audio/4.wav", blob_key=None, model_id="ghost")
    monkeypatch.setattr(local_pipeline, "SessionLocal", db_session_factory)
    monkeypatch.setattr(local_pipeline, "REGISTRY", {})

    local_pipeline.run_local_model(audio_file_id, "ghost")

    result = _status(db_session_factory, audio_file_id, "ghost")
    assert result.status == "failed"
    assert "Unknown model id" in result.error


# --- azure_pipeline (Blob) -----------------------------------------------


def _stub_well_formed_blob(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixed blob doesn't exist yet, and the source blob's header already
    matches its real byte count -- the common case, where no repair/reupload
    should happen and the original key is SASed directly."""
    monkeypatch.setattr(azure_pipeline.azure_blob, "blob_exists", lambda key: False)
    monkeypatch.setattr(azure_pipeline.azure_blob, "open_stream", lambda key: SimpleNamespace(readall=lambda: make_wav_bytes(1.0)))
    monkeypatch.setattr(azure_pipeline.azure_blob, "put_stream", lambda fileobj, key: pytest.fail("must not reupload a well-formed blob"))


def test_azure_pipeline_success_prefers_adapter_reported_processing_ms(
    db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_file_id = _seed(db_session_factory, s3_key=None, blob_key="uploads/abc.wav")
    fake_model = FakeModel(processing_ms=4242)

    monkeypatch.setattr(azure_pipeline, "SessionLocal", db_session_factory)
    monkeypatch.setattr(azure_pipeline, "REGISTRY", {"fake": fake_model})
    monkeypatch.setattr(azure_pipeline, "read_sas_url", lambda blob_key: f"https://example.test/{blob_key}?sas=1")
    _stub_well_formed_blob(monkeypatch)

    azure_pipeline.run_azure_model(audio_file_id, "fake")

    assert fake_model.received_input == "https://example.test/uploads/abc.wav?sas=1"
    result = _status(db_session_factory, audio_file_id, "fake")
    assert result.status == "done"
    assert result.processing_ms == 4242  # Azure's own reported duration, not wall-clock
    assert result.raw_output == {"native": "payload"}


def test_azure_pipeline_falls_back_to_wall_clock_when_adapter_reports_none(
    db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_file_id = _seed(db_session_factory, s3_key=None, blob_key="uploads/def.wav")
    fake_model = FakeModel(processing_ms=None)

    monkeypatch.setattr(azure_pipeline, "SessionLocal", db_session_factory)
    monkeypatch.setattr(azure_pipeline, "REGISTRY", {"fake": fake_model})
    monkeypatch.setattr(azure_pipeline, "read_sas_url", lambda blob_key: "https://example.test/x")
    _stub_well_formed_blob(monkeypatch)

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
    monkeypatch.setattr(azure_pipeline.azure_blob, "blob_exists", lambda key: pytest.fail("must not touch Azure Blob"))

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
    monkeypatch.setattr(azure_pipeline.azure_blob, "blob_exists", lambda key: pytest.fail("must not touch Azure Blob"))

    azure_pipeline.run_azure_model(9999, "fake")  # no such audio_file_id/result row

    assert fake_model.received_input is None


def test_azure_pipeline_engine_failure_marks_failed(db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch) -> None:
    audio_file_id = _seed(db_session_factory, s3_key=None, blob_key="uploads/err.wav")
    fake_model = FakeModel(error=RuntimeError("Azure said no"))

    monkeypatch.setattr(azure_pipeline, "SessionLocal", db_session_factory)
    monkeypatch.setattr(azure_pipeline, "REGISTRY", {"fake": fake_model})
    monkeypatch.setattr(azure_pipeline, "read_sas_url", lambda blob_key: "https://example.test/x")
    _stub_well_formed_blob(monkeypatch)

    azure_pipeline.run_azure_model(audio_file_id, "fake")

    result = _status(db_session_factory, audio_file_id, "fake")
    assert result.status == "failed"
    assert "Azure said no" in result.error


# --- azure_pipeline: session lifetime -------------------------------------


class _SessionTracker:
    """Wraps a session factory and counts how many sessions are currently open."""

    def __init__(self, factory: sessionmaker[Session]) -> None:
        self._factory = factory
        self.open_count = 0

    def __call__(self) -> Session:
        session = self._factory()
        self.open_count += 1
        original_close = session.close
        closed = False

        def close(*args: Any, **kwargs: Any) -> None:
            nonlocal closed
            if not closed:
                closed = True
                self.open_count -= 1
            original_close(*args, **kwargs)

        session.close = close  # type: ignore[method-assign]
        return session


def _sessions_open_while_reaching_azure(
    db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch, fake_model: FakeModel
) -> list[int]:
    """Installs a tracked session factory and records the open-session count at
    each point where the pipeline talks to Azure."""
    tracker = _SessionTracker(db_session_factory)
    counts: list[int] = []
    engine_run = fake_model.runner.run

    def run(audio_input: str, params: dict[str, Any] | None = None) -> dict:
        counts.append(tracker.open_count)
        return engine_run(audio_input, params)

    fake_model.runner = SimpleNamespace(run=run)
    monkeypatch.setattr(azure_pipeline, "SessionLocal", tracker)
    monkeypatch.setattr(azure_pipeline, "REGISTRY", {"fake": fake_model})
    monkeypatch.setattr(azure_pipeline, "read_sas_url", lambda blob_key: counts.append(tracker.open_count) or "https://example.test/x")
    _stub_well_formed_blob(monkeypatch)
    return counts


def test_azure_pipeline_holds_no_db_session_while_the_batch_job_runs(
    db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression. An Azure batch job runs for minutes on real audio, far past
    the 60 s `idle_in_transaction_session_timeout` that
    packages/database/session.py sets on every connection. Holding a session
    open across it got the connection killed mid-job, so `mark_done` raised —
    and so did the `mark_failed` in the except arm, since it shared the same
    dead session. Nothing wrote a terminal status and the row sat at "running"
    forever while the UI counted up. Caught live on a 7:40 recording."""
    audio_file_id = _seed(db_session_factory, s3_key=None, blob_key="uploads/slow.wav")
    fake_model = FakeModel(processing_ms=4242)
    counts = _sessions_open_while_reaching_azure(db_session_factory, monkeypatch, fake_model)

    azure_pipeline.run_azure_model(audio_file_id, "fake")

    assert counts == [0, 0]  # SAS build and engine call, both with no session held
    result = _status(db_session_factory, audio_file_id, "fake")
    assert result.status == "done"
    assert result.processing_ms == 4242


def test_azure_pipeline_engine_failure_marks_failed_on_a_fresh_session(
    db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The half of the same bug that hid it: with no session open when the
    engine raises, `mark_failed` has to open its own, so the row still reaches
    a terminal state."""
    audio_file_id = _seed(db_session_factory, s3_key=None, blob_key="uploads/slow.wav")
    fake_model = FakeModel(error=RuntimeError("Azure said no"))
    counts = _sessions_open_while_reaching_azure(db_session_factory, monkeypatch, fake_model)

    azure_pipeline.run_azure_model(audio_file_id, "fake")

    assert counts == [0, 0]
    result = _status(db_session_factory, audio_file_id, "fake")
    assert result.status == "failed"
    assert "Azure said no" in result.error


# --- azure_pipeline: bad-header repair ------------------------------------


def test_azure_pipeline_repairs_bad_header_before_generating_sas(
    db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact bug: a blob with a placeholder `data` chunk size (what Azure
    Batch rejects as InvalidData) gets repaired into a derived key, and the
    SAS URL points at that repaired key instead of the original."""
    audio_file_id = _seed(db_session_factory, s3_key=None, blob_key="uploads/bad.wav")
    fake_model = FakeModel()
    corrupted = _corrupt_data_chunk_size(make_wav_bytes(1.0), bogus_size=1_000_000_000)
    put_calls: list[tuple[str, bytes]] = []

    monkeypatch.setattr(azure_pipeline, "SessionLocal", db_session_factory)
    monkeypatch.setattr(azure_pipeline, "REGISTRY", {"fake": fake_model})
    monkeypatch.setattr(azure_pipeline, "read_sas_url", lambda blob_key: f"https://example.test/{blob_key}?sas=1")
    monkeypatch.setattr(azure_pipeline.azure_blob, "blob_exists", lambda key: False)
    monkeypatch.setattr(azure_pipeline.azure_blob, "open_stream", lambda key: SimpleNamespace(readall=lambda: corrupted))
    monkeypatch.setattr(azure_pipeline.azure_blob, "put_stream", lambda fileobj, key: put_calls.append((key, fileobj.read())) or key)

    azure_pipeline.run_azure_model(audio_file_id, "fake")

    assert len(put_calls) == 1
    fixed_key, fixed_bytes = put_calls[0]
    assert fixed_key == "uploads/bad-fixed.wav"
    assert azure_pipeline._header_is_valid(fixed_bytes)
    assert fake_model.received_input == f"https://example.test/{fixed_key}?sas=1"
    result = _status(db_session_factory, audio_file_id, "fake")
    assert result.status == "done"


def test_azure_pipeline_downmixes_stereo_even_with_a_well_formed_header(
    db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Azure Batch diarization rejects stereo audio outright (also surfaced
    as InvalidData), so a stereo blob must be downmixed to mono even when its
    header already correctly describes the stereo data."""
    audio_file_id = _seed(db_session_factory, s3_key=None, blob_key="uploads/stereo.wav")
    fake_model = FakeModel()
    stereo_frames = int(1.0 * 16000)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00\x00\x00" * stereo_frames)
    stereo_wav = buffer.getvalue()
    put_calls: list[tuple[str, bytes]] = []

    monkeypatch.setattr(azure_pipeline, "SessionLocal", db_session_factory)
    monkeypatch.setattr(azure_pipeline, "REGISTRY", {"fake": fake_model})
    monkeypatch.setattr(azure_pipeline, "read_sas_url", lambda blob_key: f"https://example.test/{blob_key}?sas=1")
    monkeypatch.setattr(azure_pipeline.azure_blob, "blob_exists", lambda key: False)
    monkeypatch.setattr(azure_pipeline.azure_blob, "open_stream", lambda key: SimpleNamespace(readall=lambda: stereo_wav))
    monkeypatch.setattr(azure_pipeline.azure_blob, "put_stream", lambda fileobj, key: put_calls.append((key, fileobj.read())) or key)

    azure_pipeline.run_azure_model(audio_file_id, "fake")

    assert len(put_calls) == 1
    fixed_key, fixed_bytes = put_calls[0]
    assert fixed_key == "uploads/stereo-fixed.wav"
    with wave.open(io.BytesIO(fixed_bytes), "rb") as fixed_wav:
        assert fixed_wav.getnchannels() == 1
    assert fake_model.received_input == f"https://example.test/{fixed_key}?sas=1"
    result = _status(db_session_factory, audio_file_id, "fake")
    assert result.status == "done"


def test_azure_pipeline_skips_reupload_when_fixed_blob_already_exists(
    db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running against audio that's already been repaired (e.g. a repeat
    evaluation) must SAS off the existing repaired blob, not reconvert it."""
    audio_file_id = _seed(db_session_factory, s3_key=None, blob_key="uploads/bad.wav")
    fake_model = FakeModel()

    monkeypatch.setattr(azure_pipeline, "SessionLocal", db_session_factory)
    monkeypatch.setattr(azure_pipeline, "REGISTRY", {"fake": fake_model})
    monkeypatch.setattr(azure_pipeline, "read_sas_url", lambda blob_key: f"https://example.test/{blob_key}?sas=1")
    monkeypatch.setattr(azure_pipeline.azure_blob, "blob_exists", lambda key: True)
    monkeypatch.setattr(azure_pipeline.azure_blob, "open_stream", lambda key: pytest.fail("must not redownload an already-repaired blob"))
    monkeypatch.setattr(azure_pipeline.azure_blob, "put_stream", lambda fileobj, key: pytest.fail("must not re-reupload an already-repaired blob"))

    azure_pipeline.run_azure_model(audio_file_id, "fake")

    assert fake_model.received_input == "https://example.test/uploads/bad-fixed.wav?sas=1"
    result = _status(db_session_factory, audio_file_id, "fake")
    assert result.status == "done"
