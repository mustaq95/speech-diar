"""Integration tests for GET/PATCH /evaluations/{id} and the audio proxy —
everything served must come straight from the database/storage, never
fabricated.
"""

import io

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from packages.database.models import AudioFile, EvaluationResult


def _seed_audio_file(db_session_factory: sessionmaker[Session], **kwargs) -> int:
    with db_session_factory() as session:
        audio_file = AudioFile(owner_id=1, filename="clip.wav", duration_sec=kwargs.pop("duration_sec", 10.0), **kwargs)
        session.add(audio_file)
        session.commit()
        return audio_file.id


def test_get_evaluation_404_for_unknown_id(client: TestClient) -> None:
    response = client.get("/evaluations/9999")
    assert response.status_code == 404


def test_get_evaluation_queued_model_has_no_segments_but_has_metadata(
    client: TestClient, db_session_factory: sessionmaker[Session]
) -> None:
    audio_file_id = _seed_audio_file(db_session_factory, duration_sec=12.5)
    with db_session_factory() as session:
        session.add(EvaluationResult(audio_file_id=audio_file_id, model_id="pyannote", status="queued"))
        session.commit()

    response = client.get(f"/evaluations/{audio_file_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["audioFileId"] == audio_file_id
    assert body["durationSec"] == 12.5
    assert body["uploadMs"] is None
    assert len(body["models"]) == 1
    model = body["models"][0]
    assert model["id"] == "pyannote"
    assert model["status"] == "queued"
    assert model["segs"] == []
    assert model["name"]  # filled in from the registry, not fabricated segments


def test_get_evaluation_done_model_returns_persisted_payload_and_timing(
    client: TestClient, db_session_factory: sessionmaker[Session]
) -> None:
    audio_file_id = _seed_audio_file(db_session_factory)
    payload = {
        "id": "pyannote",
        "name": "PyAnnote Audio 3.1",
        "short": "PyAnnote",
        "description": "d",
        "segs": [{"spk": 0, "s": 0.0, "e": 1.0}],
        "numSpk": 1,
    }
    with db_session_factory() as session:
        session.add(
            EvaluationResult(
                audio_file_id=audio_file_id,
                model_id="pyannote",
                status="done",
                payload=payload,
                processing_ms=123,
            )
        )
        session.commit()

    response = client.get(f"/evaluations/{audio_file_id}")
    model = response.json()["models"][0]
    assert model["status"] == "done"
    assert model["segs"] == [{"spk": 0, "s": 0.0, "e": 1.0}]
    assert model["numSpk"] == 1
    assert model["processingMs"] == 123


def test_get_evaluation_failed_model_surfaces_error(client: TestClient, db_session_factory: sessionmaker[Session]) -> None:
    audio_file_id = _seed_audio_file(db_session_factory)
    with db_session_factory() as session:
        session.add(EvaluationResult(audio_file_id=audio_file_id, model_id="azure-batch", status="failed", error="Azure Blob not configured"))
        session.commit()

    response = client.get(f"/evaluations/{audio_file_id}")
    model = response.json()["models"][0]
    assert model["status"] == "failed"
    assert model["error"] == "Azure Blob not configured"


def test_patch_evaluation_persists_upload_ms(client: TestClient, db_session_factory: sessionmaker[Session]) -> None:
    audio_file_id = _seed_audio_file(db_session_factory)
    response = client.patch(f"/evaluations/{audio_file_id}", json={"uploadMs": 987})
    assert response.status_code == 200
    assert response.json()["uploadMs"] == 987

    with db_session_factory() as session:
        assert session.query(AudioFile).filter_by(id=audio_file_id).one().upload_ms == 987


def test_audio_proxy_streams_from_minio_when_s3_key_set(
    client: TestClient, db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_file_id = _seed_audio_file(db_session_factory, s3_key="audio/1.wav")
    monkeypatch.setattr("apps.backend_api.routers.evaluations.s3_client.open_stream", lambda key: io.BytesIO(b"fake-wav-bytes"))

    response = client.get(f"/evaluations/{audio_file_id}/audio")
    assert response.status_code == 200
    assert response.content == b"fake-wav-bytes"
    assert response.headers["content-type"] == "audio/wav"


def test_audio_proxy_streams_from_blob_when_only_blob_key_set(
    client: TestClient, db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_file_id = _seed_audio_file(db_session_factory, blob_key="uploads/abc.wav")

    class FakeDownloader:
        def chunks(self):
            yield b"blob-bytes"

    monkeypatch.setattr("apps.backend_api.routers.evaluations.azure_blob.open_stream", lambda key: FakeDownloader())

    response = client.get(f"/evaluations/{audio_file_id}/audio")
    assert response.status_code == 200
    assert response.content == b"blob-bytes"


def test_audio_proxy_404_when_neither_lane_has_audio(client: TestClient, db_session_factory: sessionmaker[Session]) -> None:
    audio_file_id = _seed_audio_file(db_session_factory)
    response = client.get(f"/evaluations/{audio_file_id}/audio")
    assert response.status_code == 404
