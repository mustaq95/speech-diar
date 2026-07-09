"""Integration tests for POST /upload — dual-lane storage, per-model queued
rows, one RQ job per model. Storage clients (MinIO/Azure Blob) are stubbed
so these tests never touch real infrastructure.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from rq import Queue
from sqlalchemy.orm import Session, sessionmaker

from packages.database.models import AudioFile, EvaluationResult
from tests.conftest import make_wav_bytes


@pytest.fixture()
def stub_s3(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Records keys passed to the local (MinIO) lane's put_stream — never a real S3 call."""
    calls: list[str] = []
    monkeypatch.setattr("apps.backend_api.routers.upload.s3_client.put_stream", lambda fileobj, key: calls.append(key) or key)
    yield calls


@pytest.fixture()
def stub_azure_blob_configured(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict]:
    """A fake Azure Blob store: configured, empty, and tracks uploads/existence checks."""
    state = {"configured": True, "blobs": set(), "put_calls": []}
    monkeypatch.setattr("apps.backend_api.routers.upload.azure_blob.storage_configured", lambda: state["configured"])
    monkeypatch.setattr("apps.backend_api.routers.upload.azure_blob.blob_exists", lambda key: key in state["blobs"])

    def fake_put_stream(fileobj, key: str) -> str:
        state["put_calls"].append(key)
        state["blobs"].add(key)
        return key

    monkeypatch.setattr("apps.backend_api.routers.upload.azure_blob.put_stream", fake_put_stream)
    monkeypatch.setattr("apps.backend_api.routers.upload.azure_blob.blob_url", lambda key: f"https://fake.blob.core.windows.net/container/{key}")
    yield state


def test_upload_rejects_non_wav_content(client: TestClient, stub_s3: list[str]) -> None:
    response = client.post("/upload?models=pyannote", files={"file": ("clip.wav", b"not a real wav file", "audio/wav")})
    assert response.status_code == 415


def test_upload_rejects_when_no_valid_model_ids(client: TestClient, stub_s3: list[str]) -> None:
    response = client.post("/upload?models=not-a-real-model", files={"file": ("clip.wav", make_wav_bytes(), "audio/wav")})
    assert response.status_code == 422


def test_upload_local_model_creates_audio_file_and_queued_result(
    client: TestClient, stub_s3: list[str], db_session_factory: sessionmaker[Session], fake_queue: Queue
) -> None:
    response = client.post("/upload?models=pyannote", files={"file": ("clip.wav", make_wav_bytes(2.0), "audio/wav")})
    assert response.status_code == 200
    body = response.json()
    assert body["models"] == [{"id": "pyannote", "status": "queued"}]
    audio_file_id = body["audioFileId"]

    assert stub_s3 == [f"audio/{audio_file_id}.wav"]
    assert fake_queue.count == 1

    with db_session_factory() as session:
        audio_file = session.query(AudioFile).filter_by(id=audio_file_id).one()
        assert audio_file.s3_key == f"audio/{audio_file_id}.wav"
        assert audio_file.blob_key is None
        result = session.query(EvaluationResult).filter_by(audio_file_id=audio_file_id).one()
        assert result.model_id == "pyannote"
        assert result.status == "queued"


def test_upload_azure_model_without_configured_storage_returns_422(client: TestClient, stub_s3: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("apps.backend_api.routers.upload.azure_blob.storage_configured", lambda: False)
    response = client.post("/upload?models=azure-batch", files={"file": ("clip.wav", make_wav_bytes(), "audio/wav")})
    assert response.status_code == 422
    assert "AZURE_STORAGE" in response.json()["detail"]


def test_upload_azure_model_stores_in_blob_only_not_minio(
    client: TestClient,
    stub_s3: list[str],
    stub_azure_blob_configured: dict,
    db_session_factory: sessionmaker[Session],
    fake_queue: Queue,
) -> None:
    response = client.post("/upload?models=azure-batch", files={"file": ("clip.wav", make_wav_bytes(), "audio/wav")})
    assert response.status_code == 200
    audio_file_id = response.json()["audioFileId"]

    assert stub_s3 == []  # the azure lane never touches MinIO
    assert len(stub_azure_blob_configured["put_calls"]) == 1

    with db_session_factory() as session:
        audio_file = session.query(AudioFile).filter_by(id=audio_file_id).one()
        assert audio_file.s3_key is None
        assert audio_file.blob_key is not None
        assert audio_file.blob_url is not None


def test_upload_mixed_models_writes_to_both_lanes_independently(
    client: TestClient, stub_s3: list[str], stub_azure_blob_configured: dict, fake_queue: Queue
) -> None:
    response = client.post("/upload?models=pyannote,azure-batch", files={"file": ("clip.wav", make_wav_bytes(), "audio/wav")})
    assert response.status_code == 200
    body = response.json()
    assert {m["id"] for m in body["models"]} == {"pyannote", "azure-batch"}
    assert len(stub_s3) == 1
    assert len(stub_azure_blob_configured["put_calls"]) == 1
    assert fake_queue.count == 2  # one RQ job per model


def test_upload_same_content_twice_reuses_existing_blob(client: TestClient, stub_s3: list[str], stub_azure_blob_configured: dict) -> None:
    content = make_wav_bytes(3.0)
    first = client.post("/upload?models=azure-batch", files={"file": ("clip.wav", content, "audio/wav")})
    second = client.post("/upload?models=azure-batch", files={"file": ("clip.wav", content, "audio/wav")})
    assert first.status_code == 200
    assert second.status_code == 200

    # Same content -> same blob key -> the second upload must not re-upload.
    assert len(stub_azure_blob_configured["put_calls"]) == 1
    assert first.json()["audioFileId"] != second.json()["audioFileId"]  # still two distinct evaluation rows


def test_upload_unknown_and_known_model_ids_keeps_only_known_ones(client: TestClient, stub_s3: list[str], fake_queue: Queue) -> None:
    response = client.post("/upload?models=pyannote,not-a-model", files={"file": ("clip.wav", make_wav_bytes(), "audio/wav")})
    assert response.status_code == 200
    assert [m["id"] for m in response.json()["models"]] == ["pyannote"]
