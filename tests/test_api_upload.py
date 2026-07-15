"""Integration tests for POST /upload — dual-lane storage, per-model queued
rows, one RQ job per model. Storage clients (MinIO/Azure Blob) are stubbed
so these tests never touch real infrastructure.
"""

import io
import shutil
import wave
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from rq import Queue
from sqlalchemy.orm import Session, sessionmaker

from apps.backend_api.routers.upload import _transcode_to_wav_file
from packages.database.models import AudioFile, EvaluationResult
from tests.conftest import make_wav_bytes

SAMPLE_MP3 = Path(__file__).parent / "samples" / "youtube-video-en-two-speaker.mp3"
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


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


def test_upload_rejects_unknown_model_ids_by_name(client: TestClient, stub_s3: list[str], fake_queue: Queue) -> None:
    response = client.post("/upload?models=pyannote,not-a-model", files={"file": ("clip.wav", make_wav_bytes(), "audio/wav")})
    assert response.status_code == 422
    assert response.json()["detail"] == "Unknown model id(s): not-a-model"
    assert fake_queue.count == 0  # nothing enqueued, including the known model


def test_transcode_to_wav_returns_none_on_garbage_bytes() -> None:
    # None either way: ffmpeg absent, or present but can't decode non-audio bytes.
    assert _transcode_to_wav_file(b"not audio at all") is None


@needs_ffmpeg
def test_upload_mp3_is_transcoded_and_stored_as_valid_wav(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, db_session_factory: sessionmaker[Session]
) -> None:
    stored: dict[str, bytes] = {}
    monkeypatch.setattr(
        "apps.backend_api.routers.upload.s3_client.put_stream",
        lambda fileobj, key: stored.__setitem__(key, fileobj.read()) or key,
    )

    response = client.post(
        "/upload?models=pyannote",
        files={"file": ("clip.mp3", SAMPLE_MP3.read_bytes(), "audio/mpeg")},
    )
    assert response.status_code == 200
    audio_file_id = response.json()["audioFileId"]

    # Stored under a .wav key even though the upload was .mp3.
    key = f"audio/{audio_file_id}.wav"
    assert list(stored) == [key]

    # The stored object is a canonical 16 kHz mono 16-bit PCM WAV with the sample's true
    # ~5:06 duration (guards the seekable-output regression: a piped header would report a
    # bogus length).
    with wave.open(io.BytesIO(stored[key]), "rb") as wav:
        assert wav.getsampwidth() == 2
        assert wav.getcomptype() == "NONE"
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert 305.0 < wav.getnframes() / wav.getframerate() < 308.0

    with db_session_factory() as session:
        audio_file = session.query(AudioFile).filter_by(id=audio_file_id).one()
        assert audio_file.filename == "clip.mp3"  # original name kept for display
        assert audio_file.s3_key == key
        assert 305.0 < audio_file.duration_sec < 308.0
