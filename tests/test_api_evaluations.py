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
    monkeypatch.setattr("apps.backend_api.routers.evaluations.s3_client.head_object", lambda key: len(b"fake-wav-bytes"))
    monkeypatch.setattr("apps.backend_api.routers.evaluations.s3_client.open_stream", lambda key, start=0: io.BytesIO(b"fake-wav-bytes"[start:]))

    response = client.get(f"/evaluations/{audio_file_id}/audio")
    assert response.status_code == 200
    assert response.content == b"fake-wav-bytes"
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["content-length"] == str(len(b"fake-wav-bytes"))


def test_audio_proxy_streams_from_blob_when_only_blob_key_set(
    client: TestClient, db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_file_id = _seed_audio_file(db_session_factory, blob_key="uploads/abc.wav")

    class FakeDownloader:
        def __init__(self, start: int = 0) -> None:
            self._start = start

        def chunks(self):
            yield b"blob-bytes"[self._start :]

    monkeypatch.setattr("apps.backend_api.routers.evaluations.azure_blob.blob_size", lambda key: len(b"blob-bytes"))
    monkeypatch.setattr("apps.backend_api.routers.evaluations.azure_blob.open_stream", lambda key, start=0: FakeDownloader(start))

    response = client.get(f"/evaluations/{audio_file_id}/audio")
    assert response.status_code == 200
    assert response.content == b"blob-bytes"


def test_audio_proxy_404_when_neither_lane_has_audio(client: TestClient, db_session_factory: sessionmaker[Session]) -> None:
    audio_file_id = _seed_audio_file(db_session_factory)
    response = client.get(f"/evaluations/{audio_file_id}/audio")
    assert response.status_code == 404


# --- Range requests (seeking into long audio without a full re-download) --


def test_audio_proxy_serves_206_partial_content_for_mid_file_range(
    client: TestClient, db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_file_id = _seed_audio_file(db_session_factory, s3_key="audio/1.wav")
    full = bytes(range(100))

    monkeypatch.setattr("apps.backend_api.routers.evaluations.s3_client.head_object", lambda key: len(full))
    monkeypatch.setattr("apps.backend_api.routers.evaluations.s3_client.open_stream", lambda key, start=0: io.BytesIO(full[start:]))

    response = client.get(f"/evaluations/{audio_file_id}/audio", headers={"Range": "bytes=20-29"})

    assert response.status_code == 206
    assert response.content == full[20:30]
    assert response.headers["content-range"] == f"bytes 20-29/{len(full)}"
    assert response.headers["content-length"] == "10"
    assert response.headers["accept-ranges"] == "bytes"


def test_audio_proxy_serves_206_for_open_ended_range_to_seek_near_the_end(
    client: TestClient, db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`bytes=90-` (no end) is what a browser sends when seeking near the end
    of a long file — must stream to EOF, not just the first chunk."""
    audio_file_id = _seed_audio_file(db_session_factory, s3_key="audio/1.wav")
    full = bytes(range(100))

    monkeypatch.setattr("apps.backend_api.routers.evaluations.s3_client.head_object", lambda key: len(full))
    monkeypatch.setattr("apps.backend_api.routers.evaluations.s3_client.open_stream", lambda key, start=0: io.BytesIO(full[start:]))

    response = client.get(f"/evaluations/{audio_file_id}/audio", headers={"Range": "bytes=90-"})

    assert response.status_code == 206
    assert response.content == full[90:]
    assert response.headers["content-range"] == f"bytes 90-99/{len(full)}"


def test_audio_proxy_range_works_for_blob_lane_too(
    client: TestClient, db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_file_id = _seed_audio_file(db_session_factory, blob_key="uploads/abc.wav")
    full = bytes(range(50))

    class FakeDownloader:
        def __init__(self, start: int = 0) -> None:
            self._data = full[start:]

        def chunks(self):
            yield self._data

    monkeypatch.setattr("apps.backend_api.routers.evaluations.azure_blob.blob_size", lambda key: len(full))
    monkeypatch.setattr("apps.backend_api.routers.evaluations.azure_blob.open_stream", lambda key, start=0: FakeDownloader(start))

    response = client.get(f"/evaluations/{audio_file_id}/audio", headers={"Range": "bytes=10-19"})

    assert response.status_code == 206
    assert response.content == full[10:20]
    assert response.headers["content-range"] == f"bytes 10-19/{len(full)}"


def test_audio_proxy_falls_back_to_full_200_for_malformed_range(
    client: TestClient, db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_file_id = _seed_audio_file(db_session_factory, s3_key="audio/1.wav")
    full = bytes(range(50))

    monkeypatch.setattr("apps.backend_api.routers.evaluations.s3_client.head_object", lambda key: len(full))
    monkeypatch.setattr("apps.backend_api.routers.evaluations.s3_client.open_stream", lambda key, start=0: io.BytesIO(full[start:]))

    response = client.get(f"/evaluations/{audio_file_id}/audio", headers={"Range": "not-a-real-range"})

    assert response.status_code == 200
    assert response.content == full


def test_audio_proxy_resumes_from_offset_after_mid_stream_s3_failure(
    client: TestClient, db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single read failure partway through a large file (e.g. IncompleteRead)
    must not truncate the response — the proxy should reopen at the last
    delivered byte and complete the stream."""
    audio_file_id = _seed_audio_file(db_session_factory, s3_key="audio/1.wav")
    full = bytes(range(50))
    opened_at: list[int] = []
    reads = {"count": 0}

    class FlakyStream:
        def __init__(self, start: int):
            self._data = full[start:]

        def read(self, size: int) -> bytes:
            reads["count"] += 1
            if reads["count"] == 3:
                raise ConnectionError("simulated mid-stream read failure")
            chunk, self._data = self._data[:10], self._data[10:]
            return chunk

    def fake_open_stream(key: str, start: int = 0):
        opened_at.append(start)
        return FlakyStream(start)

    monkeypatch.setattr("apps.backend_api.routers.evaluations.s3_client.head_object", lambda key: len(full))
    monkeypatch.setattr("apps.backend_api.routers.evaluations.s3_client.open_stream", fake_open_stream)

    response = client.get(f"/evaluations/{audio_file_id}/audio")
    assert response.status_code == 200
    assert response.content == full
    assert opened_at == [0, 20]  # failed after 20 bytes delivered; resumed from there, not from 0


def test_audio_proxy_resumes_from_offset_after_mid_stream_blob_failure(
    client: TestClient, db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_file_id = _seed_audio_file(db_session_factory, blob_key="uploads/abc.wav")
    full = bytes(range(50))
    opened_at: list[int] = []
    yields = {"count": 0}

    class FakeDownloader:
        def __init__(self, start: int):
            self._data = full[start:]

        def chunks(self):
            data = self._data
            while data:
                yields["count"] += 1
                if yields["count"] == 3:
                    raise ConnectionError("simulated mid-stream read failure")
                chunk, data = data[:10], data[10:]
                yield chunk

    def fake_open_stream(blob_key: str, start: int = 0):
        opened_at.append(start)
        return FakeDownloader(start)

    monkeypatch.setattr("apps.backend_api.routers.evaluations.azure_blob.blob_size", lambda key: len(full))
    monkeypatch.setattr("apps.backend_api.routers.evaluations.azure_blob.open_stream", fake_open_stream)

    response = client.get(f"/evaluations/{audio_file_id}/audio")
    assert response.status_code == 200
    assert response.content == full
    assert opened_at == [0, 20]


def test_audio_proxy_gives_up_after_max_retries(
    client: TestClient, db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store that never recovers must still surface an error instead of
    retrying forever."""
    audio_file_id = _seed_audio_file(db_session_factory, s3_key="audio/1.wav")

    class AlwaysFailsStream:
        def read(self, size: int) -> bytes:
            raise ConnectionError("store is down")

    monkeypatch.setattr("apps.backend_api.routers.evaluations.s3_client.head_object", lambda key: 1024)
    monkeypatch.setattr("apps.backend_api.routers.evaluations.s3_client.open_stream", lambda key, start=0: AlwaysFailsStream())

    with pytest.raises(Exception):
        client.get(f"/evaluations/{audio_file_id}/audio")
