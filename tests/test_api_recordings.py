"""GET /recordings — the recordings list, scoped by surface.

This route exists to replace a pattern, so the tests assert the properties that
pattern got wrong: that the two surfaces do not leak into each other, that the
counts are computed server-side rather than left for the client to derive, and that
a recording with no results is still listed rather than quietly dropped.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from packages.database.models import (
    AudioFile,
    EvaluationResult,
    TranscriptReference,
    TranscriptResult,
    User,
)
from packages.database.session import DEV_USER_EMAIL


def _recording(factory: sessionmaker[Session], *, surface: str = "diarization",
               filename: str = "clip.wav", duration: float = 10.0) -> int:
    with factory() as session:
        owner = session.query(User).filter_by(email=DEV_USER_EMAIL).one()
        audio_file = AudioFile(
            owner_id=owner.id, filename=filename, duration_sec=duration,
            s3_key=f"audio/{filename}-{surface}-{duration}.wav", surface=surface,
        )
        session.add(audio_file)
        session.commit()
        return audio_file.id


def test_empty_surface_is_an_empty_list_not_a_404(client: TestClient) -> None:
    """A surface with no recordings yet is a real state, not an error."""
    response = client.get("/recordings?surface=transcript")
    assert response.status_code == 200
    assert response.json() == []


def test_the_two_surfaces_do_not_leak_into_each_other(
    client: TestClient, db_session_factory: sessionmaker[Session]
) -> None:
    """The whole point of the column."""
    diar = _recording(db_session_factory, surface="diarization", filename="upload.wav")
    live = _recording(db_session_factory, surface="transcript", filename="recording_20260819_120000")

    diar_ids = [r["audioFileId"] for r in client.get("/recordings?surface=diarization").json()]
    live_ids = [r["audioFileId"] for r in client.get("/recordings?surface=transcript").json()]

    assert diar in diar_ids and diar not in live_ids
    assert live in live_ids and live not in diar_ids


def test_default_surface_is_diarization(
    client: TestClient, db_session_factory: sessionmaker[Session]
) -> None:
    diar = _recording(db_session_factory, surface="diarization")
    _recording(db_session_factory, surface="transcript", filename="recording_x")
    assert [r["audioFileId"] for r in client.get("/recordings").json()] == [diar]


def test_newest_first(client: TestClient, db_session_factory: sessionmaker[Session]) -> None:
    """The list is ordered so recent work is on top; ids ascend with time here."""
    first = _recording(db_session_factory, filename="a.wav", duration=1.0)
    second = _recording(db_session_factory, filename="b.wav", duration=2.0)
    third = _recording(db_session_factory, filename="c.wav", duration=3.0)
    ids = [r["audioFileId"] for r in client.get("/recordings").json()]
    assert ids == [third, second, first]


def test_diarization_counts_are_computed_server_side(
    client: TestClient, db_session_factory: sessionmaker[Session]
) -> None:
    """Model, done and failed counts, and the highest speaker count any model found.

    The client used to derive these by fetching each recording's whole evaluation.
    """
    audio_file_id = _recording(db_session_factory)
    with db_session_factory() as session:
        session.add_all([
            EvaluationResult(audio_file_id=audio_file_id, model_id="a", status="done",
                             payload={"numSpk": 3}),
            EvaluationResult(audio_file_id=audio_file_id, model_id="b", status="done",
                             payload={"numSpk": 7}),
            EvaluationResult(audio_file_id=audio_file_id, model_id="c", status="failed"),
            EvaluationResult(audio_file_id=audio_file_id, model_id="d", status="running"),
        ])
        session.commit()

    row = client.get("/recordings").json()[0]
    assert row["modelCount"] == 4
    assert row["doneCount"] == 2
    assert row["failedCount"] == 1
    assert row["speakerCount"] == 7, "the MAX across models, not the last one seen"


def test_a_recording_with_no_results_is_still_listed(
    client: TestClient, db_session_factory: sessionmaker[Session]
) -> None:
    """An abandoned or failed ingest still exists, and hiding it would make the list
    disagree with the database."""
    audio_file_id = _recording(db_session_factory)
    row = client.get("/recordings").json()[0]
    assert row["audioFileId"] == audio_file_id
    assert row["modelCount"] == 0
    assert row["speakerCount"] == 0


def test_transcript_counts_engines_and_reports_best_wer(
    client: TestClient, db_session_factory: sessionmaker[Session]
) -> None:
    audio_file_id = _recording(db_session_factory, surface="transcript", filename="recording_1")
    with db_session_factory() as session:
        session.add_all([
            TranscriptResult(audio_file_id=audio_file_id, asr_id="hamsa", status="done",
                             source="live", text="a", wer=0.25),
            TranscriptResult(audio_file_id=audio_file_id, asr_id="inception-stt", status="done",
                             source="live", text="b", wer=0.40),
            TranscriptReference(audio_file_id=audio_file_id, source="script", text="one two"),
        ])
        session.commit()

    row = client.get("/recordings?surface=transcript").json()[0]
    assert row["engineCount"] == 2
    assert row["scored"] is True
    assert row["bestWer"] == pytest.approx(0.25), "the LOWEST error rate, not the last"


def test_unscored_transcript_recording_reports_no_wer(
    client: TestClient, db_session_factory: sessionmaker[Session]
) -> None:
    """None, not 0.0 — a 0.0 WER is a perfect transcript."""
    audio_file_id = _recording(db_session_factory, surface="transcript", filename="recording_2")
    with db_session_factory() as session:
        session.add(TranscriptResult(audio_file_id=audio_file_id, asr_id="hamsa",
                                    status="done", source="live", text="a"))
        session.commit()

    row = client.get("/recordings?surface=transcript").json()[0]
    assert row["engineCount"] == 1
    assert row["scored"] is False
    assert row["bestWer"] is None


def test_a_reference_alone_does_not_count_as_scored(
    client: TestClient, db_session_factory: sessionmaker[Session]
) -> None:
    """Having ground truth is not the same as having run against it."""
    audio_file_id = _recording(db_session_factory, surface="transcript", filename="recording_3")
    with db_session_factory() as session:
        session.add(TranscriptReference(audio_file_id=audio_file_id, source="pasted", text="x y"))
        session.commit()

    assert client.get("/recordings?surface=transcript").json()[0]["scored"] is False


def test_an_unknown_surface_is_rejected(client: TestClient) -> None:
    assert client.get("/recordings?surface=nope").status_code == 422


def test_deleting_a_recording_removes_it_from_the_list(
    client: TestClient, db_session_factory: sessionmaker[Session]
) -> None:
    audio_file_id = _recording(db_session_factory)
    assert len(client.get("/recordings").json()) == 1
    assert client.delete(f"/evaluations/{audio_file_id}").status_code == 204
    assert client.get("/recordings").json() == []
