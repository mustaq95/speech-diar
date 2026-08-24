"""Script generation as a SAVE, not a stateless call.

`POST /transcript/script` used to persist nothing: its own docstring said "there is
no script table and an abandoned script costs nothing". That made a generated script
unrecoverable the moment you regenerated or left the page, and nothing appeared in
Projects until Stop.

It now writes the recording row and its reference as the script is generated, and
`finalize` attaches the audio to that same row. Two failure modes are worth guarding
because both are invisible from a 200:

  * a second row at finalize, giving one script two Projects entries;
  * a second reference row for the same recording, which trips the unique index on
    `audio_file_id` and 500s (the exact bug class that once 500'd every finalize).
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from packages import script_gen
from packages.database.models import AudioFile, TranscriptReference, TranscriptResult

SCRIPT_TEXT = "one two three four five six seven eight nine ten"


@pytest.fixture(autouse=True)
def stub_generator(monkeypatch: pytest.MonkeyPatch):
    """A generator that returns a fixed script without touching an LLM gateway.

    Patched at the router's import site, like the rest of the suite: the route's own
    persistence is what is under test, not the gateway call.
    """
    calls: list[dict] = []

    def generate(*, minutes, language_mix, hard_cases, settings):
        calls.append({"minutes": minutes, "language_mix": language_mix, "hard_cases": list(hard_cases)})
        return script_gen.GeneratedScript(
            text=SCRIPT_TEXT,
            word_count=len(SCRIPT_TEXT.split()),
            generator_model="stub-model",
            params={
                "minutes": minutes,
                "languageMix": language_mix,
                "hardCases": list(hard_cases),
                "generatorModel": "stub-model",
            },
        )

    monkeypatch.setattr("apps.backend_api.routers.transcript.script_gen.generate", generate)
    return calls


@pytest.fixture(autouse=True)
def script_options(monkeypatch: pytest.MonkeyPatch):
    """Fix the host's offered options so validation is never what fails."""
    monkeypatch.setattr(
        "packages.config.settings.Settings.script_language_mixes", "ar,mixed-50-50,en", raising=False
    )
    monkeypatch.setattr(
        "packages.config.settings.Settings.script_hard_cases",
        "proper-nouns,numbers-dates,emirati-dialect,fast-speech",
        raising=False,
    )


def _generate(client: TestClient, *, minutes: float = 1, mix: str = "en", cases=("proper-nouns",)):
    return client.post(
        "/transcript/script",
        json={"minutes": minutes, "languageMix": mix, "hardCases": list(cases)},
    )


def test_generating_a_script_creates_its_recording_row(
    client: TestClient, db_session_factory: sessionmaker[Session]
) -> None:
    response = _generate(client)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["audioFileId"] > 0
    with db_session_factory() as session:
        row = session.get(AudioFile, body["audioFileId"])
        assert row is not None
        assert row.surface == "transcript"
        # No audio yet, and the name says so.
        assert row.s3_key is None
        assert row.blob_key is None
        assert row.filename.startswith("recording_")


def test_generating_a_script_saves_the_script_itself(
    client: TestClient, db_session_factory: sessionmaker[Session]
) -> None:
    """The point of the change: the text survives without recording anything."""
    audio_file_id = _generate(client).json()["audioFileId"]

    with db_session_factory() as session:
        refs = session.query(TranscriptReference).filter_by(audio_file_id=audio_file_id).all()
    assert len(refs) == 1
    assert refs[0].text == SCRIPT_TEXT
    assert refs[0].source == "script"
    # The params travel with it, so a score can be traced back to what was asked for.
    assert refs[0].params["languageMix"] == "en"
    assert refs[0].params["hardCases"] == ["proper-nouns"]


def test_each_generation_gets_its_own_entry(
    client: TestClient, db_session_factory: sessionmaker[Session]
) -> None:
    """One entry per generation, so no generated script is ever lost."""
    ids = [_generate(client, mix=mix).json()["audioFileId"] for mix in ("en", "ar", "mixed-50-50")]

    assert len(set(ids)) == 3
    with db_session_factory() as session:
        rows = session.query(AudioFile).filter_by(surface="transcript").all()
        refs = session.query(TranscriptReference).all()
    assert len(rows) == 3
    assert len(refs) == 3


def test_a_saved_script_is_listed_as_not_yet_recorded(client: TestClient) -> None:
    """It has to reach Projects, which is the whole request. `hasAudio` is what tells
    the list not to print `0:00` as though a duration had been measured."""
    audio_file_id = _generate(client).json()["audioFileId"]

    listing = client.get("/recordings", params={"surface": "transcript"})

    assert listing.status_code == 200, listing.text
    rows = {row["audioFileId"]: row for row in listing.json()}
    assert audio_file_id in rows
    row = rows[audio_file_id]
    assert row["hasAudio"] is False
    assert row["durationSec"] == 0
    assert row["engineCount"] == 0
    assert row["scored"] is False


def test_a_saved_script_has_no_audio_to_stream(client: TestClient) -> None:
    """A script-only row must 404 rather than 500 on the audio route, because the
    player is one click away in Projects."""
    audio_file_id = _generate(client).json()["audioFileId"]

    response = client.get(f"/evaluations/{audio_file_id}/audio")

    assert response.status_code == 404


def test_a_saved_script_can_be_deleted(
    client: TestClient, db_session_factory: sessionmaker[Session]
) -> None:
    """Deleting a script-only row must not trip over its null `s3_key`; discarded
    drafts are the cost of keeping every generation, so removing them has to work."""
    audio_file_id = _generate(client).json()["audioFileId"]

    response = client.delete(f"/evaluations/{audio_file_id}")

    assert response.status_code in (200, 204), response.text
    with db_session_factory() as session:
        assert session.get(AudioFile, audio_file_id) is None
        assert session.query(TranscriptReference).filter_by(audio_file_id=audio_file_id).count() == 0
        assert session.query(TranscriptResult).filter_by(audio_file_id=audio_file_id).count() == 0


# --- bringing your own script -------------------------------------------------
#
# TTS synthesis reads its text from a STORED reference, so a script the operator
# typed has to become a real row before either engine can see it. Until this
# route existed the only way to get such a row was to generate one from the LLM.


def test_create_reference_makes_a_synthesizable_row(client, db_session) -> None:
    from packages.database.models import AudioFile, TranscriptReference

    response = client.post("/transcript/reference", json={"text": "  مرحبا بالعالم  "})
    assert response.status_code == 201, response.text
    body = response.json()

    assert body["source"] == "pasted", "a typed script must not claim a generator"
    assert body["text"] == "مرحبا بالعالم", "surrounding whitespace should be stripped"
    assert body["wordCount"] == 2

    # The same row shape generate_script writes: listable, and audio-less until
    # someone actually reads it aloud.
    audio_file = db_session.get(AudioFile, body["audioFileId"])
    assert audio_file is not None
    assert audio_file.surface == "transcript"
    assert audio_file.s3_key is None
    assert audio_file.filename.startswith("recording_")

    references = (
        db_session.query(TranscriptReference).filter_by(audio_file_id=audio_file.id).all()
    )
    assert len(references) == 1
    assert references[0].source == "pasted"


def test_create_reference_rejects_empty_text(client) -> None:
    """Whitespace is not a script; failing here beats a row nothing can synthesize."""
    for text in ("", "   ", "\n\t"):
        assert client.post("/transcript/reference", json={"text": text}).status_code == 422


def test_created_reference_appears_on_the_transcript_surface(client) -> None:
    """It has to reach Projects, or a pasted script is invisible until synthesis."""
    created = client.post("/transcript/reference", json={"text": "نص للاختبار"}).json()
    rows = client.get("/recordings?surface=transcript").json()
    row = next(r for r in rows if r["audioFileId"] == created["audioFileId"])
    assert row["hasAudio"] is False
    assert row["ttsCount"] == 0
