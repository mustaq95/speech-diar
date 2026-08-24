"""End-to-end coverage of the TTS comparison routes.

Stubbed at each engine's `runner.run`, NOT at the route: a green suite proved
nothing once before in this repo, when `finalize_session` 500'd on every real
call while every test passed, because nothing drove the route body. These tests
drive the real handler and assert on PERSISTED STATE (rows, stored bytes,
response headers) rather than on status codes alone.

The route deliberately opens its own `SessionLocal()` for the write, because it
must close the request-scoped session before a synthesis that can outlast the
60s `idle_in_transaction_session_timeout` (see the route's docstring). That
bypasses the `get_db` override, so `SessionLocal` is patched here to the test
factory — without it these tests would silently write to the real Postgres.
"""

import wave
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from packages.audio import AudioProbe
from packages.config.settings import get_settings
from packages.database.models import AudioFile, TranscriptReference, TranscriptResult, TtsResult, User
from packages.database.session import DEV_USER_EMAIL

HAMSA = "hamsa-tts"
INCEPTION = "inception-tts"


def _wav(duration_sec: float = 1.0, framerate: int = 16000) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(framerate)
        out.writeframes(b"\x00\x00" * int(duration_sec * framerate))
    return buffer.getvalue()


#: A real MP3 frame sync, enough for the adapter's magic-byte sniff. Kept
#: distinct from the WAV payload so the audio route's Content-Type cannot pass
#: by accident on a hardcoded "audio/wav".
MP3_BYTES = b"\xff\xf3\x84\xc4" + b"\x00" * 512


@pytest.fixture(autouse=True)
def configured_tts(monkeypatch: pytest.MonkeyPatch):
    """Both TTS engines configured, so engine selection is never what fails.

    Patched on the `get_settings()` SINGLETON, not on the Settings class: these
    are pydantic fields living in the instance dict, so a class-level setattr is
    a silent no-op that leaves the real .env showing through -- which would make
    these tests pass on a configured host and fail on a bare one.
    """
    settings = get_settings()
    for field, value in (
        ("hamsa_tts_api_url", "https://hamsa.test/tts/stream"),
        ("hamsa_tts_key", "k"),
        ("hamsa_tts_bearer_token", "b"),
        ("litellm_base_url", "https://gateway.test"),
        ("litellm_api_key", "k"),
        ("tts_max_input_chars", 4000),
        ("inception_tts_response_format", "wav"),
    ):
        monkeypatch.setattr(settings, field, value)


@pytest.fixture()
def store(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """An in-memory stand-in for MinIO, so the routes exercise their real
    storage calls (key layout, Range slicing) without an object store."""
    objects: dict[str, bytes] = {}

    def put_stream(fileobj, key):
        objects[key] = fileobj.read()
        return key

    def open_stream(key, start=0):
        return BytesIO(objects[key][start:])

    monkeypatch.setattr("apps.backend_api.routers.tts.s3_client.put_stream", put_stream)
    monkeypatch.setattr("apps.backend_api.routers.tts.s3_client.head_object", lambda key: len(objects[key]))
    monkeypatch.setattr("apps.backend_api.routers.evaluations.s3_client.open_stream", open_stream)
    monkeypatch.setattr(
        "apps.backend_api.routers.evaluations.s3_client.delete_object", lambda key: objects.pop(key, None)
    )
    return objects


@pytest.fixture(autouse=True)
def write_session(monkeypatch: pytest.MonkeyPatch, db_session_factory):
    """Point the route's own write session at the test database. See module docstring."""
    monkeypatch.setattr("apps.backend_api.routers.tts.SessionLocal", db_session_factory)


@pytest.fixture()
def script_row(db_session: Session) -> AudioFile:
    """What `POST /transcript/script` leaves behind: an audio-less recording row
    plus its reference. TTS hangs off exactly this, with no surface of its own."""
    owner = db_session.query(User).filter_by(email=DEV_USER_EMAIL).one()
    audio_file = AudioFile(owner_id=owner.id, filename="script_test", duration_sec=0.0, surface="transcript")
    db_session.add(audio_file)
    db_session.flush()
    db_session.add(
        TranscriptReference(audio_file_id=audio_file.id, source="script", text="مرحبا بالعالم", params={})
    )
    db_session.commit()
    return audio_file


def _stub_engine(monkeypatch: pytest.MonkeyPatch, tts_id: str, *, audio: bytes, first_ms: int = 100, synth_ms: int = 500):
    """Replace one engine's runner at its HTTP boundary, leaving its adapter,
    the registry lookup and the whole route body running for real."""
    module = "hamsa" if tts_id == HAMSA else "inception"

    def fake_run(text: str, voice: str) -> dict:
        return {
            "audio": audio,
            "status_code": 200,
            "headers": {"content-type": "audio/mpeg"},
            "synth_ms": synth_ms,
            "first_audio_ms": first_ms,
            "voice": voice,
        }

    monkeypatch.setattr(f"apps.background_worker.tts.{module}.runner.run", fake_run)
    # The registry captured the original function reference at import time.
    monkeypatch.setitem(
        __import__("apps.background_worker.tts", fromlist=["TTS_ENGINES"]).TTS_ENGINES,
        tts_id,
        _replace_run(tts_id, fake_run),
    )


def _replace_run(tts_id: str, fake_run):
    from dataclasses import replace

    from apps.background_worker.tts import TTS_ENGINES

    return replace(TTS_ENGINES[tts_id], run=fake_run)


@pytest.fixture()
def both_engines(monkeypatch: pytest.MonkeyPatch):
    """Hamsa renders WAV (its adapter wraps raw PCM); Inception is stubbed to
    return MP3 so the two engines' stored formats genuinely differ."""
    # Hamsa's adapter wraps headerless PCM itself, so feed it raw frames.
    _stub_engine(monkeypatch, HAMSA, audio=b"\x00\x00" * 16000, first_ms=100, synth_ms=500)
    # Inception's adapter cross-checks the sniffed container against the
    # REQUESTED format, so the host has to be asking for mp3 for this to be a
    # valid response rather than the mismatch error it is designed to catch.
    monkeypatch.setattr(get_settings(), "inception_tts_response_format", "mp3")
    _stub_engine(monkeypatch, INCEPTION, audio=MP3_BYTES, first_ms=480, synth_ms=490)


def test_both_engines_persist_one_row_each(
    client: TestClient, db_session: Session, script_row: AudioFile, store, both_engines
) -> None:
    """The headline path: two engines, two rows, real stored bytes."""
    for tts_id in (HAMSA, INCEPTION):
        response = client.post(f"/evaluations/{script_row.id}/tts/{tts_id}", json={})
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "done", response.text

    rows = db_session.query(TtsResult).filter_by(audio_file_id=script_row.id).all()
    assert {row.tts_id for row in rows} == {HAMSA, INCEPTION}
    for row in rows:
        assert row.s3_key, f"{row.tts_id} stored no audio"
        assert row.s3_key in store, f"{row.tts_id}'s key is not in the object store"
        assert row.synth_ms and row.synth_ms > 0
        assert row.text_chars == len("مرحبا بالعالم")


def test_synthesis_writes_no_second_reference(
    client: TestClient, db_session: Session, script_row: AudioFile, store, both_engines
) -> None:
    """`audio_file_id` is unique on transcript_references. A TTS run must never
    add one -- the same blind-`db.add`-against-a-unique-index shape that once
    500'd every finalize."""
    for tts_id in (HAMSA, INCEPTION):
        client.post(f"/evaluations/{script_row.id}/tts/{tts_id}", json={})

    assert db_session.query(TranscriptReference).filter_by(audio_file_id=script_row.id).count() == 1


def test_resynthesize_resets_the_row_instead_of_adding_one(
    client: TestClient, db_session: Session, script_row: AudioFile, store, monkeypatch, both_engines
) -> None:
    """Re-running an engine is the most natural second action in this UI. A blind
    `db.add` would trip uq_tts_results_audio_file_tts and 500 on the second call."""
    first = client.post(f"/evaluations/{script_row.id}/tts/{HAMSA}", json={})
    assert first.status_code == 200

    # A different measurement the second time round, so a stale row is visible.
    _stub_engine(monkeypatch, HAMSA, audio=b"\x00\x00" * 32000, first_ms=222, synth_ms=999)
    second = client.post(f"/evaluations/{script_row.id}/tts/{HAMSA}", json={})
    assert second.status_code == 200, second.text

    rows = db_session.query(TtsResult).filter_by(audio_file_id=script_row.id, tts_id=HAMSA).all()
    assert len(rows) == 1, "re-synthesizing inserted a second row instead of resetting"
    assert rows[0].synth_ms == 999, "the row kept the previous run's timings beside fresh audio"


def test_one_engine_failing_leaves_the_other_intact(
    client: TestClient, db_session: Session, script_row: AudioFile, store, monkeypatch, both_engines
) -> None:
    """One engine per request exists precisely so a failure cannot be contagious.
    The failed engine still gets a row: without one, "never tried" and "failed"
    look identical to the operator."""
    ok = client.post(f"/evaluations/{script_row.id}/tts/{INCEPTION}", json={})
    assert ok.json()["status"] == "done"

    def boom(text: str, voice: str) -> dict:
        raise RuntimeError("hamsa exploded")

    monkeypatch.setitem(
        __import__("apps.background_worker.tts", fromlist=["TTS_ENGINES"]).TTS_ENGINES,
        HAMSA,
        _replace_run(HAMSA, boom),
    )
    failed = client.post(f"/evaluations/{script_row.id}/tts/{HAMSA}", json={})
    # 200, not 500: a failed ENGINE is not a failed request.
    assert failed.status_code == 200, failed.text
    assert failed.json()["status"] == "failed"
    assert "hamsa exploded" in failed.json()["error"]

    rows = {row.tts_id: row for row in db_session.query(TtsResult).filter_by(audio_file_id=script_row.id)}
    assert rows[HAMSA].status == "failed" and rows[HAMSA].s3_key is None
    assert rows[INCEPTION].status == "done", "the healthy engine's row was disturbed"
    assert rows[INCEPTION].s3_key in store, "the healthy engine's audio was dropped"


def test_delete_recording_removes_every_tts_row(
    client: TestClient, db_session: Session, script_row: AudioFile, store, both_engines
) -> None:
    """SQLite does not enforce the FK, so a missing delete would orphan silently
    here and only blow up on Postgres. The row COUNT is the assertion."""
    # Held as a plain int: the row itself is about to be deleted, so reading
    # script_row.id after expire_all() would try to reload a gone row.
    audio_file_id = script_row.id
    for tts_id in (HAMSA, INCEPTION):
        client.post(f"/evaluations/{audio_file_id}/tts/{tts_id}", json={})
    assert db_session.query(TtsResult).filter_by(audio_file_id=audio_file_id).count() == 2

    assert client.delete(f"/evaluations/{audio_file_id}").status_code == 204

    db_session.expire_all()
    assert db_session.query(TtsResult).filter_by(audio_file_id=audio_file_id).count() == 0


def test_audio_route_labels_each_container_correctly(
    client: TestClient, script_row: AudioFile, store, both_engines
) -> None:
    """`stream_audio` hardcodes audio/wav; copying that here would serve an mp3
    mislabelled. The two engines return different containers on purpose, so a
    hardcoded value cannot pass this."""
    for tts_id in (HAMSA, INCEPTION):
        client.post(f"/evaluations/{script_row.id}/tts/{tts_id}", json={})

    wav = client.get(f"/evaluations/{script_row.id}/tts/{HAMSA}/audio")
    assert wav.status_code == 200
    assert wav.headers["content-type"] == "audio/wav"
    assert wav.content[:4] == b"RIFF", "served bytes are not the WAV that was stored"

    mp3 = client.get(f"/evaluations/{script_row.id}/tts/{INCEPTION}/audio")
    assert mp3.status_code == 200
    assert mp3.headers["content-type"] == "audio/mpeg"
    assert mp3.content == MP3_BYTES


def test_audio_route_honours_range_and_download(
    client: TestClient, script_row: AudioFile, store, both_engines
) -> None:
    client.post(f"/evaluations/{script_row.id}/tts/{HAMSA}", json={"voice": "Ruba"})
    total = len(store[f"tts/{script_row.id}/{HAMSA}/ruba.wav"])

    partial = client.get(
        f"/evaluations/{script_row.id}/tts/{HAMSA}/audio", headers={"Range": "bytes=0-99"}
    )
    assert partial.status_code == 206
    assert partial.headers["content-range"] == f"bytes 0-99/{total}"
    assert len(partial.content) == 100

    download = client.get(f"/evaluations/{script_row.id}/tts/{HAMSA}/audio?download=1")
    assert "attachment" in download.headers["content-disposition"]


def test_unmeasurable_duration_serializes_null_not_zero(
    client: TestClient, script_row: AudioFile, store, monkeypatch, both_engines
) -> None:
    """A figure that could not be measured renders as absent. A 0.0 here would
    read as "instant", which is a fabricated measurement."""
    monkeypatch.setattr("apps.backend_api.routers.tts.probe_audio", lambda payload: AudioProbe())

    body = client.post(f"/evaluations/{script_row.id}/tts/{HAMSA}", json={}).json()
    assert body["status"] == "done"
    assert body["audioSec"] is None
    assert body["rtf"] is None
    assert body["nativeSampleRate"] is None


def test_unknown_engine_is_404(client: TestClient, script_row: AudioFile) -> None:
    assert client.post(f"/evaluations/{script_row.id}/tts/not-an-engine", json={}).status_code == 404


def test_unconfigured_engine_is_422(
    client: TestClient, script_row: AudioFile, monkeypatch
) -> None:
    """An unconfigured host says so, rather than failing every run at the gateway."""
    monkeypatch.setattr(get_settings(), "hamsa_tts_api_url", None)
    response = client.post(f"/evaluations/{script_row.id}/tts/{HAMSA}", json={})
    assert response.status_code == 422
    assert "not configured" in response.json()["detail"]


def test_synthesis_without_a_script_is_422(
    client: TestClient, db_session: Session, both_engines
) -> None:
    """A recording with no reference has nothing to synthesize."""
    owner = db_session.query(User).filter_by(email=DEV_USER_EMAIL).one()
    bare = AudioFile(owner_id=owner.id, filename="bare", duration_sec=0.0, surface="transcript")
    db_session.add(bare)
    db_session.commit()

    response = client.post(f"/evaluations/{bare.id}/tts/{HAMSA}", json={})
    assert response.status_code == 422
    assert "script" in response.json()["detail"].lower()


def test_text_over_the_host_limit_is_422(
    client: TestClient, script_row: AudioFile, monkeypatch, both_engines
) -> None:
    """Fails at the request with the real limit, instead of at a gateway timeout
    minutes later."""
    monkeypatch.setattr(get_settings(), "tts_max_input_chars", 5)
    response = client.post(f"/evaluations/{script_row.id}/tts/{HAMSA}", json={})
    assert response.status_code == 422
    assert "5" in response.json()["detail"]


def test_editing_the_reference_clears_stale_clips(
    client: TestClient, db_session: Session, script_row: AudioFile, store, both_engines
) -> None:
    """Clips synthesized from the OLD words must not sit beside a rewritten
    reference with nothing saying so -- the same reasoning `put_reference`
    already applies to stale transcript scores."""
    for tts_id in (HAMSA, INCEPTION):
        client.post(f"/evaluations/{script_row.id}/tts/{tts_id}", json={})
    assert db_session.query(TtsResult).filter_by(audio_file_id=script_row.id).count() == 2

    updated = client.put(
        f"/evaluations/{script_row.id}/reference", json={"text": "نص مختلف تماما", "source": "pasted"}
    )
    assert updated.status_code == 200, updated.text

    db_session.expire_all()
    assert db_session.query(TtsResult).filter_by(audio_file_id=script_row.id).count() == 0


def test_tts_count_surfaces_on_the_recordings_list(
    client: TestClient, script_row: AudioFile, store, both_engines
) -> None:
    """Projects tells a synthesized script from a plain one by what the row HAS,
    which is why there is no third surface."""
    for tts_id in (HAMSA, INCEPTION):
        client.post(f"/evaluations/{script_row.id}/tts/{tts_id}", json={})

    rows = client.get("/recordings?surface=transcript").json()
    row = next(entry for entry in rows if entry["audioFileId"] == script_row.id)
    assert row["ttsCount"] == 2
    assert row["hasAudio"] is False, "a script row still has no recording of its own"


def test_raw_route_returns_metadata_not_audio(
    client: TestClient, script_row: AudioFile, store, both_engines
) -> None:
    client.post(f"/evaluations/{script_row.id}/tts/{HAMSA}", json={})
    body = client.get(f"/evaluations/{script_row.id}/tts/{HAMSA}/raw").json()
    assert body["ttsId"] == HAMSA
    assert body["rawOutput"]["status_code"] == 200
    assert "audio" not in body["rawOutput"], "the audio bytes leaked into the JSON column"


def test_transcript_results_are_untouched_by_synthesis(
    client: TestClient, db_session: Session, script_row: AudioFile, store, both_engines
) -> None:
    """The two surfaces share a recording row; they must not share result rows."""
    client.post(f"/evaluations/{script_row.id}/tts/{HAMSA}", json={})
    assert db_session.query(TranscriptResult).filter_by(audio_file_id=script_row.id).count() == 0


def test_the_row_records_which_voice_actually_produced_the_clip(
    client: TestClient, db_session: Session, script_row: AudioFile, store, monkeypatch, both_engines
) -> None:
    """Each VOICE keeps its own stored clip.

    One row per (recording, engine, voice), so synthesizing a second voice adds
    a take rather than replacing the first -- which is what lets the UI show
    each voice's own clip instead of relabelling one clip as another's.
    """
    first = client.post(f"/evaluations/{script_row.id}/tts/{HAMSA}", json={"voice": "Ruba"})
    assert first.json()["voice"] == "Ruba"

    second = client.post(f"/evaluations/{script_row.id}/tts/{HAMSA}", json={"voice": "Sandra"})
    assert second.json()["voice"] == "Sandra"

    rows = db_session.query(TtsResult).filter_by(audio_file_id=script_row.id, tts_id=HAMSA).all()
    assert len(rows) == 2, "a second VOICE must add a row, not overwrite the first"
    by_voice = {row.voice: row for row in rows}
    assert set(by_voice) == {"Ruba", "Sandra"}
    # Distinct objects, or the second synthesis silently overwrote the first's
    # bytes while both rows still pointed at them.
    assert by_voice["Ruba"].s3_key != by_voice["Sandra"].s3_key
    assert by_voice["Ruba"].s3_key in store and by_voice["Sandra"].s3_key in store

    # And the reload path the UI actually uses returns both.
    listed = client.get(f"/evaluations/{script_row.id}/tts").json()
    assert sorted(r["voice"] for r in listed if r["ttsId"] == HAMSA) == ["Ruba", "Sandra"]


def test_omitting_the_voice_uses_the_engine_default_not_the_raw_setting(
    client: TestClient, script_row: AudioFile, store, monkeypatch, both_engines
) -> None:
    """With a multi-voice .env the default is entry 0, never the whole list."""
    monkeypatch.setattr(get_settings(), "hamsa_tts_speaker", "Ruba,Sandra")
    body = client.post(f"/evaluations/{script_row.id}/tts/{HAMSA}", json={}).json()
    assert body["voice"] == "Ruba"
    assert "," not in body["voice"]


def test_resynthesizing_the_same_voice_replaces_only_that_voice(
    client: TestClient, db_session: Session, script_row: AudioFile, store, monkeypatch, both_engines
) -> None:
    """Per-voice storage must not turn re-running into row accumulation.

    The unique key is (recording, engine, voice), so the SAME voice still
    resets in place while a different voice adds. Getting this wrong either way
    is silent: too few rows loses a take, too many trips the constraint.
    """
    client.post(f"/evaluations/{script_row.id}/tts/{HAMSA}", json={"voice": "Ruba"})
    client.post(f"/evaluations/{script_row.id}/tts/{HAMSA}", json={"voice": "Sandra"})

    _stub_engine(monkeypatch, HAMSA, audio=b"\x00\x00" * 32000, first_ms=222, synth_ms=999)
    again = client.post(f"/evaluations/{script_row.id}/tts/{HAMSA}", json={"voice": "Ruba"})
    assert again.status_code == 200, again.text

    rows = db_session.query(TtsResult).filter_by(audio_file_id=script_row.id, tts_id=HAMSA).all()
    assert len(rows) == 2, "re-running an existing voice added a row instead of resetting it"
    by_voice = {row.voice: row for row in rows}
    db_session.refresh(by_voice["Ruba"])
    db_session.refresh(by_voice["Sandra"])
    assert by_voice["Ruba"].synth_ms == 999, "Ruba kept the previous take's timings"
    assert by_voice["Sandra"].synth_ms == 500, "re-running Ruba disturbed Sandra's row"


def test_audio_route_reports_ambiguity_instead_of_guessing(
    client: TestClient, script_row: AudioFile, store, both_engines
) -> None:
    """With one clip, an omitted voice is unambiguous and still served -- which
    is what keeps pre-existing callers (the saved Postman requests) working.
    With several, handing back an arbitrary take would be a wrong answer that
    looks like a right one, so the route names the voices instead.
    """
    client.post(f"/evaluations/{script_row.id}/tts/{HAMSA}", json={"voice": "Ruba"})
    assert client.get(f"/evaluations/{script_row.id}/tts/{HAMSA}/audio").status_code == 200

    client.post(f"/evaluations/{script_row.id}/tts/{HAMSA}", json={"voice": "Sandra"})
    ambiguous = client.get(f"/evaluations/{script_row.id}/tts/{HAMSA}/audio")
    assert ambiguous.status_code == 400
    detail = ambiguous.json()["detail"]
    assert "Ruba" in detail and "Sandra" in detail, detail

    # Naming the voice resolves it, and the two clips are genuinely different.
    ruba = client.get(f"/evaluations/{script_row.id}/tts/{HAMSA}/audio?voice=Ruba")
    sandra = client.get(f"/evaluations/{script_row.id}/tts/{HAMSA}/audio?voice=Sandra")
    assert ruba.status_code == 200 and sandra.status_code == 200
    assert ruba.content == sandra.content, "both stubs render the same bytes here"
    assert "Ruba" in ruba.headers.get("content-disposition", "Ruba")


def test_unknown_voice_is_404_naming_what_exists(
    client: TestClient, script_row: AudioFile, store, both_engines
) -> None:
    client.post(f"/evaluations/{script_row.id}/tts/{HAMSA}", json={"voice": "Ruba"})
    missing = client.get(f"/evaluations/{script_row.id}/tts/{HAMSA}/audio?voice=Nobody")
    assert missing.status_code == 404
    assert "Ruba" in missing.json()["detail"]
