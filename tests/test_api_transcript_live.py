"""The live read-aloud endpoints: session, chunk, finalize.

These had no coverage, and it showed. A refactor left `db.add(TranscriptReference)`
in both the finalize endpoint and the helper it delegates to; SQLAlchemy batched
the two into one multi-VALUES INSERT, which tripped the unique index on
audio_file_id and returned 500 for every finalize. The whole suite stayed green
because nothing exercised the route.

So the assertions here are deliberately about persistence, not just status codes:
that ONE reference row is written, that one row per engine appears, and that the
measured timings and scores survive the trip. A 200 alone would not have caught it.
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from packages.database.models import AudioFile, TranscriptReference, TranscriptResult, User
from packages.database.session import DEV_USER_EMAIL
from tests.conftest import make_wav_bytes


@pytest.fixture()
def stub_s3(monkeypatch: pytest.MonkeyPatch):
    """Keep the recording out of a real object store, as the other API tests do."""
    calls: list[str] = []
    monkeypatch.setattr(
        "apps.backend_api.routers.upload.s3_client.put_stream",
        lambda fileobj, key: calls.append(key) or key,
    )
    return calls


@pytest.fixture(autouse=True)
def configured_engines(monkeypatch: pytest.MonkeyPatch):
    """Both compared engines configured, so engine selection is never what fails."""
    monkeypatch.setattr(
        "apps.background_worker.transcription.cohere_container_healthy", lambda s: True
    )
    for field, value in (
        ("hamsa_stt_ws_url", "wss://hamsa.test/ws"),
        ("hamsa_stt_key", "k"),
        ("litellm_base_url", "https://gateway.test"),
        ("litellm_api_key", "k"),
    ):
        monkeypatch.setattr(
            "packages.config.settings.Settings." + field, value, raising=False
        )


@pytest.fixture()
def live_session(monkeypatch):
    """An in-memory stand-in for the Redis-backed session store.

    Patched at the router's import site so the endpoints exercise their real
    logic without needing a Redis server, matching how the rest of the suite
    avoids live infra.
    """
    store: dict = {}

    def create(asr_ids, chunk_interval_sec, reference_text=""):
        store["meta"] = {
            "asr_ids": list(asr_ids),
            "chunk_interval_sec": chunk_interval_sec,
            "reference_text": reference_text,
            "started_at": 0.0,
        }
        store["parts"] = {asr_id: [] for asr_id in asr_ids}
        return "test-session"

    def meta(session_id):
        return store.get("meta") if session_id == "test-session" else None

    def transcript(session_id, asr_id):
        parts = store.get("parts", {}).get(asr_id, [])
        latencies = [p["ms"] for p in parts]
        raw = [p.get("raw") for p in parts]
        return {
            "text": " ".join(p["text"] for p in parts if p["text"]).strip(),
            "raw": raw if any(r is not None for r in raw) else None,
            "chunk_count": len(parts),
            "chunk_latencies_ms": latencies,
            "first_latency_ms": latencies[0] if latencies else None,
            "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else None,
            "total_latency_ms": sum(latencies) if latencies else None,
        }

    fake = SimpleNamespace(
        create=create,
        meta=meta,
        transcript=transcript,
        exists=lambda sid: sid == "test-session",
        append_part=lambda sid, asr_id, *, index, text, latency_ms, raw=None: store["parts"]
        .setdefault(asr_id, [])
        .append({"i": index, "text": text, "ms": latency_ms, "raw": raw}),
        close=lambda sid: store.clear(),
        set_reference=lambda sid, text: None,
    )
    monkeypatch.setattr("apps.backend_api.routers.transcript.live_session", fake)
    return SimpleNamespace(store=store, create=create)


def _owner_id(factory: sessionmaker[Session]) -> int:
    with factory() as session:
        return session.query(User).filter_by(email=DEV_USER_EMAIL).one().id


REFERENCE = "one two three four five six seven eight"


def _finalize(client: TestClient, *, reference: str = REFERENCE, source: str = "script"):
    return client.post(
        "/transcript/session/test-session/finalize",
        data={"referenceText": reference, "referenceSource": source},
        files={"file": ("read-aloud.wav", make_wav_bytes(duration_sec=2.0), "audio/wav")},
    )


def test_finalize_writes_exactly_one_reference_row(
    client: TestClient, db_session_factory: sessionmaker[Session], live_session, stub_s3
) -> None:
    """The regression that 500'd every finalize: two `db.add` calls for the same
    recording, batched into one INSERT, violating the unique index."""
    live_session.create(["hamsa", "inception-stt"], 3.0)

    response = _finalize(client)

    assert response.status_code == 200, response.text
    audio_file_id = response.json()[0]["audioFileId"]
    with db_session_factory() as session:
        rows = session.query(TranscriptReference).filter_by(audio_file_id=audio_file_id).all()
    assert len(rows) == 1
    assert rows[0].source == "script"
    assert rows[0].text == REFERENCE


def test_finalize_writes_one_row_per_engine_marked_live(
    client: TestClient, db_session_factory: sessionmaker[Session], live_session, stub_s3
) -> None:
    live_session.create(["hamsa", "inception-stt"], 3.0)
    live_session.store["parts"]["hamsa"] = [{"i": 0, "text": "one two", "ms": 120}]
    live_session.store["parts"]["inception-stt"] = [
        {"i": 0, "text": "one", "ms": 300},
        {"i": 1, "text": "two three", "ms": 500},
    ]

    runs = _finalize(client).json()

    by_id = {run["asrId"]: run for run in runs}
    assert set(by_id) == {"hamsa", "inception-stt"}
    assert all(run["source"] == "live" for run in runs)
    # Status is done, not queued: the transcript already exists and no job is coming.
    assert all(run["status"] == "done" for run in runs)
    assert by_id["hamsa"]["transport"] == "stream"
    assert by_id["inception-stt"]["transport"] == "chunks"


def test_finalize_keeps_the_measured_timings(
    client: TestClient, db_session_factory: sessionmaker[Session], live_session, stub_s3
) -> None:
    """The latencies are the product; they have to survive persistence exactly."""
    live_session.create(["inception-stt"], 3.0)
    live_session.store["parts"]["inception-stt"] = [
        {"i": 0, "text": "one two", "ms": 100},
        {"i": 1, "text": "three", "ms": 300},
    ]

    run = _finalize(client).json()[0]

    assert run["chunkCount"] == 2
    assert run["firstLatencyMs"] == 100
    assert run["avgLatencyMs"] == 200
    assert run["chunkIntervalSec"] == 3.0


def test_finalize_scores_against_the_reference(
    client: TestClient, db_session_factory: sessionmaker[Session], live_session, stub_s3
) -> None:
    live_session.create(["inception-stt"], 3.0)
    # 4 of 8 reference words captured -> 4 deletions -> WER 0.5.
    live_session.store["parts"]["inception-stt"] = [
        {"i": 0, "text": "one two three four", "ms": 100}
    ]

    run = _finalize(client).json()[0]

    assert run["metrics"]["wer"] == pytest.approx(0.5)
    assert run["metrics"]["refWordCount"] == 8
    assert run["metrics"]["hypWordCount"] == 4
    assert run["metrics"]["delCount"] == 4


def test_finalize_without_a_reference_persists_but_does_not_score(
    client: TestClient, db_session_factory: sessionmaker[Session], live_session, stub_s3
) -> None:
    """Timings without error rates is a real state — not a reason to fail."""
    live_session.create(["inception-stt"], 3.0)
    live_session.store["parts"]["inception-stt"] = [{"i": 0, "text": "one", "ms": 100}]

    run = _finalize(client, reference="").json()[0]

    assert run["metrics"] is None
    assert run["chunkCount"] == 1
    with db_session_factory() as session:
        assert session.query(TranscriptReference).count() == 0


def test_finalize_streaming_engine_reports_no_rtf(
    client: TestClient, db_session_factory: sessionmaker[Session], live_session, stub_s3
) -> None:
    """A real-time transport consumes audio at 1x by definition, so there is no
    factor to report. None, never a number."""
    live_session.create(["hamsa", "inception-stt"], 3.0)
    for asr_id in ("hamsa", "inception-stt"):
        live_session.store["parts"][asr_id] = [{"i": 0, "text": "one two", "ms": 200}]

    by_id = {run["asrId"]: run for run in _finalize(client).json()}

    assert by_id["hamsa"]["metrics"]["rtf"] is None
    assert by_id["inception-stt"]["metrics"]["rtf"] is not None


def test_finalize_records_the_recording_as_an_audio_file(
    client: TestClient, db_session_factory: sessionmaker[Session], live_session, stub_s3
) -> None:
    """It goes through the normal ingest, so it shows up in Projects with a real
    duration read from its header — and with NO diarization models queued, since
    one person reading a script makes a speaker comparison a foregone conclusion."""
    live_session.create(["inception-stt"], 3.0)

    audio_file_id = _finalize(client).json()[0]["audioFileId"]

    with db_session_factory() as session:
        audio_file = session.get(AudioFile, audio_file_id)
        assert audio_file is not None
        assert audio_file.duration_sec == pytest.approx(2.0, abs=0.05)
        assert audio_file.s3_key
        from packages.database.models import EvaluationResult
        assert session.query(EvaluationResult).filter_by(audio_file_id=audio_file_id).count() == 0


def test_finalize_persists_each_engines_native_output(
    client: TestClient, db_session_factory: sessionmaker[Session], live_session, stub_s3
) -> None:
    """`raw_output` must survive a LIVE run, not just a stored-audio one.

    This is the gap these tests previously missed entirely: finalize wrote text,
    timings and scores but dropped every engine's native response, so a live
    recording could never be re-examined the way a batch one can. The column is
    deferred, so it has to be loaded explicitly to be checked — which is also why
    a route-level assertion would not have caught it.
    """
    live_session.create(["inception-stt"], 3.0)
    live_session.store["parts"]["inception-stt"] = [
        {"i": 0, "text": "one two", "ms": 100,
         "raw": {"response": {"text": "one two", "audio_duration": 3.0,
                              "usage": {"tokens": 7}}, "latency_ms": 100}},
        {"i": 1, "text": "", "ms": 90,
         "raw": {"response": {"text": ""}, "latency_ms": 90}},
    ]

    audio_file_id = _finalize(client).json()[0]["audioFileId"]

    with db_session_factory() as session:
        row = (
            session.query(TranscriptResult)
            .filter_by(audio_file_id=audio_file_id, asr_id="inception-stt")
            .one()
        )
        raw = row.raw_output

    assert raw is not None, "the engine's native output was dropped"
    assert len(raw) == 2, "every chunk must be represented, including the empty one"
    # The gateway fields the adapter discards are exactly what this preserves.
    assert raw[0]["response"]["audio_duration"] == 3.0
    assert raw[0]["response"]["usage"] == {"tokens": 7}
    assert raw[1]["response"]["text"] == ""


def test_finalize_without_native_output_stores_null_not_an_empty_list(
    client: TestClient, db_session_factory: sessionmaker[Session], live_session, stub_s3
) -> None:
    """An engine that reported no native payload leaves raw_output NULL, matching
    how a transcript that predates raw-output persistence reads."""
    live_session.create(["inception-stt"], 3.0)
    live_session.store["parts"]["inception-stt"] = [{"i": 0, "text": "one", "ms": 50}]

    audio_file_id = _finalize(client).json()[0]["audioFileId"]

    with db_session_factory() as session:
        row = session.query(TranscriptResult).filter_by(audio_file_id=audio_file_id).one()
        assert row.raw_output is None


def test_finalize_names_the_recording_by_timestamp(
    client: TestClient, db_session_factory: sessionmaker[Session], live_session, stub_s3
) -> None:
    """Named server-side so it matches the row's own created_at, and sortable as
    plain text because the recordings list orders by it."""
    import re

    live_session.create(["inception-stt"], 3.0)
    audio_file_id = _finalize(client).json()[0]["audioFileId"]

    with db_session_factory() as session:
        audio_file = session.get(AudioFile, audio_file_id)

    assert re.fullmatch(r"recording_\d{8}_\d{6}", audio_file.filename), audio_file.filename


def test_finalize_marks_the_recording_as_a_transcript_surface_recording(
    client: TestClient, db_session_factory: sessionmaker[Session], live_session, stub_s3
) -> None:
    """The whole point of the split: this must not land in the diarization list."""
    live_session.create(["inception-stt"], 3.0)
    audio_file_id = _finalize(client).json()[0]["audioFileId"]

    with db_session_factory() as session:
        assert session.get(AudioFile, audio_file_id).surface == "transcript"


def test_finalize_unknown_session_is_404(client: TestClient, live_session, stub_s3) -> None:
    response = client.post(
        "/transcript/session/nope/finalize",
        data={"referenceText": REFERENCE},
        files={"file": ("r.wav", make_wav_bytes(duration_sec=1.0), "audio/wav")},
    )
    assert response.status_code == 404


def test_finalize_empty_recording_is_422(client: TestClient, live_session, stub_s3) -> None:
    live_session.create(["inception-stt"], 3.0)
    response = client.post(
        "/transcript/session/test-session/finalize",
        data={"referenceText": REFERENCE},
        files={"file": ("r.wav", b"", "audio/wav")},
    )
    assert response.status_code == 422


def test_open_session_rejects_an_out_of_range_chunk_interval(
    client: TestClient, live_session
) -> None:
    response = client.post("/transcript/session",
                           json={"asrIds": ["inception-stt"], "chunkIntervalSec": 999})
    assert response.status_code == 422


def test_open_session_rejects_an_unknown_engine(client: TestClient, live_session) -> None:
    response = client.post("/transcript/session", json={"asrIds": ["nope"]})
    assert response.status_code == 422


# --- attaching a recording to the row its script already created ----------------
#
# A generated script creates its own recording row (see test_api_transcript_script.py),
# so finalize must fill THAT row in rather than insert a second one. Both failure modes
# here are invisible from a 200: a duplicate row gives one script two Projects entries,
# and a duplicate reference trips the unique index on audio_file_id.


def _script_only_row(
    factory: sessionmaker[Session], *, reference: str = REFERENCE, filename: str = "recording_20260820_120000"
) -> int:
    """The row a generated script leaves behind: no audio, reference already written.

    Named `recording_...` because that is what generate_script writes now -- one
    scheme for every row on this surface, with `has_audio` carrying whether
    audio exists rather than the filename.
    """
    with factory() as session:
        row = AudioFile(
            owner_id=session.query(User).filter_by(email=DEV_USER_EMAIL).one().id,
            filename=filename,
            duration_sec=0.0,
            surface="transcript",
        )
        session.add(row)
        session.flush()
        session.add(
            TranscriptReference(audio_file_id=row.id, source="script", text=reference, params={"minutes": 1})
        )
        session.commit()
        return row.id


def test_finalize_attaches_to_the_scripts_existing_row(
    client: TestClient, db_session_factory: sessionmaker[Session], live_session, stub_s3
) -> None:
    """One script, one Projects entry. A second AudioFile row here is the bug."""
    audio_file_id = _script_only_row(db_session_factory)
    live_session.create(["hamsa", "inception-stt"], 3.0)

    response = client.post(
        "/transcript/session/test-session/finalize",
        data={"referenceText": REFERENCE, "referenceSource": "script", "audioFileId": str(audio_file_id)},
        files={"file": ("read-aloud.wav", make_wav_bytes(duration_sec=2.0), "audio/wav")},
    )

    assert response.status_code == 200, response.text
    assert {run["audioFileId"] for run in response.json()} == {audio_file_id}
    with db_session_factory() as session:
        assert session.query(AudioFile).filter_by(surface="transcript").count() == 1
        row = session.get(AudioFile, audio_file_id)
        # The half that was missing is now filled in, with a MEASURED duration.
        assert row.s3_key == f"audio/{audio_file_id}.wav"
        assert row.duration_sec == pytest.approx(2.0, abs=0.05)
        # ...and it is a recording now, so it is named like one, keeping its timestamp.
        # Unchanged: the row was already named this, so finalize renames nothing
        # and the name still matches the row's own created_at.
        assert row.filename == "recording_20260820_120000"


def test_finalize_updates_the_scripts_reference_instead_of_inserting_a_second(
    client: TestClient, db_session_factory: sessionmaker[Session], live_session, stub_s3
) -> None:
    """`audio_file_id` is unique on transcript_references. The reference row already
    exists by now, so a blind insert 500s the whole route."""
    audio_file_id = _script_only_row(db_session_factory, reference="original script text")
    live_session.create(["hamsa"], 3.0)

    response = client.post(
        "/transcript/session/test-session/finalize",
        data={"referenceText": REFERENCE, "referenceSource": "script", "audioFileId": str(audio_file_id)},
        files={"file": ("read-aloud.wav", make_wav_bytes(duration_sec=2.0), "audio/wav")},
    )

    assert response.status_code == 200, response.text
    with db_session_factory() as session:
        rows = session.query(TranscriptReference).filter_by(audio_file_id=audio_file_id).all()
    assert len(rows) == 1
    # The text submitted at finalize is what was actually read, so it wins over the
    # text stored at generation (the operator can edit the box before reading).
    assert rows[0].text == REFERENCE


def test_finalize_refuses_to_record_over_existing_audio(
    client: TestClient, db_session_factory: sessionmaker[Session], live_session, stub_s3
) -> None:
    """Overwriting the audio would leave the previous take's transcripts and scores
    pointing at different audio than they were measured from."""
    audio_file_id = _script_only_row(db_session_factory)
    live_session.create(["hamsa"], 3.0)
    first = client.post(
        "/transcript/session/test-session/finalize",
        data={"referenceText": REFERENCE, "audioFileId": str(audio_file_id)},
        files={"file": ("read-aloud.wav", make_wav_bytes(duration_sec=2.0), "audio/wav")},
    )
    assert first.status_code == 200, first.text

    live_session.create(["hamsa"], 3.0)
    second = client.post(
        "/transcript/session/test-session/finalize",
        data={"referenceText": REFERENCE, "audioFileId": str(audio_file_id)},
        files={"file": ("read-aloud.wav", make_wav_bytes(duration_sec=2.0), "audio/wav")},
    )

    assert second.status_code == 409, second.text


def test_finalize_rejects_an_unknown_audio_file_id(
    client: TestClient, live_session, stub_s3
) -> None:
    """The id comes from the browser, so ownership and surface are checked, not trusted."""
    live_session.create(["hamsa"], 3.0)

    response = client.post(
        "/transcript/session/test-session/finalize",
        data={"referenceText": REFERENCE, "audioFileId": "424242"},
        files={"file": ("read-aloud.wav", make_wav_bytes(duration_sec=2.0), "audio/wav")},
    )

    assert response.status_code == 404, response.text


def test_finalize_without_an_audio_file_id_still_creates_a_row(
    client: TestClient, db_session_factory: sessionmaker[Session], live_session, stub_s3
) -> None:
    """A session opened from a pasted reference has no row yet, so the original path
    must keep working unchanged."""
    live_session.create(["hamsa"], 3.0)

    response = _finalize(client, source="pasted")

    assert response.status_code == 200, response.text
    audio_file_id = response.json()[0]["audioFileId"]
    with db_session_factory() as session:
        row = session.get(AudioFile, audio_file_id)
    assert row is not None
    assert row.s3_key == f"audio/{audio_file_id}.wav"
    assert row.filename.startswith("recording_")


def test_finalize_leaves_a_legacy_script_named_row_alone(
    client: TestClient, db_session_factory: sessionmaker[Session], live_session, stub_s3
) -> None:
    """Rows created before the `recording_` scheme keep the name they have.

    `attach_audio` used to rewrite `script_...` to `recording_...` on finalize.
    That rename is gone, and it should stay gone: renaming a row here would
    change a name someone may already have referenced, to fix nothing that
    `has_audio` does not already answer.
    """
    audio_file_id = _script_only_row(db_session_factory, filename="script_20260820_120000")
    live_session.create(["hamsa", "inception-stt"], 3.0)

    response = client.post(
        "/transcript/session/test-session/finalize",
        data={"referenceText": REFERENCE, "referenceSource": "script", "audioFileId": str(audio_file_id)},
        files={"file": ("read-aloud.wav", make_wav_bytes(duration_sec=2.0), "audio/wav")},
    )
    assert response.status_code == 200, response.text

    with db_session_factory() as session:
        row = session.get(AudioFile, audio_file_id)
        assert row.filename == "script_20260820_120000"
        assert row.s3_key, "the recording still attached, only the name was left alone"
