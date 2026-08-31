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

    def create(asr_ids, chunk_interval_sec, reference_text="", modes=None):
        store["meta"] = {
            "asr_ids": list(asr_ids),
            "chunk_interval_sec": chunk_interval_sec,
            "reference_text": reference_text,
            "started_at": 0.0,
            "modes": dict(modes or {}),
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


# --- The chunk route dispatches on asr_id, and file-transport engines ---------
#
# Two failures that a green suite would not have shown.
#
# `POST /transcript/chunk` validated `asr_id` and then called the inception
# runner unconditionally, while filing the result under `asr_id`. With only one
# chunked engine registered, label and runner happened to agree and nothing was
# visibly wrong. A second chunked engine would have had Inception's text and
# measured latency stored under its own name at 200 OK.
#
# And cohere-transcribe is `transport="file"`: it is fed nothing live (3s chunks
# take the same Arabic clip from 391 Arabic characters to zero), so finalize has
# to queue it over the stored audio instead of writing it an empty live row.


@pytest.fixture()
def stub_queue(monkeypatch: pytest.MonkeyPatch):
    """Capture what finalize enqueues, without a worker or a Redis."""
    jobs: list[tuple] = []
    monkeypatch.setattr(
        "apps.backend_api.routers.transcript.queue",
        SimpleNamespace(enqueue=lambda fn, *args: jobs.append((fn.__name__, *args))),
    )
    return jobs


def test_chunk_route_runs_the_engine_it_was_asked_for(
    client: TestClient, live_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fails against the hardcoded version, which ran Inception for every asr_id.

    Registers a second chunked engine with its own runner and asserts the text
    that comes back is THAT engine's. With dispatch hardcoded, this returns
    Inception's text under the other engine's name and still answers 200.
    """
    from apps.background_worker.transcription import (
        ASR_ENGINES,
        ASR_TRANSPORTS,
        AsrEngine,
    )

    other = AsrEngine(
        asr_id="other-chunked", mode="online", name="Other",
        run=lambda path: {}, adapt=lambda raw: "",
        configured=lambda s: True,
        run_chunk=lambda payload, filename: {"text": "from the OTHER engine", "latency_ms": 42},
        chunk_text=lambda entry: entry["text"],
    )
    monkeypatch.setitem(ASR_ENGINES, "other-chunked", other)
    monkeypatch.setitem(ASR_TRANSPORTS, "other-chunked", "chunks")
    # Inception's own chunk call must never be reached for this request.
    monkeypatch.setattr(
        "apps.background_worker.transcription.inception.runner.transcribe_bytes",
        lambda *a, **k: pytest.fail("the chunk route called Inception for another engine"),
    )
    live_session.create(["other-chunked"], 3.0)

    response = client.post(
        "/transcript/chunk",
        data={"sessionId": "test-session", "asrId": "other-chunked", "chunkIndex": "0"},
        files={"file": ("chunk.wav", make_wav_bytes(duration_sec=1.0), "audio/wav")},
    )

    assert response.status_code == 200, response.text
    assert response.json()["text"] == "from the OTHER engine"
    assert response.json()["latencyMs"] == 42
    # And it was stored under the engine that actually produced it.
    assert live_session.store["parts"]["other-chunked"][0]["text"] == "from the OTHER engine"


def test_chunk_route_refuses_an_engine_running_in_batch_mode(
    client: TestClient, live_session
) -> None:
    """Cohere CAN be chunked, so capability alone no longer decides. The session's
    pick does: a run opened in batch mode must not accept live chunks, or the row
    would carry live timings while `run_asr` also writes it from storage."""
    live_session.create(["cohere-transcribe"], 3.0, modes={"cohere-transcribe": "batch"})

    response = client.post(
        "/transcript/chunk",
        data={"sessionId": "test-session", "asrId": "cohere-transcribe", "chunkIndex": "0"},
        files={"file": ("chunk.wav", make_wav_bytes(duration_sec=1.0), "audio/wav")},
    )

    assert response.status_code == 422
    assert "batch" in response.json()["detail"]
    assert live_session.store["parts"]["cohere-transcribe"] == []


def test_chunk_route_accepts_cohere_when_the_session_chose_live(
    client: TestClient, live_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half: the same engine, the other pick, is fed here and its own
    runner is what runs.

    The stub replaces the REGISTRY ENTRY, not the runner module attribute.
    `AsrEngine.run_chunk` binds the function object at import time, so patching
    `cohere.runner.transcribe_bytes` leaves the frozen dataclass still holding the
    original -- and this test then quietly transcribed a silent WAV against the
    real container on :9025 and asserted on its answer. Nothing in the harness is
    allowed to reach real infrastructure; that it passed a 200 back made it look
    like it was working.
    """
    import dataclasses

    from apps.background_worker.transcription import ASR_ENGINES

    monkeypatch.setitem(
        ASR_ENGINES,
        "cohere-transcribe",
        dataclasses.replace(
            ASR_ENGINES["cohere-transcribe"],
            run_chunk=lambda payload, filename="chunk.wav": {
                "response": {"text": "from cohere", "usage": {"type": "duration", "seconds": 3}},
                "latency_ms": 130,
            },
        ),
    )
    live_session.create(["cohere-transcribe"], 3.0, modes={"cohere-transcribe": "live"})

    response = client.post(
        "/transcript/chunk",
        data={"sessionId": "test-session", "asrId": "cohere-transcribe", "chunkIndex": "0"},
        files={"file": ("chunk.wav", make_wav_bytes(duration_sec=1.0), "audio/wav")},
    )

    assert response.status_code == 200, response.text
    assert response.json()["text"] == "from cohere"
    assert response.json()["latencyMs"] == 130
    assert live_session.store["parts"]["cohere-transcribe"][0]["text"] == "from cohere"


def test_finalize_queues_a_batch_mode_engine_over_the_stored_audio(
    client: TestClient, db_session_factory: sessionmaker[Session],
    live_session, stub_s3, stub_queue,
) -> None:
    """Cohere is fed nothing live, so finalize must QUEUE it, not write it an
    empty transcript. An empty 'done' row would score as total error and read as
    the engine having failed at a job it was never given."""
    live_session.create(["hamsa", "cohere-transcribe"], 3.0,
                        modes={"hamsa": "live", "cohere-transcribe": "batch"})
    live_session.store["parts"]["hamsa"] = [{"i": 0, "text": "one two three", "ms": 120}]

    runs = _finalize(client).json()

    by_id = {run["asrId"]: run for run in runs}
    assert set(by_id) == {"hamsa", "cohere-transcribe"}

    # The live engine is unchanged by cohere's presence.
    assert by_id["hamsa"]["status"] == "done"
    assert by_id["hamsa"]["source"] == "live"
    assert by_id["hamsa"]["transport"] == "stream"

    # The file engine is queued, labelled batch/file, and carries no fabricated
    # live measurements.
    cohere = by_id["cohere-transcribe"]
    assert cohere["status"] == "queued"
    assert cohere["source"] == "batch"
    assert cohere["transport"] == "file"
    assert not cohere.get("text")
    assert cohere.get("chunkCount") is None
    assert cohere.get("firstLatencyMs") is None

    # Exactly one job, for the file engine only, enqueued after the commit.
    assert stub_queue == [("run_asr", by_id["hamsa"]["audioFileId"], "cohere-transcribe", "batch")]


def test_finalize_keeps_the_live_transcripts_when_the_batch_engine_is_gone(
    client: TestClient, db_session_factory: sessionmaker[Session],
    live_session, stub_s3, stub_queue, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The container can stop between opening the session and finalizing. The
    live transcripts were measured while someone was speaking and cannot be
    reproduced; the batch one can be re-run from the recording. So the batch
    engine fails alone and takes nothing with it."""
    monkeypatch.setattr(
        "apps.background_worker.transcription.cohere_container_healthy", lambda s: False
    )
    live_session.create(["hamsa", "cohere-transcribe"], 3.0,
                        modes={"hamsa": "live", "cohere-transcribe": "batch"})
    live_session.store["parts"]["hamsa"] = [{"i": 0, "text": "one two three", "ms": 120}]

    response = _finalize(client)

    assert response.status_code == 200, response.text
    by_id = {run["asrId"]: run for run in response.json()}
    assert by_id["hamsa"]["status"] == "done"
    assert by_id["hamsa"]["text"] == "one two three"
    assert by_id["cohere-transcribe"]["status"] == "failed"
    assert stub_queue == []

    # And the live transcript really is on disk, not just in the response.
    with db_session_factory() as session:
        stored = session.query(TranscriptResult).filter_by(
            audio_file_id=by_id["hamsa"]["audioFileId"], asr_id="hamsa"
        ).one()
        assert stored.text == "one two three"


def test_finalize_keeps_cohere_live_when_the_session_chose_live(
    client: TestClient, db_session_factory: sessionmaker[Session],
    live_session, stub_s3, stub_queue,
) -> None:
    """The toggle's other half at finalize.

    Chosen as chunks, cohere is a LIVE row like Inception: done, source=live,
    transport=chunks, carrying the interval and the latencies measured while
    someone was speaking — and NOT queued, because there is nothing to run.
    Reading the transport from the static registry instead of the session would
    queue it here and label it "file", losing the live measurement entirely.
    """
    live_session.create(["cohere-transcribe"], 3.0,
                        modes={"cohere-transcribe": "live"})
    live_session.store["parts"]["cohere-transcribe"] = [
        {"i": 0, "text": "one two", "ms": 130},
        {"i": 1, "text": "three", "ms": 150},
    ]

    run = _finalize(client).json()[0]

    assert run["asrId"] == "cohere-transcribe"
    assert run["status"] == "done"
    assert run["source"] == "live"
    assert run["transport"] == "chunks"
    assert run["chunkIntervalSec"] == 3.0
    assert run["chunkCount"] == 2
    assert run["avgLatencyMs"] == 140
    assert run["text"] == "one two three"
    assert stub_queue == [], "a live cohere run has nothing to queue"


def test_the_same_engine_takes_either_mode_across_two_sessions(
    client: TestClient, db_session_factory: sessionmaker[Session],
    live_session, stub_s3, stub_queue,
) -> None:
    """One engine, two runs, two honest labels — which is the point of the toggle.

    Guards the thing that would quietly break it: a transport resolved from the
    registry at finalize rather than from the session would give both runs the
    same label no matter what was picked.
    """
    live_session.create(["cohere-transcribe"], 3.0,
                        modes={"cohere-transcribe": "live"})
    live_session.store["parts"]["cohere-transcribe"] = [{"i": 0, "text": "live text", "ms": 130}]
    chunked = _finalize(client).json()[0]

    live_session.create(["cohere-transcribe"], 3.0,
                        modes={"cohere-transcribe": "batch"})
    filed = _finalize(client).json()[0]

    assert (chunked["transport"], chunked["source"], chunked["status"]) == ("chunks", "live", "done")
    assert (filed["transport"], filed["source"], filed["status"]) == ("file", "batch", "queued")
    assert stub_queue == [("run_asr", filed["audioFileId"], "cohere-transcribe", "batch")]


def test_both_engines_get_the_whole_file_in_batch_mode(
    client: TestClient, db_session_factory: sessionmaker[Session],
    live_session, stub_s3, stub_queue,
) -> None:
    """The symmetry batch mode exists for.

    Both engines are fed the SAME input in batch -- one whole-file call each --
    so their error rates are comparable on identical audio. Inception used to
    stay chunked here, which made its batch panel indistinguishable from its
    stream panel and meant the two engines were never compared on the same thing.

    The cost is real and accepted: this gateway silently truncates long audio
    (140 words split vs 65 whole, on one 65s recording), so inception's batch WER
    carries that. INCEPTION_BATCH_WHOLE_FILE is what trades it, and
    `transport_for` follows the same setting so the label cannot drift from it.
    """
    live_session.create(
        ["inception-stt", "cohere-transcribe"], 3.0,
        modes={"inception-stt": "batch", "cohere-transcribe": "batch"},
    )

    by_id = {run["asrId"]: run for run in _finalize(client).json()}

    assert by_id["inception-stt"]["transport"] == "file"
    assert by_id["cohere-transcribe"]["transport"] == "file", "both, or it is not a comparison"
    assert all(run["source"] == "batch" for run in by_id.values())
    assert all(run["status"] == "queued" for run in by_id.values())
    # No chunk figures on either: one call has no interval and no chunk count, and
    # a fabricated 1 would read as a measurement.
    for run in by_id.values():
        assert run.get("chunkIntervalSec") is None
        assert run.get("chunkCount") is None
    assert sorted(job[2] for job in stub_queue) == ["cohere-transcribe", "inception-stt"]


def test_live_mode_chunks_both_engines_identically(
    client: TestClient, db_session_factory: sessionmaker[Session],
    live_session, stub_s3, stub_queue,
) -> None:
    """The other half of the symmetry: in LIVE mode both take 3s chunks.

    Guards the pair. Making batch symmetric by moving inception to whole-file
    must not have moved its live transport too -- live is where the two engines
    are compared on identical short chunks.
    """
    live_session.create(
        ["inception-stt", "cohere-transcribe"], 3.0,
        modes={"inception-stt": "live", "cohere-transcribe": "live"},
    )
    for asr_id in ("inception-stt", "cohere-transcribe"):
        live_session.store["parts"][asr_id] = [{"i": 0, "text": "one two", "ms": 100}]

    by_id = {run["asrId"]: run for run in _finalize(client).json()}

    for asr_id in ("inception-stt", "cohere-transcribe"):
        assert by_id[asr_id]["transport"] == "chunks"
        assert by_id[asr_id]["source"] == "live"
        assert by_id[asr_id]["chunkIntervalSec"] == 3.0
    assert stub_queue == [], "a live run has nothing to queue"


def test_chunk_route_refuses_inception_in_batch_mode(client: TestClient, live_session) -> None:
    """The engine that has always owned this route is not exempt: in batch mode it
    must be refused too, or the row would carry live chunk timings while `run_asr`
    independently overwrote it from storage."""
    live_session.create(["inception-stt"], 3.0, modes={"inception-stt": "batch"})

    response = client.post(
        "/transcript/chunk",
        data={"sessionId": "test-session", "asrId": "inception-stt", "chunkIndex": "0"},
        files={"file": ("chunk.wav", make_wav_bytes(duration_sec=1.0), "audio/wav")},
    )

    assert response.status_code == 422
    assert "batch" in response.json()["detail"]
    assert live_session.store["parts"]["inception-stt"] == []
