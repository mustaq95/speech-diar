"""The transcript endpoints: GET/POST /evaluations/{id}/transcript, the upload
hook, and delete cleanup.

The transcript is deliberately NOT part of the evaluation payload — these tests
pin that separation along with the rule that makes the comparison possible:
transcripts are keyed on (recording, engine), so running one mode never touches
the other mode's result, and each run stays labelled with the engine that made
it.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from rq import Queue
from sqlalchemy.orm import Session, sessionmaker

from packages.database.models import AudioFile, TranscriptResult, User
from packages.database.session import DEV_USER_EMAIL
from tests.conftest import make_wav_bytes


@pytest.fixture()
def stub_s3(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    calls: list[str] = []
    monkeypatch.setattr(
        "apps.backend_api.routers.upload.s3_client.put_stream", lambda fileobj, key: calls.append(key) or key
    )
    yield calls


@pytest.fixture(autouse=True)
def configured_engines(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Configure BOTH transcription engines for every test in this file.

    Without this the suite inherits whatever is in the developer's `.env`, which
    is the harness reaching into real config exactly like `conftest.py` refuses
    to do for the DB and queue. Online needs its credentials pinned; offline is
    "configured" only when its container answers a health probe, so that probe
    is stubbed healthy here (there is no real container in tests). Tests that
    care about an unconfigured engine override these explicitly.
    """
    from packages.config.settings import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "hamsa_stt_url", "wss://hamsa.test/ws")
    monkeypatch.setattr(settings, "hamsa_stt_key", "test-key")
    monkeypatch.setattr(
        "apps.background_worker.transcription.cohere_container_healthy", lambda s: True
    )
    yield


def _recording(factory: sessionmaker[Session], *, s3_key: str | None = "audio/1.wav") -> int:
    with factory() as session:
        owner = session.query(User).filter_by(email=DEV_USER_EMAIL).one()
        audio_file = AudioFile(owner_id=owner.id, filename="clip.wav", duration_sec=2.0, s3_key=s3_key)
        session.add(audio_file)
        session.commit()
        return audio_file.id


def _asr_jobs(queue: Queue) -> list[tuple]:
    return [job.args for job in queue.get_jobs() if job.func_name.endswith("run_asr")]


def test_config_reports_per_mode_availability(client: TestClient) -> None:
    """The panel renders the online/offline toggle from this: each mode's engine
    name and whether it is usable on this host. `defaultTranscriptionMode` only
    seeds the toggle; it must not infer the mode from an existing transcript,
    which records the engine that produced it (a different question)."""
    body = client.get("/config").json()

    assert body["defaultTranscriptionMode"] == "offline"
    assert body["transcriptionModes"]["online"] == {"asrName": "TryHamsa", "configured": True}
    assert body["transcriptionModes"]["offline"] == {"asrName": "Cohere", "configured": True}
    # The pre-existing key must keep working; the frontend reads both.
    assert isinstance(body["pollIntervalMs"], int)


def test_config_reports_an_unconfigured_mode(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Drives the disabled toggle side: offering a mode that is certain to fail
    is worse than saying why it cannot run."""
    from packages.config.settings import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "hamsa_stt_url", None)
    monkeypatch.setattr(settings, "hamsa_stt_ws_url", None)

    modes = client.get("/config").json()["transcriptionModes"]
    assert modes["online"]["configured"] is False
    # The other side stays usable — one unconfigured engine must not hide both.
    assert modes["offline"]["configured"] is True


def test_offline_is_unconfigured_when_the_container_is_down(
    client: TestClient, db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline is 'configured' only when its container answers, not merely when a
    URL is set. A down container disables the offline toggle and rejects a run
    that would fail at connect."""
    monkeypatch.setattr(
        "apps.background_worker.transcription.cohere_container_healthy", lambda s: False
    )

    assert client.get("/config").json()["transcriptionModes"]["offline"]["configured"] is False

    audio_file_id = _recording(db_session_factory)
    response = client.post(f"/evaluations/{audio_file_id}/transcript", json={"mode": "offline"})
    assert response.status_code == 422


def test_get_transcript_is_empty_when_none_was_ever_started(
    client: TestClient, db_session_factory: sessionmaker[Session]
) -> None:
    """A recording predating the feature, or a host with no engine configured.
    The client renders an empty list as an offer to run one, not as an error."""
    audio_file_id = _recording(db_session_factory)
    response = client.get(f"/evaluations/{audio_file_id}/transcript")
    assert response.status_code == 200
    assert response.json() == []


def test_get_transcript_returns_the_row_with_its_recorded_engine(
    client: TestClient, db_session_factory: sessionmaker[Session]
) -> None:
    audio_file_id = _recording(db_session_factory)
    with db_session_factory() as session:
        session.add(
            TranscriptResult(
                audio_file_id=audio_file_id,
                asr_id="cohere-transcribe",
                status="done",
                text="مرحبا",
                words=[{"w": "مرحبا", "s": 0.1, "e": 0.5, "score": 0.9}],
                asr_ms=1800,
                align_ms=420,
            )
        )
        session.commit()

    (body,) = client.get(f"/evaluations/{audio_file_id}/transcript").json()

    assert body["status"] == "done"
    assert body["asrId"] == "cohere-transcribe"
    assert body["mode"] == "offline"
    assert body["asrName"] == "Cohere"
    assert body["text"] == "مرحبا"
    assert body["words"] == [{"w": "مرحبا", "s": 0.1, "e": 0.5, "score": 0.9}]
    # The two stages report separately — that is what the panel exists to show.
    assert body["asrMs"] == 1800
    assert body["alignMs"] == 420


def test_mode_comes_from_the_engine_that_ran(
    client: TestClient, db_session_factory: sessionmaker[Session]
) -> None:
    """A row's mode is derived from its own asr_id, never from what a new run
    would use. A row produced online stays labelled online."""
    audio_file_id = _recording(db_session_factory)
    with db_session_factory() as session:
        session.add(TranscriptResult(audio_file_id=audio_file_id, asr_id="hamsa", status="done", text="hi"))
        session.commit()

    (body,) = client.get(f"/evaluations/{audio_file_id}/transcript").json()

    assert body["mode"] == "online"
    assert body["asrId"] == "hamsa"


def test_get_transcript_returns_both_engines_runs(
    client: TestClient, db_session_factory: sessionmaker[Session]
) -> None:
    """The comparison the panel exists for: one recording, one run per engine,
    each carrying its own text and its own timings."""
    audio_file_id = _recording(db_session_factory)
    with db_session_factory() as session:
        session.add_all([
            TranscriptResult(
                audio_file_id=audio_file_id, asr_id="hamsa", status="done", text="online text", asr_ms=989387
            ),
            TranscriptResult(
                audio_file_id=audio_file_id, asr_id="cohere-transcribe", status="done",
                text="offline text", asr_ms=13643,
            ),
        ])
        session.commit()

    body = client.get(f"/evaluations/{audio_file_id}/transcript").json()

    assert {run["mode"]: (run["text"], run["asrMs"]) for run in body} == {
        "online": ("online text", 989387),
        "offline": ("offline text", 13643),
    }


def test_post_transcript_enqueues_the_selected_online_engine(
    client: TestClient, db_session_factory: sessionmaker[Session], fake_queue: Queue
) -> None:
    audio_file_id = _recording(db_session_factory)

    response = client.post(f"/evaluations/{audio_file_id}/transcript", json={"mode": "online"})

    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    # args[1] is the engine id, matching the job shape supervisor/state.py reads.
    assert _asr_jobs(fake_queue) == [(audio_file_id, "hamsa")]


def test_post_transcript_enqueues_the_selected_offline_engine(
    client: TestClient, db_session_factory: sessionmaker[Session], fake_queue: Queue
) -> None:
    """The mode chosen on the request, not a server setting, picks the engine."""
    audio_file_id = _recording(db_session_factory)

    body = client.post(f"/evaluations/{audio_file_id}/transcript", json={"mode": "offline"}).json()

    assert body["asrId"] == "cohere-transcribe"
    assert body["mode"] == "offline"
    assert _asr_jobs(fake_queue) == [(audio_file_id, "cohere-transcribe")]


def test_post_transcript_422s_when_the_selected_mode_is_unconfigured(
    client: TestClient, db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Picking a mode whose engine has no credentials on this host is rejected
    rather than enqueued to a guaranteed failure."""
    from packages.config.settings import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "hamsa_stt_url", None)
    monkeypatch.setattr(settings, "hamsa_stt_ws_url", None)
    audio_file_id = _recording(db_session_factory)

    response = client.post(f"/evaluations/{audio_file_id}/transcript", json={"mode": "online"})

    assert response.status_code == 422


def test_re_running_the_same_mode_resets_that_engines_row(
    client: TestClient, db_session_factory: sessionmaker[Session], fake_queue: Queue
) -> None:
    """Rows are keyed on (recording, engine), so re-running an engine reuses its
    own row rather than piling up a second one."""
    audio_file_id = _recording(db_session_factory)
    with db_session_factory() as session:
        session.add(
            TranscriptResult(
                audio_file_id=audio_file_id, asr_id="hamsa", status="failed",
                error="boom", text="old", words=[{"w": "old"}], asr_ms=1, align_ms=2,
            )
        )
        session.commit()

    body = client.post(f"/evaluations/{audio_file_id}/transcript", json={"mode": "online"}).json()

    assert body["status"] == "queued"
    assert body["error"] is None
    with db_session_factory() as session:
        rows = session.query(TranscriptResult).filter_by(audio_file_id=audio_file_id).all()
        assert len(rows) == 1
        assert rows[0].asr_id == "hamsa"
        assert rows[0].text is None
        assert rows[0].words is None
        assert rows[0].asr_ms is None


def test_running_the_other_mode_keeps_the_first_engines_transcript(
    client: TestClient, db_session_factory: sessionmaker[Session], fake_queue: Queue
) -> None:
    """The bug this keying fixes: running offline used to wipe the online
    transcript, so a recording could never hold both for comparison."""
    audio_file_id = _recording(db_session_factory)
    with db_session_factory() as session:
        session.add(
            TranscriptResult(
                audio_file_id=audio_file_id, asr_id="hamsa", status="done",
                text="online text", words=[{"w": "online"}], asr_ms=989387, align_ms=21105,
            )
        )
        session.commit()

    body = client.post(f"/evaluations/{audio_file_id}/transcript", json={"mode": "offline"}).json()

    assert body["asrId"] == "cohere-transcribe"
    assert body["status"] == "queued"
    with db_session_factory() as session:
        rows = {row.asr_id: row for row in session.query(TranscriptResult).filter_by(audio_file_id=audio_file_id)}
        assert set(rows) == {"hamsa", "cohere-transcribe"}
        assert rows["hamsa"].text == "online text"
        assert rows["hamsa"].words == [{"w": "online"}]
        assert rows["hamsa"].asr_ms == 989387
        assert rows["cohere-transcribe"].text is None


def test_post_transcript_refuses_while_that_engine_is_already_running(
    client: TestClient, db_session_factory: sessionmaker[Session]
) -> None:
    audio_file_id = _recording(db_session_factory)
    with db_session_factory() as session:
        session.add(TranscriptResult(audio_file_id=audio_file_id, asr_id="hamsa", status="running", stage="asr"))
        session.commit()

    response = client.post(f"/evaluations/{audio_file_id}/transcript", json={"mode": "online"})
    assert response.status_code == 409


def test_the_other_mode_may_start_while_one_is_running(
    client: TestClient, db_session_factory: sessionmaker[Session], fake_queue: Queue
) -> None:
    """The 409 guards one engine, not the recording. Every upload auto-starts an
    online transcript that runs for minutes on a long recording; asking for the
    offline one during that window is a legitimate request, not a conflict."""
    audio_file_id = _recording(db_session_factory)
    with db_session_factory() as session:
        session.add(TranscriptResult(audio_file_id=audio_file_id, asr_id="hamsa", status="running", stage="asr"))
        session.commit()

    response = client.post(f"/evaluations/{audio_file_id}/transcript", json={"mode": "offline"})

    assert response.status_code == 200
    assert response.json()["asrId"] == "cohere-transcribe"
    assert _asr_jobs(fake_queue) == [(audio_file_id, "cohere-transcribe")]


def test_post_transcript_rejects_a_recording_with_no_local_audio(
    client: TestClient, db_session_factory: sessionmaker[Session]
) -> None:
    audio_file_id = _recording(db_session_factory, s3_key=None)
    response = client.post(f"/evaluations/{audio_file_id}/transcript", json={"mode": "online"})
    assert response.status_code == 400


def test_upload_enqueues_a_transcript_alongside_the_model_jobs(
    client: TestClient, stub_s3: list[str], db_session_factory: sessionmaker[Session], fake_queue: Queue
) -> None:
    """Every upload gets a transcript, independent of which diarizers were
    requested — it is not part of the model comparison. The auto path uses the
    default mode, which is offline: on-host and seconds, not minutes."""
    response = client.post(
        "/upload?models=pyannote", files={"file": ("clip.wav", make_wav_bytes(2.0), "audio/wav")}
    )
    audio_file_id = response.json()["audioFileId"]

    assert _asr_jobs(fake_queue) == [(audio_file_id, "cohere-transcribe")]
    with db_session_factory() as session:
        row = session.query(TranscriptResult).filter_by(audio_file_id=audio_file_id).one()
        assert row.status == "queued"
        assert row.asr_id == "cohere-transcribe"
    (body,) = client.get(f"/evaluations/{audio_file_id}/transcript").json()
    assert body["mode"] == "offline"


def test_upload_never_auto_starts_the_online_engine(
    client: TestClient, stub_s3: list[str], db_session_factory: sessionmaker[Session],
    fake_queue: Queue, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No fallback to online, ever. With the offline container down and online
    fully configured, the upload produces NO transcript rather than quietly
    streaming the audio to a remote service nobody chose. Online runs only when
    the operator asks for it from the panel."""
    monkeypatch.setattr(
        "apps.background_worker.transcription.cohere_container_healthy", lambda s: False
    )

    response = client.post(
        "/upload?models=pyannote", files={"file": ("clip.wav", make_wav_bytes(2.0), "audio/wav")}
    )
    audio_file_id = response.json()["audioFileId"]

    assert _asr_jobs(fake_queue) == []
    with db_session_factory() as session:
        assert session.query(TranscriptResult).filter_by(audio_file_id=audio_file_id).count() == 0
    # The diarization job is untouched — a missing transcript never affects models.
    assert any(job.func_name.endswith("run_model") for job in fake_queue.get_jobs())


def test_upload_skips_the_transcript_when_no_engine_is_configured(
    client: TestClient, stub_s3: list[str], db_session_factory: sessionmaker[Session],
    fake_queue: Queue, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host with no transcription engine at all gets a neutral "no transcript"
    panel, not a red failure on every single upload."""
    from packages.config.settings import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "hamsa_stt_url", None)
    monkeypatch.setattr(settings, "hamsa_stt_ws_url", None)
    monkeypatch.setattr(
        "apps.background_worker.transcription.cohere_container_healthy", lambda s: False
    )

    response = client.post(
        "/upload?models=pyannote", files={"file": ("clip.wav", make_wav_bytes(2.0), "audio/wav")}
    )
    audio_file_id = response.json()["audioFileId"]

    assert _asr_jobs(fake_queue) == []
    with db_session_factory() as session:
        assert session.query(TranscriptResult).filter_by(audio_file_id=audio_file_id).one_or_none() is None
    # The diarization job is untouched — a missing transcript never affects models.
    assert any(job.func_name.endswith("run_model") for job in fake_queue.get_jobs())


def test_deleting_a_recording_removes_every_transcript_row(
    client: TestClient, db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing cascades in this schema, so the child rows have to go explicitly
    — all of them, one per engine that ran."""
    monkeypatch.setattr("apps.backend_api.routers.evaluations.s3_client.delete_object", lambda key: None)
    audio_file_id = _recording(db_session_factory)
    with db_session_factory() as session:
        session.add_all([
            TranscriptResult(audio_file_id=audio_file_id, asr_id="hamsa", status="done", text="hi"),
            TranscriptResult(audio_file_id=audio_file_id, asr_id="cohere-transcribe", status="done", text="hi"),
        ])
        session.commit()

    assert client.delete(f"/evaluations/{audio_file_id}").status_code == 204
    with db_session_factory() as session:
        assert session.query(TranscriptResult).filter_by(audio_file_id=audio_file_id).count() == 0


def test_evaluation_payload_does_not_carry_the_transcript(
    client: TestClient, db_session_factory: sessionmaker[Session]
) -> None:
    """The word list is far too big to ride along on a payload polled every
    pollIntervalMs, and text has no place in the diarization contract."""
    audio_file_id = _recording(db_session_factory)
    with db_session_factory() as session:
        session.add(TranscriptResult(audio_file_id=audio_file_id, asr_id="hamsa", status="done", text="hi"))
        session.commit()

    body = client.get(f"/evaluations/{audio_file_id}").json()

    assert set(body) == {"audioFileId", "durationSec", "uploadMs", "models"}


# --- GET /evaluations/{id}/transcript/{asr_id}/raw ------------------------


def test_get_transcript_raw_output_returns_the_asr_engines_native_output(
    client: TestClient, db_session_factory: sessionmaker[Session]
) -> None:
    """hamsa's adapter joins its whole frame log down to one string. The route
    hands back the frames, which is the only place the per-frame detail exists."""
    audio_file_id = _recording(db_session_factory)
    frames = [
        {"type": "partial", "text": "mar"},
        {"type": "final", "text": "marhaba", "start": 0.1, "end": 0.9},
    ]
    with db_session_factory() as session:
        session.add(
            TranscriptResult(
                audio_file_id=audio_file_id, asr_id="hamsa", status="done", text="marhaba", raw_output=frames
            )
        )
        session.commit()

    response = client.get(f"/evaluations/{audio_file_id}/transcript/hamsa/raw")
    assert response.status_code == 200
    body = response.json()
    assert body["asrId"] == "hamsa"
    assert body["status"] == "done"
    assert body["rawOutput"] == frames
    assert body["run"]["text"] == "marhaba"


def test_get_transcript_raw_output_is_keyed_per_engine(
    client: TestClient, db_session_factory: sessionmaker[Session]
) -> None:
    """Both engines' transcripts coexist, so asking for one must never return the
    other's raw output."""
    audio_file_id = _recording(db_session_factory)
    with db_session_factory() as session:
        session.add_all([
            TranscriptResult(
                audio_file_id=audio_file_id, asr_id="hamsa", status="done", text="online", raw_output=[{"f": 1}]
            ),
            TranscriptResult(
                audio_file_id=audio_file_id,
                asr_id="cohere-transcribe",
                status="done",
                text="offline",
                raw_output={"text": "offline", "usage": {"total_tokens": 12}},
            ),
        ])
        session.commit()

    online = client.get(f"/evaluations/{audio_file_id}/transcript/hamsa/raw").json()
    offline = client.get(f"/evaluations/{audio_file_id}/transcript/cohere-transcribe/raw").json()

    assert online["rawOutput"] == [{"f": 1}]
    # The usage block is exactly what cohere's adapter drops on the way to text.
    assert offline["rawOutput"]["usage"] == {"total_tokens": 12}


def test_get_transcript_raw_output_is_null_before_the_asr_stage_finishes(
    client: TestClient, db_session_factory: sessionmaker[Session]
) -> None:
    audio_file_id = _recording(db_session_factory)
    with db_session_factory() as session:
        session.add(TranscriptResult(audio_file_id=audio_file_id, asr_id="hamsa", status="running", stage="asr"))
        session.commit()

    response = client.get(f"/evaluations/{audio_file_id}/transcript/hamsa/raw")
    assert response.status_code == 200
    assert response.json()["rawOutput"] is None


def test_get_transcript_raw_output_404s_when_that_engine_never_ran(
    client: TestClient, db_session_factory: sessionmaker[Session]
) -> None:
    audio_file_id = _recording(db_session_factory)

    assert client.get(f"/evaluations/{audio_file_id}/transcript/hamsa/raw").status_code == 404


def test_get_transcript_raw_output_404s_for_an_unknown_engine(
    client: TestClient, db_session_factory: sessionmaker[Session]
) -> None:
    audio_file_id = _recording(db_session_factory)

    assert client.get(f"/evaluations/{audio_file_id}/transcript/not-an-engine/raw").status_code == 404


def test_transcript_list_does_not_carry_raw_output(
    client: TestClient, db_session_factory: sessionmaker[Session]
) -> None:
    """The transcript list is polled on its own timer while a run is in flight;
    hamsa's frame log must not ride along on it."""
    audio_file_id = _recording(db_session_factory)
    with db_session_factory() as session:
        session.add(
            TranscriptResult(
                audio_file_id=audio_file_id, asr_id="hamsa", status="done", text="hi", raw_output=[{"f": 1}]
            )
        )
        session.commit()

    body = client.get(f"/evaluations/{audio_file_id}/transcript").json()
    assert len(body) == 1
    assert "rawOutput" not in body[0]


def test_rerunning_an_engine_clears_its_raw_output(
    client: TestClient, db_session_factory: sessionmaker[Session], fake_queue: Queue
) -> None:
    audio_file_id = _recording(db_session_factory)
    with db_session_factory() as session:
        session.add(
            TranscriptResult(
                audio_file_id=audio_file_id, asr_id="hamsa", status="done", text="old", raw_output=[{"f": 1}]
            )
        )
        session.commit()

    response = client.post(f"/evaluations/{audio_file_id}/transcript", json={"mode": "online"})
    assert response.status_code == 200

    raw = client.get(f"/evaluations/{audio_file_id}/transcript/hamsa/raw").json()
    assert raw["rawOutput"] is None
    assert raw["status"] == "queued"


def test_transcript_raw_output_column_is_not_loaded_by_a_default_query(
    db_session_factory: sessionmaker[Session],
) -> None:
    """The deferral itself, invisible over HTTP."""
    from sqlalchemy import inspect

    audio_file_id = _recording(db_session_factory)
    with db_session_factory() as session:
        session.add(
            TranscriptResult(
                audio_file_id=audio_file_id, asr_id="hamsa", status="done", text="hi", raw_output={"big": "blob"}
            )
        )
        session.commit()

    with db_session_factory() as session:
        row = session.query(TranscriptResult).filter_by(audio_file_id=audio_file_id).one()
        assert "raw_output" in inspect(row).unloaded
        assert row.raw_output == {"big": "blob"}
