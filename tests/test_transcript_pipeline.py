"""The two-stage transcript pipeline: ASR, then forced alignment.

Both engines and the aligner are monkeypatched — no WebSocket, no vLLM
container, no GPU, no MinIO. What is under test is the staging itself: which
engine runs, what gets persisted when, how the two timings stay separate, and
what survives a failure in each half.
"""

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from rq import Queue
from sqlalchemy.orm import Session, sessionmaker, undefer

from apps.background_worker.transcription import pipeline
from packages.database.models import AudioFile, TranscriptResult, User
from packages.database.session import DEV_USER_EMAIL


@pytest.fixture()
def wired(
    db_session_factory: sessionmaker[Session], fake_queue: Queue, monkeypatch: pytest.MonkeyPatch
) -> Iterator[SimpleNamespace]:
    """A recording with a queued `hamsa` transcript row, with every external call
    stubbed. `add_row` seeds another engine's row on the same recording, which
    is legal now that rows are keyed on (audio_file_id, asr_id).

    `run_asr` imports the queue lazily inside the function (the API imports this
    module, and queue_app opens Redis at import time), so the patch targets the
    module it is imported FROM.
    """
    monkeypatch.setattr(pipeline, "SessionLocal", db_session_factory)
    monkeypatch.setattr("apps.background_worker.queue_app.queue", fake_queue)
    monkeypatch.setattr(pipeline, "download_to", lambda key, path: None)

    with db_session_factory() as session:
        owner = session.query(User).filter_by(email=DEV_USER_EMAIL).one()
        audio_file = AudioFile(owner_id=owner.id, filename="clip.wav", duration_sec=2.0, s3_key="audio/1.wav")
        session.add(audio_file)
        session.commit()
        session.add(TranscriptResult(audio_file_id=audio_file.id, asr_id="hamsa", status="queued"))
        session.commit()
        audio_file_id = audio_file.id

    def add_row(asr_id: str, **fields: object) -> None:
        with db_session_factory() as session:
            session.add(TranscriptResult(audio_file_id=audio_file_id, asr_id=asr_id, status="queued", **fields))
            session.commit()

    yield SimpleNamespace(
        audio_file_id=audio_file_id, queue=fake_queue, sessions=db_session_factory, add_row=add_row
    )


def _row(factory: sessionmaker[Session], audio_file_id: int, asr_id: str = "hamsa") -> TranscriptResult:
    with factory() as session:
        return (
            session.query(TranscriptResult)
            # raw_output is deferred on the model, so it is not loaded by
            # default -- and the row is detached once this session closes, after
            # which loading it raises DetachedInstanceError.
            .options(undefer(TranscriptResult.raw_output))
            .filter_by(audio_file_id=audio_file_id, asr_id=asr_id)
            .one()
        )


def _stub_engine(monkeypatch: pytest.MonkeyPatch, asr_id: str, text: str, calls: list[str] | None = None):
    def run(_path: str):
        if calls is not None:
            calls.append(asr_id)
        return {"native": text}

    monkeypatch.setattr(
        pipeline,
        "engine_for",
        lambda requested: SimpleNamespace(run=run, adapt=lambda raw: raw["native"]) if requested == asr_id else None,
    )


def test_asr_persists_text_and_its_own_timing_then_queues_alignment(
    wired: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_engine(monkeypatch, "hamsa", "مرحبا بكم")

    pipeline.run_asr(wired.audio_file_id, "hamsa")

    row = _row(wired.sessions, wired.audio_file_id)
    assert row.text == "مرحبا بكم"
    assert row.asr_ms is not None
    # Alignment has not run, so its timing must still be absent rather than 0 —
    # the panel distinguishes "not measured" from "measured as fast".
    assert row.align_ms is None
    assert row.words is None
    # The engine's native output, kept verbatim beside the text its adapter
    # reduced it to.
    assert row.raw_output == {"native": "مرحبا بكم"}

    jobs = wired.queue.get_jobs()
    assert [job.func_name.rsplit(".", 1)[-1] for job in jobs] == ["run_align"]
    # args[1] carries the stage's own id because supervisor/state.py reads
    # job.args[1] as the model id when deriving which models a worker is busy
    # on; args[2] is the engine whose row this alignment belongs to.
    assert jobs[0].args == (wired.audio_file_id, pipeline.ALIGNER_ID, "hamsa")


def test_the_job_argument_decides_the_engine(
    wired: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new run in the other mode between enqueue and execution must not switch
    engines mid-flight: the id chosen at enqueue travels with the job."""
    wired.add_row("cohere-transcribe")
    calls: list[str] = []
    _stub_engine(monkeypatch, "cohere-transcribe", "offline text", calls)

    pipeline.run_asr(wired.audio_file_id, "cohere-transcribe")

    assert calls == ["cohere-transcribe"]
    assert _row(wired.sessions, wired.audio_file_id, "cohere-transcribe").text == "offline text"


def test_each_engine_writes_only_its_own_row(
    wired: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The comparison this platform exists for: both engines' transcripts of one
    recording survive side by side, neither overwriting the other."""
    wired.add_row("cohere-transcribe")

    _stub_engine(monkeypatch, "hamsa", "online text")
    pipeline.run_asr(wired.audio_file_id, "hamsa")
    _stub_engine(monkeypatch, "cohere-transcribe", "offline text")
    pipeline.run_asr(wired.audio_file_id, "cohere-transcribe")

    assert _row(wired.sessions, wired.audio_file_id, "hamsa").text == "online text"
    assert _row(wired.sessions, wired.audio_file_id, "cohere-transcribe").text == "offline text"


def test_alignment_writes_only_the_row_its_engine_names(
    wired: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Aligning one engine's transcript must not put word timings on the other's."""
    wired.add_row("cohere-transcribe")
    _stub_engine(monkeypatch, "cohere-transcribe", "hello world")
    pipeline.run_asr(wired.audio_file_id, "cohere-transcribe")
    monkeypatch.setattr(
        pipeline.aligner_runner,
        "run",
        lambda path, text: [{"text": "hello", "start": 0.0, "end": 0.4, "score": 0.9}],
    )

    pipeline.run_align(wired.audio_file_id, pipeline.ALIGNER_ID, "cohere-transcribe")

    assert _row(wired.sessions, wired.audio_file_id, "cohere-transcribe").words is not None
    assert _row(wired.sessions, wired.audio_file_id, "hamsa").words is None


def test_unknown_engine_id_fails_the_row_rather_than_the_worker(wired: SimpleNamespace) -> None:
    """A row whose engine no longer exists (written by an older build) is failed
    honestly instead of crashing the worker."""
    wired.add_row("not-an-engine")

    pipeline.run_asr(wired.audio_file_id, "not-an-engine")

    row = _row(wired.sessions, wired.audio_file_id, "not-an-engine")
    assert row.status == "failed"
    assert "not-an-engine" in row.error
    assert wired.queue.count == 0


def test_empty_transcript_completes_without_queueing_alignment(
    wired: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Silent audio is a legitimate result, not a failure — and there is nothing
    to align, so the run finishes rather than queueing a no-op second stage."""
    _stub_engine(monkeypatch, "hamsa", "")

    pipeline.run_asr(wired.audio_file_id, "hamsa")

    row = _row(wired.sessions, wired.audio_file_id)
    assert row.status == "done"
    assert row.stage is None
    assert row.text == ""
    assert row.asr_ms is not None
    # This path commits separately from the normal one, and it is the case where
    # the native output matters most: it is what explains an empty transcript.
    assert row.raw_output == {"native": ""}
    assert wired.queue.count == 0


def test_asr_failure_marks_the_row_and_queues_nothing(
    wired: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(_path: str):
        raise RuntimeError("hamsa handshake rejected")

    monkeypatch.setattr(
        pipeline, "engine_for", lambda _id: SimpleNamespace(run=explode, adapt=lambda raw: raw)
    )

    pipeline.run_asr(wired.audio_file_id, "hamsa")

    row = _row(wired.sessions, wired.audio_file_id)
    assert row.status == "failed"
    assert "handshake rejected" in row.error
    assert wired.queue.count == 0


def test_alignment_persists_words_and_its_own_timing(
    wired: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_engine(monkeypatch, "hamsa", "hello world")
    pipeline.run_asr(wired.audio_file_id, "hamsa")
    monkeypatch.setattr(
        pipeline.aligner_runner,
        "run",
        lambda path, text: [
            {"text": "hello", "start": 0.0, "end": 0.4, "score": 0.9},
            {"text": "world", "start": 0.5, "end": 0.9, "score": 0.8},
        ],
    )

    pipeline.run_align(wired.audio_file_id, pipeline.ALIGNER_ID, "hamsa")

    row = _row(wired.sessions, wired.audio_file_id)
    assert row.status == "done"
    assert row.stage is None
    assert [word["w"] for word in row.words] == ["hello", "world"]
    # The two stages are timed separately — that separation is the whole reason
    # this is two jobs rather than one.
    assert row.asr_ms is not None
    assert row.align_ms is not None


def test_alignment_failure_keeps_the_asr_result(
    wired: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A good transcript must survive a failed alignment: the panel shows the
    text plus the error, rather than losing work that succeeded."""
    _stub_engine(monkeypatch, "hamsa", "hello world")
    pipeline.run_asr(wired.audio_file_id, "hamsa")

    def explode(path, text):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(pipeline.aligner_runner, "run", explode)

    pipeline.run_align(wired.audio_file_id, pipeline.ALIGNER_ID, "hamsa")

    row = _row(wired.sessions, wired.audio_file_id)
    assert row.status == "failed"
    assert "CUDA out of memory" in row.error
    assert row.text == "hello world"
    assert row.asr_ms is not None


def test_jobs_drop_quietly_when_the_recording_was_deleted(
    wired: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deleting a recording removes its transcript row; a job already queued
    against it must not crash the worker."""
    with wired.sessions() as session:
        session.query(TranscriptResult).delete()
        session.commit()

    pipeline.run_asr(wired.audio_file_id, "hamsa")
    pipeline.run_align(wired.audio_file_id, pipeline.ALIGNER_ID, "hamsa")

    assert wired.queue.count == 0


def test_recording_without_local_audio_fails_honestly(
    wired: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The transcript lane reads from MinIO; an azure-batch-only upload has no
    bytes here to transcribe."""
    with wired.sessions() as session:
        audio_file = session.query(AudioFile).filter_by(id=wired.audio_file_id).one()
        audio_file.s3_key = None
        session.commit()

    pipeline.run_asr(wired.audio_file_id, "hamsa")

    row = _row(wired.sessions, wired.audio_file_id)
    assert row.status == "failed"
    assert "MinIO" in row.error
