"""End-to-end coverage of the Clone routes.

Stubbed at the runner's HTTP boundary, NOT at the route: a green suite proved
nothing once before in this repo, when `finalize_session` 500'd on every real
call while every test passed because nothing drove the route body. These drive
the real handlers and assert on PERSISTED STATE.

Two shapes here are the ones that have already bitten this codebase twice, and
both are asserted directly:
  * registration UPDATES the extraction row rather than adding a second one --
    a blind `db.add` is the `finalize_session` and `tts_results` incident again,
    and here it would additionally orphan the tokens on the first row;
  * a failed pod call still writes/keeps a row, so "never tried" and "the pod
    rejected this" stay distinguishable.

Like `test_api_tts.py`, the routes open their own `SessionLocal()` for writes
(they must close the request-scoped session before a pod call that can outlast
the 60s idle-in-transaction timeout), which bypasses the `get_db` override --
so `SessionLocal` is patched to the test factory here.
"""

import wave
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, undefer

from apps.background_worker.voice_clone import runner as clone_runner
from packages.config.settings import get_settings
from packages.database.models import ClonedVoice, User
from packages.database.session import DEV_USER_EMAIL

GLOBAL_IDS = [8213, 1184, 4427, 331, 9042]
SEMANTIC_IDS = [412, 7761, 2038, 5590, 118, 907]
CLIP_URL = "https://cdn.example.test/refs/mariam.wav"
PROMPT = "مرحبا بكم في هيئة أبوظبي الرقمية"


def _wav(duration_sec: float = 1.0, framerate: int = 16000) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(framerate)
        out.writeframes(b"\x00\x00" * int(duration_sec * framerate))
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def configured_clone(monkeypatch: pytest.MonkeyPatch):
    """Cloning and the preview engine both configured, so engine selection is
    never what fails.

    Patched on the `get_settings()` SINGLETON, not the Settings class: these are
    pydantic fields living in the instance dict, so a class-level setattr is a
    silent no-op that leaves the real .env showing through.
    """
    settings = get_settings()
    for field, value in (
        ("hamsa_voice_clone_url", "https://pod.test/tts/voice_clone"),
        ("hamsa_load_voice_url", "https://pod.test/tts/load_voice_cloning"),
        ("hamsa_tts_api_url", "https://pod.test/tts/stream"),
        ("hamsa_tts_key", "k"),
        ("hamsa_tts_bearer_token", "b"),
        ("voice_clone_dialects", "msa,uae,egy,ksa,jor"),
        ("voice_clone_max_prompt_chars", 4000),
        ("voice_clone_public_base_url", None),
        ("tts_max_input_chars", 4000),
    ):
        monkeypatch.setattr(settings, field, value)


@pytest.fixture(autouse=True)
def write_session(monkeypatch: pytest.MonkeyPatch, db_session_factory):
    """Point the routes' own write sessions at the test database."""
    monkeypatch.setattr("apps.backend_api.routers.voice_clone.SessionLocal", db_session_factory)


@pytest.fixture()
def store(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """In-memory stand-in for MinIO, so the routes exercise their real storage
    calls (key layout, Range slicing) without an object store."""
    objects: dict[str, bytes] = {}

    def put_stream(fileobj, key):
        objects[key] = fileobj.read()
        return key

    monkeypatch.setattr("apps.backend_api.routers.voice_clone.s3_client.put_stream", put_stream)
    monkeypatch.setattr(
        "apps.backend_api.routers.voice_clone.s3_client.head_object", lambda key: len(objects[key])
    )
    monkeypatch.setattr(
        "apps.backend_api.routers.evaluations.s3_client.open_stream",
        lambda key, start=0: BytesIO(objects[key][start:]),
    )
    monkeypatch.setattr(
        "apps.backend_api.routers.voice_clone.s3_client.delete_object",
        lambda key: objects.pop(key, None),
    )
    return objects


def _stub_extract(monkeypatch: pytest.MonkeyPatch, *, body=None, error: Exception | None = None):
    """Replace the pod call at its HTTP boundary, leaving the adapter, the
    validation and the whole route body running for real."""

    def fake_extract(audio_url: str, prompt_text: str) -> dict:
        if error is not None:
            raise error
        return {
            "body": body
            if body is not None
            else {
                "global_token_ids": GLOBAL_IDS,
                "semantic_token_ids": SEMANTIC_IDS,
                "prompt_text": prompt_text,
            },
            "headers": {"content-type": "application/json"},
            "status_code": 200,
            "elapsed_ms": 1400,
        }

    monkeypatch.setattr(clone_runner, "extract", fake_extract)


def _stub_register(monkeypatch: pytest.MonkeyPatch, *, error: Exception | None = None, seen: list | None = None):
    def fake_register(speaker_id, global_token_ids, semantic_token_ids, dialect, prompt_text) -> dict:
        if seen is not None:
            seen.append((speaker_id, global_token_ids, semantic_token_ids, dialect, prompt_text))
        if error is not None:
            raise error
        return {"body": None, "headers": {}, "status_code": 200, "elapsed_ms": 310}

    monkeypatch.setattr(clone_runner, "register", fake_register)


def _extract(client: TestClient, **overrides) -> dict:
    payload = {"audioUrl": CLIP_URL, "promptText": PROMPT, "dialect": "uae"}
    payload.update(overrides)
    response = client.post("/voice-clone/extract", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


# --- step 1: extraction ----------------------------------------------------


def test_extract_persists_the_tokens_and_reports_the_counts(
    client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_extract(monkeypatch)
    body = _extract(client)

    assert body["status"] == "extracted"
    assert body["globalTokenCount"] == len(GLOBAL_IDS)
    assert body["semanticTokenCount"] == len(SEMANTIC_IDS)
    assert body["extractMs"] == 1400
    # No name yet: registration is what names a voice, and a placeholder here
    # would put a speaker_id in the table the pod has never heard of.
    assert body["speakerId"] == ""

    row = (
        db_session.query(ClonedVoice)
        .options(undefer(ClonedVoice.global_token_ids), undefer(ClonedVoice.semantic_token_ids))
        .one()
    )
    assert row.global_token_ids == GLOBAL_IDS
    assert row.semantic_token_ids == SEMANTIC_IDS
    assert row.dialect == "uae"
    assert row.prompt_text == PROMPT
    assert row.audio_url == CLIP_URL


def test_the_token_arrays_never_cross_the_wire(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the counts. Thousands of integers the UI has no use for would be
    dead weight on every list response."""
    _stub_extract(monkeypatch)
    body = _extract(client)
    assert "globalTokenIds" not in body
    assert "semanticTokenIds" not in body


def test_a_rejected_clip_still_writes_a_row_carrying_the_pods_words(
    client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure this endpoint actually produces is an opaque text/plain 500.
    It is kept verbatim: without the row, 'never tried' and 'the pod rejected
    this clip' are indistinguishable, and rewording it removes the only clue."""
    _stub_extract(
        monkeypatch,
        error=clone_runner.VoiceCloneUpstreamError(500, "Internal Server Error"),
    )
    body = _extract(client)

    assert body["status"] == "failed"
    assert "Internal Server Error" in body["error"]
    assert body["globalTokenCount"] is None

    row = db_session.query(ClonedVoice).one()
    assert row.status == "failed"
    assert row.global_token_ids is None


def test_extract_requires_the_transcript(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """prompt_text is a required INPUT, not a note: the pod uses it to separate
    voice identity from spoken content."""
    _stub_extract(monkeypatch)
    response = client.post("/voice-clone/extract", json={"audioUrl": CLIP_URL, "promptText": "   "})
    assert response.status_code == 422
    assert "verbatim transcript" in response.json()["detail"]


def test_extract_rejects_a_dialect_this_host_does_not_offer(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_extract(monkeypatch)
    response = client.post(
        "/voice-clone/extract", json={"audioUrl": CLIP_URL, "promptText": PROMPT, "dialect": "klingon"}
    )
    assert response.status_code == 422
    assert "msa" in response.json()["detail"]


def test_extract_enforces_the_prompt_limit(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_extract(monkeypatch)
    settings = get_settings()
    response = client.post(
        "/voice-clone/extract",
        json={"audioUrl": CLIP_URL, "promptText": "x" * (settings.voice_clone_max_prompt_chars + 1)},
    )
    assert response.status_code == 422
    assert "over this host's" in response.json()["detail"]


def test_an_unconfigured_host_refuses_rather_than_failing_upstream(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "hamsa_voice_clone_url", None)
    response = client.post("/voice-clone/extract", json={"audioUrl": CLIP_URL, "promptText": PROMPT})
    assert response.status_code == 422
    assert "not configured" in response.json()["detail"]


# --- step 2: registration ---------------------------------------------------


def test_register_updates_the_extraction_row_and_never_adds_a_second(
    client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `finalize_session` / `tts_results` incident, third time. A blind
    `db.add` here trips the unique index AND orphans the tokens on row one."""
    _stub_extract(monkeypatch)
    _stub_register(monkeypatch)
    voice = _extract(client)

    response = client.post(f"/voice-clone/{voice['id']}/register", json={"speakerId": "Mariam"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "registered"
    assert body["speakerId"] == "Mariam"
    assert body["registerMs"] == 310
    assert body["registeredAt"] is not None

    assert db_session.query(ClonedVoice).count() == 1
    row = db_session.query(ClonedVoice).one()
    assert row.id == voice["id"]
    assert row.status == "registered"


def test_register_reposts_the_stored_tokens_verbatim(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-extraction is never performed: the stored arrays are the whole reason
    a voice can be re-registered after a pod restart drops it."""
    seen: list = []
    _stub_extract(monkeypatch)
    _stub_register(monkeypatch, seen=seen)
    voice = _extract(client)
    client.post(f"/voice-clone/{voice['id']}/register", json={"speakerId": "Mariam", "dialect": "egy"})

    assert len(seen) == 1
    speaker, global_ids, semantic_ids, dialect, prompt = seen[0]
    assert speaker == "Mariam"
    assert global_ids == GLOBAL_IDS
    assert semantic_ids == SEMANTIC_IDS
    assert dialect == "egy"
    assert prompt == PROMPT


def test_a_failed_registration_keeps_the_tokens_so_a_retry_is_free(
    client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_extract(monkeypatch)
    _stub_register(monkeypatch, error=clone_runner.VoiceCloneUpstreamError(503, "no capacity"))
    voice = _extract(client)

    body = client.post(f"/voice-clone/{voice['id']}/register", json={"speakerId": "Mariam"}).json()
    assert body["status"] == "extracted"
    assert "no capacity" in body["error"]

    row = (
        db_session.query(ClonedVoice)
        .options(undefer(ClonedVoice.global_token_ids))
        .filter_by(id=voice["id"])
        .one()
    )
    assert row.global_token_ids == GLOBAL_IDS
    assert row.registered_at is None


def test_registering_a_name_this_host_already_holds_is_a_409(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reported as a clash BEFORE the pod call: the pod overwrites in memory
    silently, so letting the call through would destroy the other voice and
    then fail on our unique index afterwards."""
    _stub_extract(monkeypatch)
    _stub_register(monkeypatch)
    first = _extract(client)
    client.post(f"/voice-clone/{first['id']}/register", json={"speakerId": "Mariam"})

    second = _extract(client)
    response = client.post(f"/voice-clone/{second['id']}/register", json={"speakerId": "Mariam"})
    assert response.status_code == 409
    assert "overwrite" in response.json()["detail"]


def test_registering_a_failed_extraction_is_refused(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_extract(monkeypatch, error=clone_runner.VoiceCloneUpstreamError(500, "Internal Server Error"))
    _stub_register(monkeypatch)
    voice = _extract(client)
    response = client.post(f"/voice-clone/{voice['id']}/register", json={"speakerId": "Mariam"})
    assert response.status_code == 422
    assert "no extracted tokens" in response.json()["detail"]


def test_register_requires_a_name(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_extract(monkeypatch)
    voice = _extract(client)
    response = client.post(f"/voice-clone/{voice['id']}/register", json={"speakerId": "  "})
    assert response.status_code == 422


def test_register_on_an_unknown_voice_is_a_404(client: TestClient) -> None:
    assert client.post("/voice-clone/9999/register", json={"speakerId": "x"}).status_code == 404


# --- step 3: hearing it -----------------------------------------------------


def _stub_preview(monkeypatch: pytest.MonkeyPatch, *, audio: bytes, error: Exception | None = None, seen: list | None = None):
    """Stub `hamsa-tts`'s runner, NOT the preview route: a cloned voice must be
    usable through the ordinary engine, and a special code path here would test
    something that does not exist in production."""
    from apps.background_worker.tts.hamsa import runner as hamsa_runner

    def fake_run(text: str, voice: str) -> dict:
        if seen is not None:
            seen.append((text, voice))
        if error is not None:
            raise error
        return {
            "audio": audio,
            "status_code": 200,
            "headers": {"content-type": "text/event-stream"},
            "synth_ms": 900,
            "first_audio_ms": 120,
            "voice": voice,
        }

    monkeypatch.setattr(hamsa_runner, "run", fake_run)
    # TTS_ENGINES captured the original function reference at import time, so
    # patching the module attribute alone leaves the registry pointing at the
    # real HTTP call. Same shape as test_api_tts.py's `_stub_engine`.
    from dataclasses import replace

    from apps.background_worker.tts import TTS_ENGINES

    monkeypatch.setitem(TTS_ENGINES, "hamsa-tts", replace(TTS_ENGINES["hamsa-tts"], run=fake_run))


def _registered(client: TestClient, monkeypatch: pytest.MonkeyPatch, name: str = "Mariam") -> dict:
    _stub_extract(monkeypatch)
    _stub_register(monkeypatch)
    voice = _extract(client)
    return client.post(f"/voice-clone/{voice['id']}/register", json={"speakerId": name}).json()


def test_preview_synthesizes_through_the_ordinary_engine_with_the_cloned_name(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, store: dict
) -> None:
    seen: list = []
    voice = _registered(client, monkeypatch)
    _stub_preview(monkeypatch, audio=_wav(2.0), seen=seen)

    response = client.post(f"/voice-clone/{voice['id']}/preview", json={"text": "الربع الأول"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["speakerId"] == "Mariam"
    assert body["synthMs"] == 900
    assert body["firstAudioMs"] == 120
    assert body["audioSec"] == pytest.approx(2.0, abs=0.05)

    # The registered NAME reached the engine as its speaker — the whole point of
    # cloning is that it is used exactly like a built-in voice.
    assert seen == [("الربع الأول", "Mariam")]
    assert any(key.startswith(f"voice-clone/{voice['id']}/preview") for key in store)


def test_preview_before_registration_is_refused(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pod cannot synthesize with a name it does not hold, and letting the
    call through would surface that as an opaque upstream failure."""
    _stub_extract(monkeypatch)
    voice = _extract(client)
    response = client.post(f"/voice-clone/{voice['id']}/preview", json={"text": "hi"})
    assert response.status_code == 422
    assert "not registered" in response.json()["detail"]


def test_a_dropped_voice_surfaces_as_a_502_naming_the_speaker(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, store: dict
) -> None:
    """This is the signal a pod restart produces. The pod exposes no route that
    lists what it holds, so a failed synthesis naming the speaker is the ONLY
    way this repo can learn the registration is gone."""
    voice = _registered(client, monkeypatch)
    from apps.background_worker.tts.hamsa.runner import HamsaTtsUpstreamError

    _stub_preview(monkeypatch, audio=b"", error=HamsaTtsUpstreamError(400, "unknown speaker"))
    response = client.post(f"/voice-clone/{voice['id']}/preview", json={"text": "hi"})
    assert response.status_code == 502
    assert "Mariam" in response.json()["detail"]


def test_the_preview_clip_streams_back_with_its_own_media_type(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, store: dict
) -> None:
    voice = _registered(client, monkeypatch)
    _stub_preview(monkeypatch, audio=_wav(1.0))
    client.post(f"/voice-clone/{voice['id']}/preview", json={"text": "hi"})

    response = client.get(f"/voice-clone/{voice['id']}/preview/audio")
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["accept-ranges"] == "bytes"
    assert response.content.startswith(b"RIFF")


def test_the_preview_clip_honours_a_range_request(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, store: dict
) -> None:
    voice = _registered(client, monkeypatch)
    _stub_preview(monkeypatch, audio=_wav(1.0))
    client.post(f"/voice-clone/{voice['id']}/preview", json={"text": "hi"})

    response = client.get(f"/voice-clone/{voice['id']}/preview/audio", headers={"Range": "bytes=0-99"})
    assert response.status_code == 206
    assert len(response.content) == 100


def test_a_voice_with_no_preview_yet_is_a_404_not_an_empty_stream(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    voice = _registered(client, monkeypatch)
    assert client.get(f"/voice-clone/{voice['id']}/preview/audio").status_code == 404


# --- the registry -----------------------------------------------------------


def test_the_list_is_newest_first_and_omits_the_token_arrays(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_extract(monkeypatch)
    _stub_register(monkeypatch)
    first = _extract(client)
    client.post(f"/voice-clone/{first['id']}/register", json={"speakerId": "Mariam"})
    second = _extract(client)
    client.post(f"/voice-clone/{second['id']}/register", json={"speakerId": "Omar"})

    rows = client.get("/voice-clone").json()
    assert [r["speakerId"] for r in rows] == ["Omar", "Mariam"]
    assert all("globalTokenIds" not in r for r in rows)
    assert all(r["globalTokenCount"] == len(GLOBAL_IDS) for r in rows)


def test_delete_forgets_the_row_and_its_stored_clip(
    client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch, store: dict
) -> None:
    voice = _registered(client, monkeypatch)
    _stub_preview(monkeypatch, audio=_wav(1.0))
    client.post(f"/voice-clone/{voice['id']}/preview", json={"text": "hi"})
    assert store

    assert client.delete(f"/voice-clone/{voice['id']}").status_code == 204
    assert db_session.query(ClonedVoice).count() == 0
    assert not store


def test_deleting_an_unknown_voice_is_a_404(client: TestClient) -> None:
    assert client.delete("/voice-clone/9999").status_code == 404


def test_one_users_voices_are_not_another_users(
    client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every request is the seeded dev user (there is no real auth), so this
    pins the filter rather than an auth boundary: without `user_id` on the
    query, a second operator's catalogue would leak into this one's list and
    its clash check."""
    _stub_extract(monkeypatch)
    _extract(client)

    other = User(email="someone-else@example.com")
    db_session.add(other)
    db_session.flush()
    db_session.add(
        ClonedVoice(
            user_id=other.id,
            speaker_id="NotMine",
            status="registered",
            audio_url=CLIP_URL,
            prompt_text=PROMPT,
            dialect="msa",
        )
    )
    db_session.commit()

    rows = client.get("/voice-clone").json()
    assert [r["speakerId"] for r in rows] == [""]


# --- uploads ----------------------------------------------------------------


def test_uploads_are_refused_when_the_pod_cannot_reach_this_api(
    client: TestClient, store: dict
) -> None:
    """The pod downloads the reference itself. With no base URL it can resolve,
    returning a 127.0.0.1 URL would produce an opaque upstream 500 minutes
    later instead of a straight answer now."""
    response = client.post(
        "/voice-clone/reference", files={"file": ("ref.wav", _wav(1.0), "audio/wav")}
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "VOICE_CLONE_PUBLIC_BASE_URL" in detail


def test_an_upload_returns_a_url_built_from_the_configured_base(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, store: dict
) -> None:
    monkeypatch.setattr(get_settings(), "voice_clone_public_base_url", "https://evals.example.ae")
    response = client.post(
        "/voice-clone/reference", files={"file": ("ref.wav", _wav(2.0), "audio/wav")}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["audioUrl"].startswith("https://evals.example.ae/voice-clone/reference/")
    # Probed, not assumed: the values come from the real bytes that were posted.
    assert body["audioSec"] == pytest.approx(2.0, abs=0.05)
    assert body["nativeSampleRate"] == 16000
    assert body["channels"] == 1
    assert store


def test_an_empty_upload_is_refused(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, store: dict
) -> None:
    monkeypatch.setattr(get_settings(), "voice_clone_public_base_url", "https://evals.example.ae")
    response = client.post("/voice-clone/reference", files={"file": ("ref.wav", b"", "audio/wav")})
    assert response.status_code == 422
