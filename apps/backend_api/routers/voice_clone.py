"""The Clone sub-mode: register a speaker on the TTS pod from one reference
clip, then synthesize with it.

A third sub-mode of the Transcript surface, NOT a third surface. `AudioFile.surface`
stays two-valued for the reason CLAUDE.md already gives about TTS, and this goes
further: a cloned voice has no `AudioFile` at all. It is not a recording, it is
not scored, and it belongs to no project -- it is a speaker name the `hamsa-tts`
engine can use afterwards on any recording. Hence its own table and its own
top-level prefix rather than a route hanging off `/evaluations/{id}`.

Three steps, matching the three things the pod actually does:

    POST /voice-clone/extract          -> /tts/voice_clone       (tokens)
    POST /voice-clone/{id}/register    -> /tts/load_voice_cloning (a name)
    POST /voice-clone/{id}/preview     -> /tts/stream            (hear it)

Split into separate routes rather than one "clone this" call because the two
pod calls fail independently and the operator has a real decision between them:
extraction is the expensive part, and having tokens in hand is exactly when you
want to choose a name and a dialect rather than having guessed beforehand.

Not queued through RQ, same reasoning as `tts.py`: the flow is interactive and
short. Each route closes its request-scoped session before the pod call, since
extraction can outlast the 60s `idle_in_transaction_session_timeout`.
"""

import io
import logging
import re
from collections.abc import Generator
from datetime import datetime, timezone
from functools import partial

import httpx
from fastapi import APIRouter, Body, Depends, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session, undefer
from starlette.concurrency import run_in_threadpool

from apps.background_worker.tts import engine_for
from apps.background_worker.voice_clone import adapter as clone_adapter
from apps.background_worker.voice_clone import runner as clone_runner
from apps.backend_api.dependencies import get_current_user, get_db
from apps.backend_api.routers.evaluations import _iter_s3_object, _parse_range
from packages.audio import probe_audio
from packages.config.settings import get_settings
from packages.database.models import ClonedVoice, User
from packages.database.session import SessionLocal
from packages.shared_contracts.schemas import ClonePreview, ClonedVoice as ClonedVoiceContract
from packages.storage import s3_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/voice-clone", tags=["voice-clone"])

#: The engine a cloned voice is usable through. Cloning registers a speaker on
#: the pod `hamsa-tts` synthesizes against, so a preview MUST go through that
#: same engine -- previewing through any other would be synthesizing with a
#: voice that engine has never heard of.
PREVIEW_TTS_ID = "hamsa-tts"


def _contract(row: ClonedVoice) -> ClonedVoiceContract:
    """One row as the wire contract. Never touches the deferred token columns."""
    returned = None
    if isinstance(row.raw_output, dict):
        extract_meta = row.raw_output.get("extract")
        if isinstance(extract_meta, dict):
            value = extract_meta.get("returned_prompt_text")
            returned = value if isinstance(value, str) else None
    return ClonedVoiceContract(
        id=row.id,
        speaker_id=row.speaker_id,
        status=row.status,
        error=row.error,
        dialect=row.dialect,
        prompt_text=row.prompt_text,
        audio_url=row.audio_url,
        has_stored_clip=bool(row.s3_key),
        audio_format=row.audio_format,
        audio_sec=row.audio_sec,
        size_bytes=row.size_bytes,
        native_sample_rate=row.native_sample_rate,
        channels=row.channels,
        global_token_count=row.global_token_count,
        semantic_token_count=row.semantic_token_count,
        returned_prompt_text=returned,
        extract_ms=row.extract_ms,
        register_ms=row.register_ms,
        created_at=row.created_at,
        registered_at=row.registered_at,
    )


def _require_configured() -> None:
    if not get_settings().voice_clone_configured:
        raise HTTPException(
            status_code=422,
            detail=(
                "Voice cloning is not configured on this host — set "
                "HAMSA_VOICE_CLONE_URL, HAMSA_LOAD_VOICE_URL, HAMSA_TTS_KEY and "
                "HAMSA_TTS_BEARER_TOKEN in .env"
            ),
        )


def _get_voice(db: Session, voice_id: int, user: User) -> ClonedVoice:
    row = db.query(ClonedVoice).filter_by(id=voice_id, user_id=user.id).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"No cloned voice {voice_id}")
    return row


def _reference_key(voice_id: int, extension: str) -> str:
    """Object-store key for a mirrored reference clip.

    Slugified for the same reason `tts._clip_key` is: a key is a path, and
    nothing a caller supplies may decide the shape of the object store.
    """
    slug = re.sub(r"[^a-z0-9]+", "", extension.lower())[:8] or "bin"
    return f"voice-clone/{voice_id}/reference.{slug}"


def _preview_key(voice_id: int, extension: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "", extension.lower())[:8] or "bin"
    return f"voice-clone/{voice_id}/preview.{slug}"


@router.get("", response_model=list[ClonedVoiceContract], response_model_by_alias=True)
def list_cloned_voices(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[ClonedVoiceContract]:
    """Every voice this deployment has cloned, newest first.

    This is OUR record, not the pod's. The pod exposes no route that lists what
    it currently holds, and it holds registrations in memory, so a voice listed
    here as "registered" may have been dropped by a pod restart. Nothing in this
    repo may present this list as the pod's live state -- the UI says so.
    """
    rows = (
        db.query(ClonedVoice)
        .filter_by(user_id=current_user.id)
        # id desc is the TIEBREAK, not decoration: two voices cloned inside one
        # clock tick would otherwise come back in insertion order, and the UI
        # opens on the first row it sees.
        .order_by(ClonedVoice.created_at.desc(), ClonedVoice.id.desc())
        .all()
    )
    return [_contract(row) for row in rows]


@router.post("/reference", response_model=dict)
async def upload_reference(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Store an uploaded reference clip and hand back the URL to clone from.

    The pod fetches `audio_url` ITSELF, from wherever the pod runs, so an
    uploaded clip is only clonable when this API is reachable from the pod. That
    is what `VOICE_CLONE_PUBLIC_BASE_URL` declares; with it unset this route
    refuses rather than returning a `127.0.0.1` URL that resolves nowhere from
    the pod's side and comes back as an opaque upstream 500.
    """
    settings = get_settings()
    base = settings.voice_clone_public_base_url
    if not base:
        raise HTTPException(
            status_code=422,
            detail=(
                "Uploads cannot be cloned from on this host: the pod downloads the "
                "reference clip itself and cannot reach this API. Set "
                "VOICE_CLONE_PUBLIC_BASE_URL to a base URL the pod can resolve, or "
                "paste an already-public clip URL instead."
            ),
        )

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=422, detail="The uploaded file is empty")

    probe = await run_in_threadpool(probe_audio, payload)
    # Keyed by upload, not by voice: no row exists yet at this point. The
    # extract call that follows records which URL it was actually given.
    key = f"voice-clone/uploads/{current_user.id}/{datetime.now(timezone.utc):%Y%m%d%H%M%S%f}"
    await run_in_threadpool(s3_client.put_stream, io.BytesIO(payload), key)
    return {
        "audioUrl": f"{base.rstrip('/')}/voice-clone/reference/{key.split('/', 2)[2]}",
        "sizeBytes": len(payload),
        "audioSec": probe.duration_sec,
        "nativeSampleRate": probe.sample_rate,
        "channels": probe.channels,
    }


@router.get("/reference/{path:path}")
def serve_reference(
    path: str,
    range_header: str | None = Header(default=None, alias="Range"),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Serve an uploaded reference clip back, so the POD can download it.

    Deliberately Range-capable and streamed through the API, the same proxy
    pattern `stream_audio` uses: the browser never talks to MinIO directly, and
    neither does the pod.
    """
    key = f"voice-clone/uploads/{current_user.id}/{path}"
    try:
        total = s3_client.head_object(key)
    except Exception as exc:  # noqa: BLE001 - a missing object is a 404, not a 500
        raise HTTPException(status_code=404, detail="No such reference clip") from exc

    def _stream(start: int, length: int | None) -> Generator[bytes, None, None]:
        yield from _iter_s3_object(key, start=start, length=length)

    headers = {"Accept-Ranges": "bytes"}
    byte_range = _parse_range(range_header, total)
    if byte_range is None:
        headers["Content-Length"] = str(total)
        return StreamingResponse(_stream(0, None), media_type="audio/wav", headers=headers)
    start, end = byte_range
    headers["Content-Range"] = f"bytes {start}-{end}/{total}"
    headers["Content-Length"] = str(end - start + 1)
    return StreamingResponse(
        _stream(start, end - start + 1), status_code=206, media_type="audio/wav", headers=headers
    )


def _write_extraction(
    user_id: int,
    audio_url: str,
    prompt_text: str,
    dialect: str,
    tokens: clone_adapter.VoiceTokens | None,
    error: str | None,
) -> ClonedVoice:
    """Persist one extraction attempt, successful or not.

    A failed extraction still writes a row -- the same rule every other result
    table here follows. Without one, "never tried" and "the pod rejected this
    clip" are indistinguishable, and the opaque 500 this pod actually returns is
    precisely the failure an operator needs kept rather than dismissed.

    Opens its own session: the caller closed the request-scoped one before a pod
    call that can take minutes.
    """
    db = SessionLocal()
    try:
        row = ClonedVoice(
            user_id=user_id,
            # No name yet. Registration is what names a voice, and inventing a
            # placeholder here would put a speaker_id in the table that the pod
            # has never heard of.
            speaker_id="",
            status="failed" if tokens is None else "extracted",
            error=error,
            audio_url=audio_url,
            prompt_text=prompt_text,
            dialect=dialect,
        )
        if tokens is not None:
            global_count, semantic_count = clone_adapter.token_counts(
                tokens.global_token_ids, tokens.semantic_token_ids
            )
            row.global_token_ids = tokens.global_token_ids
            row.semantic_token_ids = tokens.semantic_token_ids
            row.global_token_count = global_count
            row.semantic_token_count = semantic_count
            row.extract_ms = tokens.extract_ms
            row.raw_output = {
                "extract": {**tokens.raw_meta, "returned_prompt_text": tokens.prompt_text}
            }
        db.add(row)
        db.commit()
        db.refresh(row)
        return _contract(row)
    finally:
        db.close()


@router.post("/extract", response_model=ClonedVoiceContract, response_model_by_alias=True)
async def extract_voice(
    audio_url: str = Body(..., embed=True, alias="audioUrl"),
    prompt_text: str = Body(..., embed=True, alias="promptText"),
    dialect: str | None = Body(None, embed=True),
    current_user: User = Depends(get_current_user),
) -> ClonedVoiceContract:
    """Step 1: turn a reference clip into voice tokens.

    The pod downloads `audio_url` itself. A URL that resolves from this host
    proves nothing about whether the pod can reach it, so no reachability check
    is performed here -- a probe from the wrong side of the network would be a
    fabricated reassurance. The pod's own failure is what gets recorded.
    """
    _require_configured()
    settings = get_settings()

    prompt = prompt_text.strip()
    if not prompt:
        raise HTTPException(
            status_code=422,
            detail="prompt_text is required — it must be the verbatim transcript of the clip",
        )
    if len(prompt) > settings.voice_clone_max_prompt_chars:
        raise HTTPException(
            status_code=422,
            detail=(
                f"prompt_text is {len(prompt)} chars, over this host's "
                f"{settings.voice_clone_max_prompt_chars}-char limit"
            ),
        )
    if not audio_url.strip():
        raise HTTPException(status_code=422, detail="audio_url is required")

    resolved_dialect = dialect or settings.voice_clone_default_dialect
    if resolved_dialect not in settings.voice_clone_dialect_options:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{resolved_dialect!r} is not one of this host's dialects: "
                f"{', '.join(settings.voice_clone_dialect_options)}"
            ),
        )

    try:
        raw = await run_in_threadpool(clone_runner.extract, audio_url.strip(), prompt)
        tokens = await run_in_threadpool(clone_adapter.adapt_extraction, raw)
    except Exception as exc:  # noqa: BLE001 - a rejected clip still writes a row
        logger.warning("Voice-clone extraction failed for %r", audio_url, exc_info=True)
        return await run_in_threadpool(
            partial(
                _write_extraction,
                current_user.id,
                audio_url.strip(),
                prompt,
                resolved_dialect,
                None,
                str(exc)[:2048],
            )
        )

    logger.info(
        "Extracted voice tokens from %r (%d global, %d semantic)",
        audio_url,
        *clone_adapter.token_counts(tokens.global_token_ids, tokens.semantic_token_ids),
    )
    return await run_in_threadpool(
        partial(
            _write_extraction,
            current_user.id,
            audio_url.strip(),
            prompt,
            resolved_dialect,
            tokens,
            None,
        )
    )


def _write_registration(
    voice_id: int,
    speaker_id: str,
    dialect: str,
    register_ms: int | None,
    raw_meta: dict | None,
    error: str | None,
) -> ClonedVoiceContract:
    """Persist one registration attempt against an existing extraction.

    An UPDATE of the extraction row, never a second `db.add`. That is the same
    shape as the `finalize_session` and `tts_results` incidents this repo has
    already had twice: a blind add against a unique index 500s every call, and
    here it would additionally orphan the tokens on the first row.
    """
    db = SessionLocal()
    try:
        row = db.query(ClonedVoice).filter_by(id=voice_id).one()
        row.speaker_id = speaker_id
        row.dialect = dialect
        if error is not None:
            # The tokens survive a failed registration: they are still valid and
            # retrying must not require paying for extraction again.
            row.status = "extracted"
            row.error = error
        else:
            row.status = "registered"
            row.error = None
            row.register_ms = register_ms
            row.registered_at = datetime.now(timezone.utc)
        merged = dict(row.raw_output or {})
        merged["register"] = raw_meta
        row.raw_output = merged
        db.commit()
        db.refresh(row)
        return _contract(row)
    finally:
        db.close()


@router.post("/{voice_id}/register", response_model=ClonedVoiceContract, response_model_by_alias=True)
async def register_voice(
    voice_id: int,
    speaker_id: str = Body(..., embed=True, alias="speakerId"),
    dialect: str | None = Body(None, embed=True),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ClonedVoiceContract:
    """Step 2: file the extracted tokens under a speaker name.

    The tokens are re-posted EXACTLY as extraction returned them, which is why
    they were stored in the first place: this route never re-extracts, so a
    registration can be repeated after a pod restart drops it without the
    reference clip needing to still exist.
    """
    _require_configured()
    settings = get_settings()

    name = speaker_id.strip()
    if not name:
        raise HTTPException(status_code=422, detail="speaker_id is required")

    # Undefer explicitly: the token columns are deferred so the list route never
    # loads them, which means they are exactly the thing this route must ask for.
    row = (
        db.query(ClonedVoice)
        .options(undefer(ClonedVoice.global_token_ids), undefer(ClonedVoice.semantic_token_ids))
        .filter_by(id=voice_id, user_id=current_user.id)
        .one_or_none()
    )
    if row is None:
        raise HTTPException(status_code=404, detail=f"No cloned voice {voice_id}")
    if row.global_token_ids is None or row.semantic_token_ids is None:
        raise HTTPException(
            status_code=422,
            detail="This voice has no extracted tokens — extraction failed, so run it again",
        )

    resolved_dialect = dialect or row.dialect or settings.voice_clone_default_dialect
    if resolved_dialect not in settings.voice_clone_dialect_options:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{resolved_dialect!r} is not one of this host's dialects: "
                f"{', '.join(settings.voice_clone_dialect_options)}"
            ),
        )

    # Our own catalogue's uniqueness, checked before the pod call so the clash is
    # reported as a clash rather than as a database error after the pod has
    # already overwritten the voice in its memory.
    clash = (
        db.query(ClonedVoice)
        .filter(
            ClonedVoice.user_id == current_user.id,
            ClonedVoice.speaker_id == name,
            ClonedVoice.id != row.id,
        )
        .one_or_none()
    )
    if clash is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"{name!r} is already registered here (voice {clash.id}). The pod would "
                "overwrite it in memory; delete that one first if that is what you want."
            ),
        )

    global_tokens = row.global_token_ids
    semantic_tokens = row.semantic_token_ids
    prompt = row.prompt_text
    db.close()

    try:
        raw = await run_in_threadpool(
            clone_runner.register, name, global_tokens, semantic_tokens, resolved_dialect, prompt
        )
        outcome = await run_in_threadpool(clone_adapter.adapt_registration, raw)
    except Exception as exc:  # noqa: BLE001 - a rejected registration keeps its tokens
        logger.warning("Voice registration failed for %r", name, exc_info=True)
        return await run_in_threadpool(
            partial(
                _write_registration, voice_id, name, resolved_dialect, None, None, str(exc)[:2048]
            )
        )

    logger.info("Registered cloned voice %r (dialect %s)", name, resolved_dialect)
    return await run_in_threadpool(
        partial(
            _write_registration,
            voice_id,
            name,
            resolved_dialect,
            outcome.register_ms,
            outcome.raw_meta,
            None,
        )
    )


@router.delete("/{voice_id}", status_code=204)
def delete_cloned_voice(
    voice_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    """Forget a voice locally.

    Local only, and the UI says so: the pod has no delete route (its OpenAPI
    lists none), so a registered speaker stays resident there until the pod
    restarts. Claiming otherwise would be the one thing worse than not offering
    delete at all.
    """
    row = _get_voice(db, voice_id, current_user)
    for key in (row.s3_key,):
        if key:
            try:
                s3_client.delete_object(key)
            except Exception:  # noqa: BLE001 - a missing object must not block the row delete
                logger.warning("Could not delete stored clip %s", key, exc_info=True)
    db.delete(row)
    db.commit()


def _write_preview(voice_id: int, key: str, audio_format: str, probe, render) -> None:
    """Record where a preview clip landed, so the audio route can serve it."""
    db = SessionLocal()
    try:
        row = db.query(ClonedVoice).filter_by(id=voice_id).one()
        row.s3_key = key
        row.audio_format = audio_format
        row.audio_sec = probe.duration_sec
        row.size_bytes = len(render.audio)
        row.native_sample_rate = probe.sample_rate
        row.channels = probe.channels
        db.commit()
    finally:
        db.close()


@router.post("/{voice_id}/preview", response_model=ClonePreview, response_model_by_alias=True)
async def preview_voice(
    voice_id: int,
    text: str = Body(..., embed=True),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ClonePreview:
    """Step 3: synthesize something with the cloned voice, to hear whether it worked.

    Goes through the `hamsa-tts` engine, unchanged and unwrapped. That is the
    whole point of cloning: the registered name is usable exactly like a
    built-in voice, so a preview that used some special code path would prove
    nothing about the thing being tested.

    Nothing here scores the clip against the reference. No similarity
    measurement exists in this repo and none is invented -- the honest artifact
    is A/B playback, which is what the UI renders.
    """
    settings = get_settings()
    engine = engine_for(PREVIEW_TTS_ID)
    if engine is None or not engine.configured(settings):
        raise HTTPException(
            status_code=422,
            detail=f"{PREVIEW_TTS_ID!r} is not configured on this host, so a clone cannot be previewed",
        )

    row = _get_voice(db, voice_id, current_user)
    if row.status != "registered":
        raise HTTPException(
            status_code=422,
            detail=f"{row.speaker_id or 'This voice'} is not registered on the pod yet (status {row.status!r})",
        )
    body = text.strip()
    if not body:
        raise HTTPException(status_code=422, detail="text is required")
    if len(body) > settings.tts_max_input_chars:
        raise HTTPException(
            status_code=422,
            detail=f"text is {len(body)} chars, over this host's {settings.tts_max_input_chars}-char limit",
        )
    speaker = row.speaker_id
    db.close()

    try:
        raw = await run_in_threadpool(engine.run, body, speaker)
        render = await run_in_threadpool(engine.adapt, raw)
    except Exception as exc:  # noqa: BLE001 - reported to the caller, no row to poison
        # A pod that dropped its voices on restart fails HERE, with the speaker
        # name in the message, which is the signal this repo otherwise has no
        # way to get: the pod exposes nothing that lists what it holds.
        raise HTTPException(
            status_code=502,
            detail=f"Synthesizing with {speaker!r} failed: {exc}",
        ) from exc

    probe = await run_in_threadpool(probe_audio, render.audio)
    key = _preview_key(voice_id, render.audio_format)
    await run_in_threadpool(s3_client.put_stream, io.BytesIO(render.audio), key)
    await run_in_threadpool(partial(_write_preview, voice_id, key, render.audio_format, probe, render))

    return ClonePreview(
        voice_id=voice_id,
        speaker_id=speaker,
        text=body,
        audio_format=render.audio_format,
        audio_sec=probe.duration_sec,
        size_bytes=len(render.audio),
        native_sample_rate=probe.sample_rate,
        first_audio_ms=render.first_audio_ms,
        synth_ms=render.synth_ms,
    )


@router.get("/{voice_id}/preview/audio")
def stream_preview_audio(
    voice_id: int,
    download: int = Query(0, description="1 to add a Content-Disposition: attachment header"),
    range_header: str | None = Header(default=None, alias="Range"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Stream the stored preview clip, Range-capable, through the API.

    `media_type` comes from the row's own `audio_format` rather than a hardcoded
    `audio/wav`, the same rule `stream_tts_audio` follows.
    """
    row = _get_voice(db, voice_id, current_user)
    if not row.s3_key:
        raise HTTPException(status_code=404, detail=f"{row.speaker_id!r} has no preview clip yet")
    key = row.s3_key
    audio_format = row.audio_format or "wav"
    speaker = row.speaker_id
    db.close()

    media_type = "audio/mpeg" if audio_format == "mp3" else "audio/wav"
    total = s3_client.head_object(key)

    def _stream(start: int, length: int | None) -> Generator[bytes, None, None]:
        yield from _iter_s3_object(key, start=start, length=length)

    headers = {"Accept-Ranges": "bytes"}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{speaker}.{audio_format}"'

    byte_range = _parse_range(range_header, total)
    if byte_range is None:
        headers["Content-Length"] = str(total)
        return StreamingResponse(_stream(0, None), media_type=media_type, headers=headers)
    start, end = byte_range
    headers["Content-Range"] = f"bytes {start}-{end}/{total}"
    headers["Content-Length"] = str(end - start + 1)
    return StreamingResponse(
        _stream(start, end - start + 1), status_code=206, media_type=media_type, headers=headers
    )
