"""The TTS comparison surface: synthesize a recording's reference text with
each TTS engine and compare the clips.

Hangs off the SAME `AudioFile`/`TranscriptReference` row `POST /transcript/script`
already creates -- there is no separate TTS "recording" and no script-generation
route here. One script can be both read aloud (the transcript surface) and
synthesized (this one), scored against the same ground truth.

Not queued through RQ: the flow is interactive and short (press a button, hear
a clip in seconds). `perf_counter` inside each engine's runner measures the
real cost regardless of whether a queue sits in front of it or not -- the
reason to skip RQ here is responsiveness, not measurement purity.

One engine per request, always -- never both in one call, so a hamsa-tts
failure can never take down inception-tts's result. The frontend fires two
requests in parallel.
"""

import io
import logging
import re
from collections.abc import Generator
from functools import partial

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from apps.background_worker.tts import engine_for
from apps.backend_api.dependencies import get_current_user, get_db
from apps.backend_api.routers.evaluations import _get_audio_file, _iter_s3_object, _parse_range
from packages.audio import probe_audio
from packages.config.settings import get_settings
from packages.database.models import TranscriptReference, TtsResult, User
from packages.database.session import SessionLocal
from packages.shared_contracts.schemas import TtsRawOutput, TtsRun
from packages.storage import s3_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evaluations", tags=["tts"])


def _tts_run(row: TtsResult) -> TtsRun:
    engine = engine_for(row.tts_id)
    return TtsRun(
        audio_file_id=row.audio_file_id,
        tts_id=row.tts_id,
        tts_name=engine.name if engine else row.tts_id,
        status=row.status,
        error=row.error,
        voice=row.voice,
        delivery=row.delivery,
        text_chars=row.text_chars,
        audio_format=row.audio_format,
        size_bytes=row.size_bytes,
        native_sample_rate=row.native_sample_rate,
        audio_sec=row.audio_sec,
        channels=row.channels,
        bit_depth=row.bit_depth,
        first_audio_ms=row.first_audio_ms,
        synth_ms=row.synth_ms,
        rtf=row.rtf,
    )


def _clip_key(audio_file_id: int, tts_id: str, voice: str, extension: str) -> str:
    """Object-store key for one clip.

    The VOICE is part of the key, not decoration: two voices of one engine are
    two separate stored takes, and a shared key would have the second silently
    overwrite the first's bytes while both rows still pointed at it.

    Slugified because a key is a path and voices come from .env -- whatever
    someone types there must not decide the shape of the object store.
    """
    slug = re.sub(r"[^a-z0-9._-]+", "-", voice.lower()).strip("-") or "voice"
    return f"tts/{audio_file_id}/{tts_id}/{slug}.{extension}"


def _get_or_reset_tts_result(db: Session, audio_file_id: int, tts_id: str, voice: str) -> TtsResult:
    """Fetch this (recording, engine, VOICE)'s row, or create one -- never a
    blind `db.add`. Re-running the same voice is the most natural second action
    in this UI, and a blind add would trip the unique constraint and 500,
    exactly the shape of the `finalize_session` incident this repo already paid
    for (see CLAUDE.md).

    Keyed on the voice as well, so synthesizing a DIFFERENT voice adds a row
    instead of overwriting the last one -- that is the whole point of the
    per-voice constraint.
    """
    row = (
        db.query(TtsResult)
        .filter_by(audio_file_id=audio_file_id, tts_id=tts_id, voice=voice)
        .one_or_none()
    )
    if row is None:
        row = TtsResult(audio_file_id=audio_file_id, tts_id=tts_id, voice=voice)
        db.add(row)
    return row


def _resolve_clip(db: Session, audio_file_id: int, tts_id: str, voice: str | None) -> TtsResult:
    """The one row a read route should serve, or an HTTPException saying why not.

    `voice` omitted is allowed only while it is UNAMBIGUOUS: with a single clip
    for that engine there is nothing to choose between, so callers that predate
    per-voice storage keep working. With several, the route names the voices
    rather than picking one -- handing back an arbitrary take would be a wrong
    answer that looks like a right one.
    """
    rows = (
        db.query(TtsResult)
        .filter_by(audio_file_id=audio_file_id, tts_id=tts_id)
        .order_by(TtsResult.created_at.desc())
        .all()
    )
    if not rows:
        raise HTTPException(
            status_code=404, detail=f"No synthesized audio for {tts_id!r} on this recording"
        )
    if voice is not None:
        for row in rows:
            if row.voice == voice:
                return row
        available = ", ".join(sorted(r.voice for r in rows))
        raise HTTPException(
            status_code=404,
            detail=f"{tts_id!r} has no clip for voice {voice!r} on this recording (have: {available})",
        )
    if len(rows) > 1:
        available = ", ".join(sorted(r.voice for r in rows))
        raise HTTPException(
            status_code=400,
            detail=f"{tts_id!r} has clips for several voices ({available}); add ?voice=",
        )
    return rows[0]


def _write_failure(
    audio_file_id: int, tts_id: str, error: str, voice: str, delivery: str, text_chars: int
) -> TtsRun:
    """Record that this engine was tried and failed.

    A failed engine still gets a row: without one, "never tried" and "failed"
    are indistinguishable to the operator. Every measured field is cleared, so
    a previous take's numbers can never sit beside a failure.

    Synchronous, and called via run_in_threadpool: SessionLocal() is a pool
    checkout and `pool_pre_ping=True` makes that a database round trip, so even
    opening the session would block the event loop.
    """
    db = SessionLocal()
    try:
        row = _get_or_reset_tts_result(db, audio_file_id, tts_id, voice)
        row.status = "failed"
        row.error = error
        row.delivery = delivery
        row.text_chars = text_chars
        row.s3_key = None
        row.audio_format = None
        row.size_bytes = None
        row.native_sample_rate = None
        row.audio_sec = None
        row.channels = None
        row.bit_depth = None
        row.first_audio_ms = None
        row.synth_ms = None
        row.rtf = None
        row.raw_output = None
        db.commit()
        return _tts_run(row)
    finally:
        db.close()


def _write_success(
    audio_file_id: int, tts_id: str, key: str, render, probe, rtf: float | None,
    voice: str, delivery: str, text_chars: int,
) -> TtsRun:
    """Store one completed synthesis. Off the event loop, see _write_failure."""
    db = SessionLocal()
    try:
        row = _get_or_reset_tts_result(db, audio_file_id, tts_id, voice)
        row.status = "done"
        row.error = None
        row.delivery = delivery
        row.text_chars = text_chars
        row.s3_key = key
        row.audio_format = render.audio_format
        row.size_bytes = len(render.audio)
        row.native_sample_rate = probe.sample_rate
        row.audio_sec = probe.duration_sec
        row.channels = probe.channels
        row.bit_depth = probe.bit_depth
        row.first_audio_ms = render.first_audio_ms
        row.synth_ms = render.synth_ms
        row.rtf = rtf
        row.raw_output = render.raw_meta
        db.commit()
        return _tts_run(row)
    finally:
        db.close()


@router.post("/{audio_file_id}/tts/{tts_id}", response_model=TtsRun, response_model_by_alias=True)
async def synthesize_tts(
    audio_file_id: int,
    tts_id: str,
    voice: str | None = Body(None, embed=True),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TtsRun:
    """Synthesize this recording's reference text with one TTS engine.

    A failed engine still gets a row here (`status="failed"`, `error` set),
    never a 500 and never no row at all -- without one, "never tried" and
    "failed" would look identical to the operator.

    The request-scoped session is closed before the engine call and a fresh one
    opened for the write: `packages/database/session.py` kills an idle
    in-transaction connection after 60s, and synthesis can take several seconds.
    Holding the request's session open across it risks the failure surfacing
    later as an unrelated error on whatever runs next.
    """
    settings = get_settings()
    engine = engine_for(tts_id)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"Unknown TTS engine {tts_id!r}")
    if not engine.configured(settings):
        raise HTTPException(status_code=422, detail=f"{tts_id!r} is not configured on this host")

    audio_file = _get_audio_file(db, audio_file_id, current_user)
    reference = (
        db.query(TranscriptReference).filter_by(audio_file_id=audio_file.id).one_or_none()
    )
    if reference is None:
        raise HTTPException(status_code=422, detail="Generate a script first")
    text = reference.text
    if len(text) > settings.tts_max_input_chars:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Reference text is {len(text)} chars, over this host's "
                f"{settings.tts_max_input_chars}-char limit"
            ),
        )

    resolved_voice = voice or engine.default_voice(settings)
    delivery = engine.delivery
    text_chars = len(text)
    db.close()

    try:
        # run_in_threadpool, not a bare call: engine.run/engine.adapt make real
        # HTTP calls and this handler is `async def` -- see transcript.py's
        # transcribe_chunk for the measured cost of getting this wrong.
        raw = await run_in_threadpool(engine.run, text, resolved_voice)
        render = await run_in_threadpool(engine.adapt, raw)
    except Exception as exc:  # noqa: BLE001 - a failed engine still writes a row
        logger.warning("TTS synthesis failed for %r on audio_file_id=%s", tts_id, audio_file_id, exc_info=True)
        return await run_in_threadpool(
            partial(_write_failure, audio_file_id, tts_id, str(exc)[:2048], resolved_voice, delivery, text_chars)
        )

    probe = await run_in_threadpool(probe_audio, render.audio)
    rtf = (render.synth_ms / 1000.0) / probe.duration_sec if probe.duration_sec else None
    key = _clip_key(audio_file_id, tts_id, resolved_voice, render.audio_format)
    await run_in_threadpool(s3_client.put_stream, io.BytesIO(render.audio), key)

    result = await run_in_threadpool(
        partial(_write_success, audio_file_id, tts_id, key, render, probe, rtf, resolved_voice, delivery, text_chars)
    )
    logger.info("Synthesized %r for audio_file_id=%s (%d chars)", tts_id, audio_file_id, text_chars)
    return result


@router.get("/{audio_file_id}/tts", response_model=list[TtsRun], response_model_by_alias=True)
def get_tts_runs(
    audio_file_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[TtsRun]:
    """Every clip stored for this recording: one per (engine, voice), so an
    engine appears once per voice that has been synthesized. An empty list
    means nothing has been synthesized yet.

    Ordered newest-first WITHIN an engine, and that ordering is behaviour, not
    cosmetics: the UI opens a saved project on the first clip it sees for each
    engine, so this is what decides which voice you land on.
    """
    audio_file = _get_audio_file(db, audio_file_id, current_user)
    rows = (
        db.query(TtsResult)
        .filter_by(audio_file_id=audio_file.id)
        .order_by(TtsResult.tts_id, TtsResult.created_at.desc())
        .all()
    )
    return [_tts_run(row) for row in rows]


@router.get("/{audio_file_id}/tts/{tts_id}/audio")
def stream_tts_audio(
    audio_file_id: int,
    tts_id: str,
    download: int = Query(0, description="1 to add a Content-Disposition: attachment header"),
    voice: str | None = Query(None, description="Which voice's clip; required once an engine has several"),
    range_header: str | None = Header(default=None, alias="Range"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Stream one engine's synthesized clip through the API, Range-capable,
    mirroring `stream_audio`'s proxy pattern.

    `media_type` comes from the row's own `audio_format`, never hardcoded --
    inception-tts can render mp3, and labelling it `audio/wav` would mislabel
    the clip the browser plays.
    """
    audio_file = _get_audio_file(db, audio_file_id, current_user)
    row = _resolve_clip(db, audio_file.id, tts_id, voice)
    if not row.s3_key:
        raise HTTPException(
            status_code=404,
            detail=f"{tts_id!r}'s {row.voice!r} run stored no audio (status {row.status!r})",
        )
    s3_key = row.s3_key
    audio_format = row.audio_format or "wav"
    row_voice = row.voice
    db.close()

    media_type = "audio/mpeg" if audio_format == "mp3" else "audio/wav"
    total = s3_client.head_object(s3_key)

    def _stream(start: int, length: int | None) -> Generator[bytes, None, None]:
        yield from _iter_s3_object(s3_key, start=start, length=length)

    headers = {"Accept-Ranges": "bytes"}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{tts_id}-{row_voice}.{audio_format}"'

    byte_range = _parse_range(range_header, total)
    if byte_range is None:
        headers["Content-Length"] = str(total)
        return StreamingResponse(_stream(0, None), media_type=media_type, headers=headers)

    start, end = byte_range
    headers["Content-Range"] = f"bytes {start}-{end}/{total}"
    headers["Content-Length"] = str(end - start + 1)
    return StreamingResponse(_stream(start, end - start + 1), status_code=206, media_type=media_type, headers=headers)


@router.get("/{audio_file_id}/tts/{tts_id}/raw", response_model=TtsRawOutput, response_model_by_alias=True)
def get_tts_raw_output(
    audio_file_id: int,
    tts_id: str,
    voice: str | None = Query(None, description="Which voice's clip; required once an engine has several"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TtsRawOutput:
    """One TTS engine's response metadata for this recording, beside the
    adapted contract. `raw_output` here is status code/headers, never the
    audio bytes -- the audio is served by `.../tts/{ttsId}/audio`."""
    audio_file = _get_audio_file(db, audio_file_id, current_user)
    if engine_for(tts_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown TTS engine {tts_id!r}")

    row = _resolve_clip(db, audio_file.id, tts_id, voice)

    return TtsRawOutput(tts_id=tts_id, status=row.status, raw_output=row.raw_output, run=_tts_run(row))
