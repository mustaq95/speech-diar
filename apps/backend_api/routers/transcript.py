"""The transcript-evaluation surface: script generation and live read-aloud runs.

Separate from `evaluations.py` because nothing here is keyed on an existing
recording. A script is generated before any audio exists, and a live session
accumulates transcripts while the audio is still being spoken; only at finalize
does any of it become an `AudioFile` with `TranscriptResult` rows.

The live path deliberately does NOT go through RQ. Its calls are short and their
latency IS the measurement — putting a queue between the microphone and the
engine would add scheduling delay to a number the scorecard reports as the
engine's. Stored-audio runs, where latency is not being measured live, still go
through the queue via `evaluations.py`.

Two transports, because the two engines have genuinely different native ones and
matching them artificially would mean handicapping one:

  * `WS /transcript/live/{sessionId}/{asrId}` — one continuous engine session for
    the whole recording, for a streaming engine (TryHamsa). PCM in, frames out.
  * `POST /transcript/chunk` — one short WAV per call, for a request/response
    engine (Inception-STT), which cannot stream and drops content on anything
    longer (see the inception runner's measurements).
"""

import json
import logging
from datetime import datetime
from functools import partial

from contextlib import suppress

from starlette.concurrency import run_in_threadpool

from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from sqlalchemy.orm import Session

from apps.background_worker.transcription import (
    ASR_ENGINES,
    COMPARISON_ASR_IDS,
    feed_modes,
    live_session,
    resolve_asr_ids,
    resolve_feed_mode,
    transport_for,
    transport_for_mode,
)
from apps.background_worker.transcription import scoring
from apps.backend_api.dependencies import get_current_user, get_db
from apps.backend_api.routers.evaluations import (
    _queue_transcript,
    _transcript_run as transcript_run_contract,
)
from apps.background_worker.queue_app import queue
from apps.background_worker.transcription.pipeline import run_asr
from packages import script_gen
from packages.config.settings import get_settings
from apps.backend_api.routers.upload import attach_audio, store_recording
from packages.shared_contracts.schemas import (
    GeneratedScript,
    ScriptRequest,
    TranscriptReference as TranscriptReferenceContract,
    TranscriptRun,
)
from packages.database.models import (
    AudioFile,
    TranscriptReference,
    TranscriptResult,
    User,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/transcript", tags=["transcript"])


@router.post("/script", response_model=GeneratedScript, response_model_by_alias=True)
def generate_script(
    request: ScriptRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> GeneratedScript:
    """Generate a script to read aloud, which becomes the reference for scoring.

    The script is SAVED as it is generated: a transcript recording row plus its
    `TranscriptReference`, so it appears in Projects immediately and a script is
    never lost by regenerating or leaving the page. The row has no audio yet
    (`s3_key` null, `duration_sec` 0) and `finalize` attaches the recording to this
    same row, which is why the browser gets its id back.

    One row per generation, so every script generated is kept. A script that is
    never read aloud stays in the list as a script-only row until it is deleted;
    that is the cost of not losing any, and per-row delete already exists.

    503 when no gateway is configured, rather than a placeholder script — a
    fabricated reference would produce fabricated error rates.
    """
    settings = get_settings()
    if request.language_mix not in settings.script_language_mix_options:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown language mix {request.language_mix!r}; "
                   f"this host offers {settings.script_language_mix_options}",
        )
    unknown = [case for case in request.hard_cases if case not in settings.script_hard_case_options]
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown hard case(s): {unknown}")

    try:
        script = script_gen.generate(
            minutes=request.minutes,
            language_mix=request.language_mix,
            hard_cases=request.hard_cases,
            settings=settings,
        )
    except script_gen.ScriptGatewayUnconfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except script_gen.ScriptGatewayError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # `recording_` from the start, even though nothing has been recorded yet:
    # one naming scheme for every row on this surface. Whether audio exists is
    # already carried by `has_audio`, so the filename does not need to say it,
    # and a row that renamed itself later made the same artifact look like two.
    audio_file = AudioFile(
        owner_id=current_user.id,
        filename=f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        # Not a measured duration and not presented as one: `has_audio` is False
        # until a recording exists. The column is non-nullable, so 0 is the only
        # value available, and no route reports it while has_audio is False.
        duration_sec=0.0,
        surface="transcript",
    )
    db.add(audio_file)
    db.flush()
    db.add(
        TranscriptReference(
            audio_file_id=audio_file.id,
            source="script",
            text=script.text,
            params=script.params,
        )
    )
    db.commit()

    return GeneratedScript(
        text=script.text,
        word_count=script.word_count,
        generator_model=script.generator_model,
        params=script.params,
        audio_file_id=audio_file.id,
    )


@router.post(
    "/reference", response_model=TranscriptReferenceContract, response_model_by_alias=True,
    status_code=201,
)
def create_reference(
    text: str = Body(..., embed=True),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TranscriptReferenceContract:
    """Save a script the operator brought themselves, with no LLM involved.

    The same row shape `generate_script` writes -- an audio-less AudioFile plus
    its TranscriptReference -- so a pasted script is a first-class recording
    from the moment it exists: it appears in Projects, it can be read aloud on
    the STT side, and it can be synthesized on the TTS side. The only
    difference is `source="pasted"`, which is what tells the UI not to claim a
    generator model it never used.

    Needed because TTS synthesis reads its text from a stored reference, and
    until now the only way to get one was to generate it. `PUT
    /evaluations/{id}/reference` replaces the text of a row that already
    exists; this creates the row.
    """
    cleaned = text.strip()
    if not cleaned:
        raise HTTPException(status_code=422, detail="Reference text is empty")

    audio_file = AudioFile(
        owner_id=current_user.id,
        # Same `recording_` scheme as generate_script; see the note there.
        filename=f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        duration_sec=0.0,
        surface="transcript",
    )
    db.add(audio_file)
    db.flush()
    reference = TranscriptReference(
        audio_file_id=audio_file.id, source="pasted", text=cleaned, params=None
    )
    db.add(reference)
    db.commit()

    return TranscriptReferenceContract(
        audio_file_id=audio_file.id,
        source="pasted",
        # Measured here, never taken from the request -- same rule as every
        # other count this platform reports.
        text=cleaned,
        word_count=len(cleaned.split()),
        params=None,
    )


@router.post("/session")
def open_session(
    asr_ids: list[str] | None = Body(None, embed=True, alias="asrIds"),
    chunk_interval_sec: float | None = Body(None, embed=True, alias="chunkIntervalSec"),
    reference_text: str = Body("", embed=True, alias="referenceText"),
    modes: dict[str, str] | None = Body(None, embed=True, alias="modes"),
    current_user: User = Depends(get_current_user),
) -> dict[str, object]:
    """Open a live read-aloud session and return its id.

    The session exists so each engine's transcript accumulates NEXT TO the calls
    that produced it, server-side. The browser could accumulate its own text and
    post it at the end, but then the transcript would be a claim rather than a
    measurement — and the latencies would be the browser's, not the engine's.
    """
    settings = get_settings()
    try:
        resolved = resolve_asr_ids(asr_ids or list(COMPARISON_ASR_IDS))
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"Unknown ASR engine {exc.args[0]!r}") from exc

    unconfigured = [i for i in resolved if not ASR_ENGINES[i].configured(settings)]
    if unconfigured:
        raise HTTPException(
            status_code=422, detail=f"Not configured on this host: {unconfigured}"
        )

    # Resolved once, here, and stored on the session: an engine that offers more
    # than one feed mode must run the whole recording in the one picked at Start,
    # not in whatever the control says by the time finalize arrives.
    try:
        chosen = {
            asr_id: resolve_feed_mode(asr_id, (modes or {}).get(asr_id))
            for asr_id in resolved
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    interval = chunk_interval_sec or settings.live_chunk_default_sec
    if not settings.live_chunk_min_sec <= interval <= settings.live_chunk_max_sec:
        raise HTTPException(
            status_code=422,
            detail=f"chunkIntervalSec must be between {settings.live_chunk_min_sec} "
                   f"and {settings.live_chunk_max_sec}",
        )

    session_id = live_session.create(resolved, interval, reference_text, modes=chosen)
    return {
        "sessionId": session_id,
        "asrIds": resolved,
        "chunkIntervalSec": interval,
        "sampleRate": settings.live_record_sample_rate,
        "engines": [
            {
                "asrId": asr_id,
                "name": ASR_ENGINES[asr_id].name,
                # The mode this SESSION runs in, and the transport that mode
                # actually produces -- the browser keys its feeding on the pair,
                # so a mismatch here would feed an engine over a route its own
                # session refuses.
                "feedMode": chosen[asr_id],
                "feedModes": list(feed_modes(asr_id)),
                "transport": transport_for_mode(asr_id, chosen[asr_id]),
            }
            for asr_id in resolved
        ],
    }


@router.post("/chunk")
async def transcribe_chunk(
    session_id: str = Form(..., alias="sessionId"),
    asr_id: str = Form(..., alias="asrId"),
    chunk_index: int = Form(..., alias="chunkIndex"),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
) -> dict[str, object]:
    """Transcribe one chunk for a request/response engine, live.

    Persists nothing on its own: the text and its MEASURED latency go into the
    session, and only finalize turns those into rows. The latency returned here
    is the same value recorded, so the panel and the scorecard cannot disagree.

    A chunk that comes back empty is still recorded. This gateway does return
    blank text for some chunks, and dropping those would understate the chunk
    count while flattering the engine's apparent output rate.
    """
    if not live_session.exists(session_id):
        raise HTTPException(status_code=404, detail="Unknown or expired session")
    engine = ASR_ENGINES.get(asr_id)
    if engine is None:
        raise HTTPException(status_code=422, detail=f"Unknown ASR engine {asr_id!r}")
    # Dispatch on asr_id, never on a hardcoded runner. The result is filed under
    # asr_id a few lines below, so an engine mismatch here would store one
    # engine's text and latency under another's name at 200 OK -- invisible
    # except as two identical columns on the scorecard.
    if engine.run_chunk is None or engine.chunk_text is None:
        raise HTTPException(
            status_code=422,
            detail=f"{engine.name} has no chunk transport; it is not fed by this route",
        )
    # An engine that CAN be chunked but whose session runs it in batch must not be
    # fed here either: the chunk would be transcribed and stored, and the row would
    # carry live chunk timings while `run_asr` also wrote it from storage. The
    # session's pick is the authority, not the engine's capability.
    session_meta = live_session.meta(session_id) or {}
    mode = (session_meta.get("modes") or {}).get(asr_id)
    if mode and mode != "live":
        raise HTTPException(
            status_code=422,
            detail=f"{engine.name} is running in {mode!r} mode for this session; "
                   "it is not fed by this route",
        )

    payload = await file.read()
    try:
        # run_in_threadpool, NOT a bare call: the engine's chunk call blocks on
        # httpx.post for as long as the gateway takes, and this handler is
        # `async def`, so calling it directly blocks the whole event loop.
        #
        # That was not theoretical. Measured with three chunk calls in flight,
        # /health latency went from 4 ms to 1066 ms — and the casualties were the
        # things sharing the loop: the streaming engine's WebSocket relay was
        # starved (3 words captured over two minutes of speech) and finalize sat
        # behind every queued chunk, which is what made Stop look like it hung.
        entry = await run_in_threadpool(
            engine.run_chunk, payload, file.filename or "chunk.wav"
        )
    except Exception as exc:
        # Recorded as an empty chunk rather than swallowed: the call was made and
        # took time, so the chunk count stays truthful, and the error reaches the
        # panel instead of looking like silence.
        #
        # Broad, because each engine raises its own error type and this route no
        # longer knows which engine it just called. Re-raised as a 502 either
        # way, so nothing is being swallowed by the width.
        await run_in_threadpool(
            partial(live_session.append_part, session_id, asr_id,
                    index=chunk_index, text="", latency_ms=0, raw=None)
        )
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    text = engine.chunk_text(entry)
    latency_ms = int(entry.get("latency_ms") or 0)
    await run_in_threadpool(
        partial(live_session.append_part, session_id, asr_id,
                index=chunk_index, text=text, latency_ms=latency_ms,
                # The whole entry, so the gateway's own JSON survives under
                # "response" alongside the latency we measured.
                raw=entry)
    )
    return {"text": text, "latencyMs": latency_ms, "chunkIndex": chunk_index}


@router.websocket("/live/{session_id}/{asr_id}")
async def live_stream(websocket: WebSocket, session_id: str, asr_id: str) -> None:
    """Relay microphone PCM to a streaming engine and its text back, live.

    One engine session for the whole recording, which is TryHamsa's native mode:
    its VAD segments speech as the audio arrives, and cutting the audio into
    independent short sessions would destroy that context across every boundary.
    Giving it its native transport is the point — matching it to the chunked
    engine's shape would handicap it and call the result fairness.

    **Latency here is lag behind live, not per-call processing time.** A streaming
    protocol consumes audio at 1x by definition, so "how long did the call take"
    is not a question that has an answer. What is measurable, and what the panel's
    lag figure means, is: at the moment a piece of text arrived, how far behind the
    speaker was it — wall-clock elapsed minus the seconds of audio delivered so
    far. A chunked engine's figure is its own send-to-return time. Both are
    labelled with their transport, because they are not the same measurement.

    Audio is NOT paced here. `hamsa_stt_chunk_sleep_sec` exists to feed a stored
    file at roughly real time; microphone audio already arrives at real time, and
    pacing it again would add delay to a number reported as the engine's.
    """
    await websocket.accept()

    if not live_session.exists(session_id):
        await websocket.send_json({"type": "error", "message": "Unknown or expired session"})
        await websocket.close()
        return
    if asr_id not in ASR_ENGINES:
        await websocket.send_json({"type": "error", "message": f"Unknown engine {asr_id!r}"})
        await websocket.close()
        return

    import asyncio
    import json
    import time

    import websockets

    from apps.background_worker.transcription.hamsa import adapter as hamsa_adapter
    from apps.background_worker.transcription.hamsa import runner as hamsa_runner

    settings = get_settings()
    bytes_per_sec = settings.live_record_sample_rate * 2  # 16-bit mono
    started = time.perf_counter()
    audio_bytes_sent = 0
    index = 0
    client_done = asyncio.Event()

    try:
        ws_url, kwargs = hamsa_runner.connect_kwargs()
    except RuntimeError as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close()
        return

    try:
        async with websockets.connect(ws_url, **kwargs) as upstream:
            await upstream.send(hamsa_runner.handshake_payload())
            ack = json.loads(await asyncio.wait_for(upstream.recv(), timeout=10.0))
            if ack.get("type") != "handshake_ack":
                await websocket.send_json(
                    {"type": "error", "message": f"handshake rejected: {str(ack)[:200]}"}
                )
                return
            await websocket.send_json({"type": "ready", "asrId": asr_id})

            async def pump_audio() -> None:
                """Browser -> engine. Re-chunks to the engine's fixed frame size."""
                nonlocal audio_bytes_sent
                buffer = bytearray()
                while True:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        break
                    payload = message.get("bytes")
                    if payload:
                        buffer.extend(payload)
                        # CHUNK_BYTES is the engine's required frame size, not a
                        # tunable — its VAD is tuned around it.
                        while len(buffer) >= hamsa_runner.CHUNK_BYTES:
                            frame = bytes(buffer[: hamsa_runner.CHUNK_BYTES])
                            del buffer[: hamsa_runner.CHUNK_BYTES]
                            await upstream.send(frame)
                            audio_bytes_sent += len(frame)
                        continue
                    text = message.get("text")
                    if text and json.loads(text).get("type") == "stop":
                        break
                # Flush the partial frame, then finalize and pad: the VAD needs
                # trailing quiet to close and emit its last segment.
                if buffer:
                    await upstream.send(bytes(buffer))
                    audio_bytes_sent += len(buffer)
                await upstream.send(json.dumps({"type": "finalize"}))
                await upstream.send(b"\x00" * hamsa_runner.CHUNK_BYTES * 3)
                client_done.set()

            async def pump_text() -> None:
                """Engine -> browser, recording each segment in the session.

                Every wait is bounded for the reason the stored-file runner
                documents: blocking in a timeout-less recv() would stop this task
                from ever re-checking whether the client finished, so a quiet
                server would hang the socket open forever.
                """
                nonlocal index
                while True:
                    try:
                        raw = await asyncio.wait_for(
                            upstream.recv(), timeout=settings.hamsa_stt_idle_timeout_sec
                        )
                    except asyncio.TimeoutError:
                        if client_done.is_set():
                            return
                        continue
                    try:
                        message = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if message.get("type") == "error":
                        await websocket.send_json(
                            {"type": "error", "message": str(message)[:300]}
                        )
                        continue
                    # The engine's own adapter decides what counts as text, so the
                    # live path and the stored path read this shape identically.
                    text = hamsa_adapter.adapt([message]).strip()
                    if not text:
                        continue
                    elapsed_ms = int((time.perf_counter() - started) * 1000)
                    audio_ms = int(audio_bytes_sent / bytes_per_sec * 1000)
                    # Floored at 0: a segment can be reported slightly ahead of
                    # the audio counter, and a negative lag is not a thing.
                    lag_ms = max(0, elapsed_ms - audio_ms)
                    live_session.append_part(
                        session_id, asr_id, index=index, text=text, latency_ms=lag_ms,
                        # The engine's own frame, not just the text the adapter
                        # pulled out of it: it also carries language detection and
                        # this deployment's per-word timings.
                        raw=message,
                    )
                    await websocket.send_json(
                        {"type": "transcript", "text": text, "index": index, "latencyMs": lag_ms}
                    )
                    index += 1

            audio_task = asyncio.create_task(pump_audio())
            text_task = asyncio.create_task(pump_text())
            try:
                await asyncio.wait_for(
                    asyncio.gather(audio_task, text_task),
                    timeout=settings.hamsa_stt_session_timeout_sec,
                )
            finally:
                for task in (audio_task, text_task):
                    if not task.done():
                        task.cancel()
            await websocket.send_json({"type": "done", "chunkCount": index})
    except WebSocketDisconnect:
        logger.info("live transcript socket closed by client (session %s)", session_id)
    except Exception as exc:  # noqa: BLE001 - surface the reason, then close
        logger.exception("live transcript relay failed for session %s", session_id)
        with suppress(RuntimeError):
            await websocket.send_json({"type": "error", "message": str(exc)[:300]})
    finally:
        with suppress(RuntimeError):
            await websocket.close()


@router.post("/session/{session_id}/finalize", response_model=list[TranscriptRun],
             response_model_by_alias=True)
async def finalize_session(
    session_id: str,
    file: UploadFile = File(...),
    reference_text: str = Form("", alias="referenceText"),
    reference_source: str = Form("script", alias="referenceSource"),
    script_params: str = Form("", alias="scriptParams"),
    audio_file_id: int | None = Form(None, alias="audioFileId"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[TranscriptRun]:
    """Persist a finished read-aloud session: the recording, its reference, and
    each engine's captured transcript with its measured timings and scores.

    Nothing is re-run. What gets scored is exactly what each engine returned while
    the audio was being spoken, which is the whole point of holding the session
    server-side — a second pass would produce different numbers from the ones the
    panels showed, and there would be no honest way to label which was "the"
    result.

    The recording goes through the SAME ingest as an upload, so it lands in
    Projects, gets a real duration read from its header, and can be diarized later.
    It is ingested with no diarization models: one person reading a script aloud
    makes a speaker comparison a foregone conclusion, and spending GPU on it would
    be waste the operator did not ask for.
    """
    session = await run_in_threadpool(live_session.meta, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown or expired session")

    reference = (reference_text or session.get("reference_text") or "").strip()
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=422, detail="The recording was empty")

    # Shares the upload path's canonicalization and storage, so a read-aloud
    # recording is the same stored shape as any other and shows up in Projects
    # with a real duration read from its header. No diarization models and no
    # auto-transcript: this session already has both engines' transcripts.
    # Named here, not by the client: the name should match the row's own created_at
    # rather than whatever a browser's clock said. Sortable as plain text, which is
    # what the recordings list orders by.
    #
    # `filename` has no unique constraint (only s3_key does, and that is keyed on the
    # row id), so two recordings finished inside the same second share a display name.
    # Harmless, and not worth a counter.
    if audio_file_id is not None:
        # Attach to the row created when the script was generated, so one script
        # produces ONE Projects entry rather than a script-only row plus a separate
        # recorded one. Ownership and surface are checked here rather than trusted:
        # the id comes from the browser.
        audio_file = (
            db.query(AudioFile)
            .filter(
                AudioFile.id == audio_file_id,
                AudioFile.owner_id == current_user.id,
                AudioFile.surface == "transcript",
            )
            .one_or_none()
        )
        if audio_file is None:
            raise HTTPException(status_code=404, detail="Unknown recording")
        if audio_file.s3_key:
            # Re-recording over stored audio would leave the transcripts and scores
            # of the previous take pointing at different audio.
            raise HTTPException(
                status_code=409, detail="This recording already has audio; delete it to re-record"
            )
        audio_file = await run_in_threadpool(attach_audio, payload, audio_file, db)
    else:
        # No row yet: a session opened from a pasted reference rather than a
        # generated script. Same ingest as an upload.
        #
        # `filename` has no unique constraint (only s3_key does, and that is keyed on
        # the row id), so two recordings finished inside the same second share a
        # display name. Harmless, and not worth a counter.
        recording_name = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        audio_file = await run_in_threadpool(
            partial(store_recording, surface="transcript"),
            payload, recording_name, db, current_user,
        )

    # The reference row is written inside _persist_session, not here: an earlier
    # refactor left a copy in both places and SQLAlchemy batched them into one
    # multi-VALUES INSERT, which tripped the unique index on audio_file_id and
    # 500'd every finalize.
    #
    # Everything below blocks: Redis reads, WER over the reference, and the DB
    # writes. `async def` runs it on the event loop unless it is handed off, and a
    # long reference would stall every other request for the duration.
    def persist() -> list[TranscriptResult]:
        return _persist_session(db, session, session_id, audio_file, reference,
                                reference_source, script_params)

    rows = await run_in_threadpool(persist)
    live_session.close(session_id)
    logger.info(
        "Finalized live session %s as audio_file_id=%s (%d engines, %d reference words)",
        session_id, audio_file.id, len(rows), len(reference.split()),
    )
    return [transcript_run_contract(row) for row in rows]


def _persist_session(
    db: Session,
    session: dict,
    session_id: str,
    audio_file,
    reference: str,
    reference_source: str,
    script_params: str,
) -> list[TranscriptResult]:
    """The blocking half of finalize: read the session, score, write the rows.

    Split out so the endpoint can hand it to a threadpool in one call rather than
    awaiting a dozen small ones.
    """
    if reference:
        source = reference_source if reference_source in ("script", "pasted") else "script"
        params = json.loads(script_params) if script_params else None
        # Update, not insert, when the row is already there: a generated script
        # writes its reference at generation time, and `audio_file_id` is unique.
        # A blind add here batched into a multi-VALUES INSERT is what 500'd every
        # finalize once before, so this branch is load-bearing rather than defensive.
        existing = (
            db.query(TranscriptReference)
            .filter(TranscriptReference.audio_file_id == audio_file.id)
            .one_or_none()
        )
        if existing is None:
            db.add(TranscriptReference(
                audio_file_id=audio_file.id,
                source=source,
                text=reference,
                params=params,
            ))
        else:
            # The operator can edit the box before reading, so the text submitted at
            # finalize is the one that was actually read and wins over the generated
            # one. `params` only when supplied, so an edit does not erase how the
            # script was generated.
            existing.source = source
            existing.text = reference
            if params is not None:
                existing.params = params

    rows: list[TranscriptResult] = []
    # How this session actually ran each engine, resolved at open time. Falls back
    # to "live" only for a session opened before the field existed, which is what
    # every such session was.
    session_modes: dict[str, str] = session.get("modes") or {}
    for asr_id in session["asr_ids"]:
        mode = session_modes.get(asr_id) or "live"
        if mode == "batch":
            # Fed NOTHING while the operator read: this engine runs once over the
            # stored recording instead, and reaches done|failed on its own like any
            # stored-audio run.
            #
            # `_queue_transcript` stamps it source="batch", which is the honest
            # label and, with `transport`, says everything the mode did: the
            # reference and therefore the error rates are the same as the live
            # engines', the timings are not. For cohere the transport also changes
            # (file, one whole-file call); for inception-stt it does not (still
            # chunks, cut at BATCH_SEGMENT_SECONDS, just from storage).
            try:
                rows.append(_queue_transcript(db, audio_file, asr_id, get_settings()))
            except HTTPException as exc:
                # The engine was checked at open_session; if it has gone away since
                # (its container stopped mid-reading), that must not cost the live
                # engines their transcripts. Those were measured while someone was
                # speaking and cannot be reproduced -- this one runs over stored
                # audio and can be re-run from the recording at any time.
                logger.warning(
                    "Could not queue %s at finalize for audio_file_id=%s: %s",
                    asr_id, audio_file.id, exc.detail,
                )
                rows.append(TranscriptResult(
                    audio_file_id=audio_file.id, asr_id=asr_id, status="failed",
                    source="batch", transport="file", error=str(exc.detail),
                ))
                db.add(rows[-1])
            continue
        # Live from here down. The transport is derived from the mode rather than
        # read from ASR_TRANSPORTS, which is the BATCH half: cohere is chunks live
        # and file in batch, so the static map would mislabel a live cohere run.
        # (The batch branch above does not need this -- `_queue_transcript` stamps
        # the transport `run_asr` genuinely uses.)
        live_transport = transport_for_mode(asr_id, mode) or transport_for(asr_id)
        captured = live_session.transcript(session_id, asr_id)
        row = TranscriptResult(
            audio_file_id=audio_file.id,
            asr_id=asr_id,
            # Done, not queued: this transcript already exists. There is no job to
            # wait for, and showing "queued" would imply one was coming.
            status="done",
            source="live",
            transport=live_transport,
            text=captured["text"],
            chunk_count=captured["chunk_count"],
            chunk_latencies_ms=captured["chunk_latencies_ms"],
            first_latency_ms=captured["first_latency_ms"],
            avg_latency_ms=captured["avg_latency_ms"],
            # Only meaningful for a chunked transport; a stream has no interval.
            chunk_interval_sec=(
                session["chunk_interval_sec"] if live_transport == "chunks" else None
            ),
            asr_ms=captured["total_latency_ms"],
            # Native output, verbatim, exactly as the stored-audio pipeline keeps
            # it. Deferred on the column, so no polled route pays for it.
            raw_output=captured["raw"],
        )
        if reference:
            scoring.score_row(row, reference, audio_file.duration_sec)
        db.add(row)
        rows.append(row)

    db.commit()
    # After the commit, so a job can never start against a row that is not yet
    # visible to the worker's own session. Same ordering as `start_transcripts`.
    for row in rows:
        if row.status == "queued":
            # The row's own feed mode, so the job finds the row it was written
            # for: rows are keyed on (recording, engine, source) now, and a job
            # that assumed batch would look past a live row and drop itself.
            queue.enqueue(run_asr, audio_file.id, row.asr_id, row.source)
    return rows
