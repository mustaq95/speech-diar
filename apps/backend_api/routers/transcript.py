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
    live_session,
    resolve_asr_ids,
    transport_for,
)
from apps.background_worker.transcription import scoring
from apps.background_worker.transcription.inception import runner as inception_runner
from apps.backend_api.dependencies import get_current_user, get_db
from apps.backend_api.routers.evaluations import _transcript_run as transcript_run_contract
from packages import script_gen
from packages.config.settings import get_settings
from apps.backend_api.routers.upload import store_recording
from packages.shared_contracts.schemas import GeneratedScript, ScriptRequest, TranscriptRun
from packages.database.models import (
    TranscriptReference,
    TranscriptResult,
    User,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/transcript", tags=["transcript"])


@router.post("/script", response_model=GeneratedScript, response_model_by_alias=True)
def generate_script(
    request: ScriptRequest,
    current_user: User = Depends(get_current_user),
) -> GeneratedScript:
    """Generate a script to read aloud, which becomes the reference for scoring.

    Stateless: the browser holds the script and submits it at finalize, so there
    is no script table and an abandoned script costs nothing.

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

    return GeneratedScript(
        text=script.text,
        word_count=script.word_count,
        generator_model=script.generator_model,
        params=script.params,
    )


@router.post("/session")
def open_session(
    asr_ids: list[str] | None = Body(None, embed=True, alias="asrIds"),
    chunk_interval_sec: float | None = Body(None, embed=True, alias="chunkIntervalSec"),
    reference_text: str = Body("", embed=True, alias="referenceText"),
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

    interval = chunk_interval_sec or settings.live_chunk_default_sec
    if not settings.live_chunk_min_sec <= interval <= settings.live_chunk_max_sec:
        raise HTTPException(
            status_code=422,
            detail=f"chunkIntervalSec must be between {settings.live_chunk_min_sec} "
                   f"and {settings.live_chunk_max_sec}",
        )

    session_id = live_session.create(resolved, interval, reference_text)
    return {
        "sessionId": session_id,
        "asrIds": resolved,
        "chunkIntervalSec": interval,
        "sampleRate": settings.live_record_sample_rate,
        "engines": [
            {
                "asrId": asr_id,
                "name": ASR_ENGINES[asr_id].name,
                "transport": transport_for(asr_id),
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
    if asr_id not in ASR_ENGINES:
        raise HTTPException(status_code=422, detail=f"Unknown ASR engine {asr_id!r}")

    payload = await file.read()
    try:
        # run_in_threadpool, NOT a bare call: `transcribe_bytes` blocks on
        # httpx.post for as long as the gateway takes, and this handler is
        # `async def`, so calling it directly blocks the whole event loop.
        #
        # That was not theoretical. Measured with three chunk calls in flight,
        # /health latency went from 4 ms to 1066 ms — and the casualties were the
        # things sharing the loop: the streaming engine's WebSocket relay was
        # starved (3 words captured over two minutes of speech) and finalize sat
        # behind every queued chunk, which is what made Stop look like it hung.
        entry = await run_in_threadpool(
            inception_runner.transcribe_bytes, payload, file.filename or "chunk.wav"
        )
    except inception_runner.InceptionError as exc:
        # Recorded as an empty chunk rather than swallowed: the call was made and
        # took time, so the chunk count stays truthful, and the error reaches the
        # panel instead of looking like silence.
        await run_in_threadpool(
            partial(live_session.append_part, session_id, asr_id,
                    index=chunk_index, text="", latency_ms=0, raw=None)
        )
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    text = inception_runner.text_of(entry)
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
        db.add(TranscriptReference(
            audio_file_id=audio_file.id,
            source=reference_source if reference_source in ("script", "pasted") else "script",
            text=reference,
            params=json.loads(script_params) if script_params else None,
        ))

    rows: list[TranscriptResult] = []
    for asr_id in session["asr_ids"]:
        captured = live_session.transcript(session_id, asr_id)
        row = TranscriptResult(
            audio_file_id=audio_file.id,
            asr_id=asr_id,
            # Done, not queued: this transcript already exists. There is no job to
            # wait for, and showing "queued" would imply one was coming.
            status="done",
            source="live",
            transport=transport_for(asr_id),
            text=captured["text"],
            chunk_count=captured["chunk_count"],
            chunk_latencies_ms=captured["chunk_latencies_ms"],
            first_latency_ms=captured["first_latency_ms"],
            avg_latency_ms=captured["avg_latency_ms"],
            # Only meaningful for a chunked transport; a stream has no interval.
            chunk_interval_sec=(
                session["chunk_interval_sec"] if transport_for(asr_id) == "chunks" else None
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
    return rows
