"""Speechmatics Real-Time v2 -- live relay for the read-aloud surface.

The batch (whole-file, job queue) half of this engine lives in `runner.py`.
This module is the STREAMING half and is only reached by the WebSocket relay
in `apps/backend_api/routers/transcript.py` when `asr_id == "speechmatics"`.

## Protocol

Verified end to end on 2026-09-01 with the pyannote 30 s sample: WS open at
t+420 ms, RecognitionStarted at t+602 ms confirming
`language_pack_info.language_description == "Arabic and English"`, first
partial at t+856 ms, 53 finals, clean EndOfTranscript.

Endpoint (region matches the JWT audience; default `eu.rt.speechmatics.com`):

    wss://<region>.rt.speechmatics.com/v2?jwt=<temp-jwt>

Auth: a SHORT-LIVED JWT minted at connect time from the account API key via
`POST https://mp.speechmatics.com/v1/api_keys?type=rt` with body
`{"ttl": <SPEECHMATICS_RT_TEMP_KEY_TTL_SEC>}`. The relay mints its own token
per session because RT tokens are single-use per session and short-lived by
design; a long-lived token in the query string is what the mint route exists
to make unnecessary.

Message flow:

    client -> {"message":"StartRecognition",
               "audio_format":{"type":"raw","encoding":"pcm_s16le","sample_rate":16000},
               "transcription_config":{"language":"ar_en","operating_point":"enhanced",
                                        "enable_partials":true,"max_delay":2}}
    server -> {"message":"RecognitionStarted", ...} (plus optional Info frames)
    client -> <raw PCM_S16LE bytes>   (BINARY, not JSON)
    server -> {"message":"AudioAdded","seq_no":N}    (one per binary frame we sent)
    server -> {"message":"AddPartialTranscript","metadata":{"transcript":"...", ...}, ...}
    server -> {"message":"AddTranscript","metadata":{"transcript":"...", ...}, ...}
    client -> {"message":"EndOfStream","last_seq_no":<total-audio-frames>}
    server -> ...trailing finals...
    server -> {"message":"EndOfTranscript"}

Two load-bearing details from probing and the docs:

1. **Audio is BINARY, transcripts and control are JSON.** Send raw PCM bytes
   on the same socket as your control frames -- do NOT wrap them. The server
   counts one AudioAdded per binary frame received, and that counter is what
   EndOfStream's `last_seq_no` must equal.
2. **Only AddTranscript is scoreable.** AddPartialTranscript is a hypothesis
   the engine then rewrites; scoring a partial would attribute a word the
   model already retracted. Partials are forwarded to the browser for visible
   feedback only, same rule Hamsa and ElevenLabs follow.

## Latency

Same rule as every stream: a real-time protocol consumes audio at 1x by
definition, so "how long did the call take" has no answer. What is measurable
is lag behind live -- wall-clock elapsed since session start minus the
audio-seconds already delivered. That is `latency_ms` per recorded segment,
and the panel labels the figure with the transport.
"""

import asyncio
import json
import logging
import time
from contextlib import suppress
from typing import Any

import httpx
from fastapi import WebSocket, WebSocketDisconnect

from apps.background_worker.transcription import live_session
from packages.config.settings import get_settings

logger = logging.getLogger(__name__)

#: Bytes per second at 16 kHz mono 16-bit PCM, used to convert audio bytes
#: consumed into audio-seconds delivered for the lag figure.
_BYTES_PER_SEC = 16000 * 2

#: PCM frame size sent as one binary WS message. 200 ms was verified end to
#: end on 2026-09-01. Smaller frames give a tighter lag but multiply the
#: server's AudioAdded chatter; larger frames delay the first partial.
_FRAME_MS = 200
_FRAME_BYTES = 16000 * 2 * _FRAME_MS // 1000


class SpeechmaticsRealtimeError(RuntimeError):
    """Raised when the Speechmatics RT relay cannot proceed."""


async def _mint_temp_key(api_key: str, mint_url: str, ttl: int) -> str:
    """Mint a short-lived RT JWT from the long-lived API key.

    Split out so a caller can substitute a stub in tests. The management-plane
    call is deliberately synchronous relative to the socket open below; the
    token has to exist BEFORE the wss:// URL is built, and the mint routinely
    returns in under 300 ms.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(
            f"{mint_url}?type=rt",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"ttl": ttl},
        )
    if response.status_code != 201:
        raise SpeechmaticsRealtimeError(
            f"speechmatics RT: temp key mint failed "
            f"({response.status_code}): {response.text[:300]}"
        )
    payload = response.json()
    token = payload.get("key_value") or payload.get("jwt")
    if not token:
        raise SpeechmaticsRealtimeError(
            f"speechmatics RT: mint response missing key_value: {str(payload)[:200]}"
        )
    return token


def start_recognition_message() -> dict[str, Any]:
    """The StartRecognition frame for a run, from settings.

    Exposed so a test can assert the language pack and partial/max_delay
    values without opening a socket. Language and operating point are shared
    with the batch runner because a live comparison against the same engine's
    batch output must be measuring the same model.
    """
    settings = get_settings()
    return {
        "message": "StartRecognition",
        "audio_format": {
            "type": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": 16000,
        },
        "transcription_config": {
            "language": settings.speechmatics_language,
            "operating_point": settings.speechmatics_operating_point,
            "enable_partials": settings.speechmatics_rt_enable_partials,
            "max_delay": settings.speechmatics_rt_max_delay,
        },
    }


async def run(
    websocket: WebSocket,
    session_id: str,
    asr_id: str,
) -> None:
    """Relay one live session between the browser and Speechmatics RT.

    Same shape as the Hamsa and ElevenLabs relays: two tasks under one
    upstream connection pump audio one way and text the other, both bounded
    on waits so a silent server cannot hang the socket open.
    """
    import websockets

    settings = get_settings()
    if not settings.speechmatics_api_key:
        await websocket.send_json(
            {"type": "error",
             "message": "speechmatics RT is not configured -- set SPEECHMATICS_API_KEY in .env"}
        )
        await websocket.close()
        return

    try:
        token = await _mint_temp_key(
            settings.speechmatics_api_key,
            settings.speechmatics_rt_mint_url,
            settings.speechmatics_rt_temp_key_ttl_sec,
        )
    except SpeechmaticsRealtimeError as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close()
        return

    url = f"{settings.speechmatics_rt_url.rstrip('/')}?jwt={token}"

    started = time.perf_counter()
    audio_bytes_sent = 0
    audio_seq_no = 0  # one per BINARY frame sent; last_seq_no on EndOfStream
    index = 0
    client_done = asyncio.Event()

    try:
        async with websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
            max_size=None,
        ) as upstream:
            await upstream.send(json.dumps(start_recognition_message()))
            # Wait for RecognitionStarted before signalling ready. Info frames
            # (concurrent_session_usage, recognition_quality) can arrive first
            # and are legitimate; skip past them.
            ready_deadline = time.perf_counter() + 10.0
            while True:
                if time.perf_counter() >= ready_deadline:
                    await websocket.send_json(
                        {"type": "error", "message": "speechmatics RT: RecognitionStarted timed out"}
                    )
                    return
                raw = await asyncio.wait_for(
                    upstream.recv(), timeout=ready_deadline - time.perf_counter()
                )
                if not isinstance(raw, str):
                    continue
                try:
                    handshake = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                mt = handshake.get("message")
                if mt == "RecognitionStarted":
                    break
                if mt == "Error":
                    await websocket.send_json(
                        {"type": "error", "message": f"speechmatics RT rejected: {str(handshake)[:300]}"}
                    )
                    return
                # Info / anything else: keep waiting.
            await websocket.send_json({"type": "ready", "asrId": asr_id})

            async def pump_audio() -> None:
                """Browser -> Speechmatics. Binary PCM frames, one per send.

                Re-frames to the 200 ms size verified in the probe. AudioAdded
                acks land on the reader side and are counted only for logging;
                the authoritative sequence counter is `audio_seq_no` here,
                which is what EndOfStream's last_seq_no must equal.
                """
                nonlocal audio_bytes_sent, audio_seq_no
                buffer = bytearray()
                while True:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        break
                    payload = message.get("bytes")
                    if payload:
                        buffer.extend(payload)
                        while len(buffer) >= _FRAME_BYTES:
                            frame = bytes(buffer[:_FRAME_BYTES])
                            del buffer[:_FRAME_BYTES]
                            await upstream.send(frame)
                            audio_seq_no += 1
                            audio_bytes_sent += len(frame)
                        continue
                    text = message.get("text")
                    if text and json.loads(text).get("type") == "stop":
                        break
                # Flush any partial-frame tail as one final binary send. The
                # v2 protocol does not require frames to be a fixed size; the
                # 200 ms shape is our own for even latency, not the wire's.
                if buffer:
                    await upstream.send(bytes(buffer))
                    audio_seq_no += 1
                    audio_bytes_sent += len(buffer)
                # EndOfStream tells the server we are done sending audio; it
                # will emit any trailing AddTranscripts and then
                # EndOfTranscript. last_seq_no must equal the total number of
                # binary frames actually sent.
                await upstream.send(
                    json.dumps({"message": "EndOfStream", "last_seq_no": audio_seq_no})
                )
                client_done.set()

            async def pump_text() -> None:
                """Speechmatics -> browser, recording each final segment.

                Bounded waits, same reason as the other stream relays: an
                unbounded recv() would stop this task ever re-checking whether
                the client finished, so a silent server would hang the socket
                open forever.
                """
                nonlocal index
                while True:
                    try:
                        raw = await asyncio.wait_for(
                            upstream.recv(),
                            timeout=settings.speechmatics_rt_idle_timeout_sec,
                        )
                    except asyncio.TimeoutError:
                        if client_done.is_set():
                            return
                        continue
                    if not isinstance(raw, str):
                        # Speechmatics does not send binary from server to
                        # client, but ignore rather than crash if that changes.
                        continue
                    try:
                        message = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    mt = message.get("message")
                    if mt == "Error" or mt == "Warning":
                        # Forward and keep going. Some server-side errors
                        # (job_error, session termination) will close the
                        # socket next anyway, and we surface the reason.
                        await websocket.send_json(
                            {"type": "error" if mt == "Error" else "warning",
                             "message": str(message)[:300]}
                        )
                        if mt == "Error":
                            return
                        continue
                    if mt in ("Info", "AudioAdded"):
                        # Housekeeping only; the sequence counter for
                        # EndOfStream is tracked on the sender side. Nothing
                        # to record.
                        continue
                    if mt == "AddPartialTranscript":
                        # Forwarded for visual feedback only, NOT recorded.
                        text = str(message.get("metadata", {}).get("transcript") or "").strip()
                        if text:
                            await websocket.send_json(
                                {"type": "partial", "text": text}
                            )
                        continue
                    if mt == "AddTranscript":
                        text = str(message.get("metadata", {}).get("transcript") or "").strip()
                        if not text:
                            continue
                        elapsed_ms = int((time.perf_counter() - started) * 1000)
                        audio_ms = int(audio_bytes_sent / _BYTES_PER_SEC * 1000)
                        lag_ms = max(0, elapsed_ms - audio_ms)
                        live_session.append_part(
                            session_id,
                            asr_id,
                            index=index,
                            text=text,
                            latency_ms=lag_ms,
                            # The engine's own frame, not just the text. It
                            # carries per-word timings and speaker labels the
                            # adapter necessarily drops.
                            raw=message,
                        )
                        await websocket.send_json(
                            {
                                "type": "transcript",
                                "text": text,
                                "index": index,
                                "latencyMs": lag_ms,
                            }
                        )
                        index += 1
                        continue
                    if mt == "EndOfTranscript":
                        return
                    # Unknown frame: log once at info level. Not an error --
                    # Speechmatics can add message types (documented via
                    # release notes) without breaking older clients.
                    logger.info(
                        "speechmatics RT: unhandled message type %r in session %s",
                        mt, session_id,
                    )

            audio_task = asyncio.create_task(pump_audio())
            text_task = asyncio.create_task(pump_text())
            try:
                await asyncio.wait_for(
                    asyncio.gather(audio_task, text_task),
                    timeout=settings.speechmatics_rt_session_timeout_sec,
                )
            finally:
                for task in (audio_task, text_task):
                    if not task.done():
                        task.cancel()
            await websocket.send_json({"type": "done", "chunkCount": index})
    except WebSocketDisconnect:
        logger.info(
            "speechmatics RT relay: client closed session %s", session_id
        )
    except Exception as exc:  # noqa: BLE001 - surface the reason, then close
        logger.exception(
            "speechmatics RT relay failed for session %s", session_id
        )
        with suppress(RuntimeError):
            await websocket.send_json({"type": "error", "message": str(exc)[:300]})
    finally:
        with suppress(RuntimeError):
            await websocket.close()


#: Native shape returned by `stream_replay`: the ordered list of AddTranscript
#: frames the server emitted for a full pass over the file. Partials are
#: dropped for the same reason a live run does not score them (hypothesis
#: the model then rewrites).
StreamReplayFrames = list[dict[str, Any]]


async def _stream_replay_async(audio_path: str) -> tuple[StreamReplayFrames, list[int]]:
    """Feed a stored WAV through the Speechmatics RT v2 WS at 1x pace.

    Returns `(frames, latencies_ms)` where:
      * `frames` is the list of `AddTranscript` messages, in arrival order.
        Nothing else: partials, Info, AudioAdded and RecognitionStarted are
        housekeeping.
      * `latencies_ms` is the arrival-lag per final frame -- wall-clock
        elapsed since session start minus the audio-seconds delivered when
        that frame arrived. Same formula the live relay uses, so a replayed
        row and a read-aloud row measure lag the same way.

    Pacing is 1x realtime (200 ms PCM every 200 ms of wall clock). Do NOT
    remove the sleep. Feeding as fast as the socket accepts makes the server
    consume everything and emit AddTranscripts back-to-back at the end,
    which measures nothing about lag and would produce a row whose
    latencies read as near-zero. If the goal is speed, use the batch runner
    (job queue on `asr.api.speechmatics.com`); this function measures the
    streaming product (`<region>.rt.speechmatics.com`) on stored audio.
    """
    import wave

    import websockets

    with wave.open(audio_path, "rb") as wav:
        if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise SpeechmaticsRealtimeError(
                f"speechmatics stream replay expects 16 kHz mono 16-bit PCM; got "
                f"rate={wav.getframerate()} channels={wav.getnchannels()} "
                f"sampwidth={wav.getsampwidth()}"
            )
        pcm = wav.readframes(wav.getnframes())

    settings = get_settings()
    if not settings.speechmatics_api_key:
        raise SpeechmaticsRealtimeError(
            "speechmatics stream replay is not configured -- set SPEECHMATICS_API_KEY in .env"
        )
    token = await _mint_temp_key(
        settings.speechmatics_api_key,
        settings.speechmatics_rt_mint_url,
        settings.speechmatics_rt_temp_key_ttl_sec,
    )
    url = f"{settings.speechmatics_rt_url.rstrip('/')}?jwt={token}"

    frames: StreamReplayFrames = []
    latencies_ms: list[int] = []
    started = time.perf_counter()
    audio_bytes_sent = 0
    audio_seq_no = 0
    end_of_transcript = asyncio.Event()

    async with websockets.connect(
        url,
        ping_interval=20,
        ping_timeout=20,
        close_timeout=5,
        max_size=None,
    ) as upstream:
        await upstream.send(json.dumps(start_recognition_message()))
        ready_deadline = time.perf_counter() + 10.0
        while True:
            if time.perf_counter() >= ready_deadline:
                raise SpeechmaticsRealtimeError(
                    "speechmatics stream replay: RecognitionStarted timed out"
                )
            raw = await asyncio.wait_for(
                upstream.recv(), timeout=ready_deadline - time.perf_counter()
            )
            if not isinstance(raw, str):
                continue
            try:
                handshake = json.loads(raw)
            except json.JSONDecodeError:
                continue
            mt = handshake.get("message")
            if mt == "RecognitionStarted":
                break
            if mt == "Error":
                raise SpeechmaticsRealtimeError(
                    f"speechmatics stream replay rejected: {str(handshake)[:300]}"
                )

        async def sender() -> None:
            nonlocal audio_bytes_sent, audio_seq_no
            for offset in range(0, len(pcm), _FRAME_BYTES):
                frame = pcm[offset:offset + _FRAME_BYTES]
                await upstream.send(frame)
                audio_seq_no += 1
                audio_bytes_sent += len(frame)
                # 1x pacing (see the docstring).
                await asyncio.sleep(_FRAME_MS / 1000)
            await upstream.send(
                json.dumps({"message": "EndOfStream", "last_seq_no": audio_seq_no})
            )

        async def reader() -> None:
            while True:
                try:
                    raw = await asyncio.wait_for(
                        upstream.recv(),
                        timeout=settings.speechmatics_rt_idle_timeout_sec,
                    )
                except asyncio.TimeoutError:
                    return
                if not isinstance(raw, str):
                    continue
                try:
                    message = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                mt = message.get("message")
                if mt == "AddTranscript":
                    elapsed_ms = int((time.perf_counter() - started) * 1000)
                    audio_ms = int(audio_bytes_sent / _BYTES_PER_SEC * 1000)
                    lag_ms = max(0, elapsed_ms - audio_ms)
                    frames.append(message)
                    latencies_ms.append(lag_ms)
                elif mt == "EndOfTranscript":
                    end_of_transcript.set()
                    return
                elif mt == "Error":
                    raise SpeechmaticsRealtimeError(
                        f"speechmatics stream replay: {str(message)[:300]}"
                    )
                # AddPartialTranscript / Info / AudioAdded / Warning: skip.

        sender_task = asyncio.create_task(sender())
        reader_task = asyncio.create_task(reader())
        try:
            await sender_task
            # Wait for EndOfTranscript up to the idle timeout after audio
            # end. The reader exits on either event.
            await asyncio.wait_for(
                reader_task, timeout=settings.speechmatics_rt_idle_timeout_sec + 2
            )
        except asyncio.TimeoutError:
            if not reader_task.done():
                reader_task.cancel()
        finally:
            if not sender_task.done():
                sender_task.cancel()

    return frames, latencies_ms


def stream_replay(audio_path: str) -> tuple[StreamReplayFrames, list[int]]:
    """Sync entry point for `AsrEngine.run_stream` -- feeds a stored WAV
    through Speechmatics RT at 1x pace. See `_stream_replay_async`."""
    return asyncio.run(_stream_replay_async(audio_path))
