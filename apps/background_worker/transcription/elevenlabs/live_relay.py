"""ElevenLabs Scribe v2 Realtime -- streaming half of this engine.

Two entry points, one wire protocol:

  * `run(websocket, session_id, asr_id)` -- the LIVE read-aloud relay. Bridges
    a browser WebSocket to the ElevenLabs realtime WS while a human is
    speaking. Called from `apps/backend_api/routers/transcript.py`.
  * `stream_replay(audio_path)` / `run_stream(audio_path)` -- the REPLAY-LIVE
    entry point. Feeds a saved WAV through the same realtime WS at 1x pace,
    returning the frames and per-final-arrival latencies. Called from
    `apps/background_worker/transcription/pipeline.py::_replay_live`. Both
    use the same protocol so the two measurements are directly comparable.

The batch/whole-file half of this engine (a plain HTTP POST to scribe_v2, a
different model id from the realtime one) lives in `runner.py`.

## Protocol

Verified end to end on 2026-09-01 by streaming the pyannote 30 s sample WAV
through the real API and reading back real transcripts. The docs page is
https://elevenlabs.io/docs/api-reference/speech-to-text/v-1-speech-to-text-realtime
and it matches what the server actually did.

Endpoint:

    wss://api.elevenlabs.io/v1/speech-to-text/realtime
        ?model_id=scribe_v2_realtime&commit_strategy=<manual|vad>

Auth: `xi-api-key` request header (server-side call). A separate single-use
token flow exists for browser use; we do not need it here because the browser
never talks to ElevenLabs directly, only to this relay.

The server sends `session_started` on connect with the FULL accepted config
(sample_rate, audio_format, vad thresholds, model_id, etc.). We wait for it
before signalling `ready` to our own client so the browser only starts
capturing after the upstream has confirmed. The rest of the exchange:

    client -> {"message_type":"input_audio_chunk",
               "audio_base_64":"<base64 PCM_S16LE>",
               "commit": false,
               "sample_rate": 16000}
    server -> {"message_type":"partial_transcript", "text":"...", ...}
    server -> {"message_type":"committed_transcript", "text":"...", ...}

Two load-bearing details from probing:

1. **The audio body is BASE64 inside JSON, not raw binary.** Sending raw PCM
   bytes on this socket makes the server close cleanly with WebSocket close
   code 1000 ~1 second in, with no error frame. A caller who reads Hamsa's
   runner and expects the same wire shape gets a mystery drop.
2. **`partial_transcript` is a hypothesis the model then rewrites**, not a
   segment. We forward partials to the browser for visible feedback but do
   NOT record them in `live_session`, because a scored partial that gets
   walked back would attribute a word the engine already retracted. Only
   `committed_transcript` (and `committed_transcript_with_timestamps` when
   `include_timestamps` is on) is a scoreable event.

## Latency

Same rule as Hamsa: a streaming protocol consumes audio at 1x by definition,
so "how long did the call take" has no answer. What is measurable is lag
behind live -- wall-clock elapsed since session start MINUS the audio-seconds
already delivered. That is what `latency_ms` on each recorded part carries,
and the panel labels the reported figure with the transport.
"""

import asyncio
import base64
import json
import logging
import time
from contextlib import suppress
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from apps.background_worker.transcription import live_session
from packages.config.settings import get_settings

logger = logging.getLogger(__name__)

#: Bytes per second at 16 kHz mono 16-bit PCM, used to convert audio bytes
#: consumed into audio-seconds delivered for the lag figure.
_BYTES_PER_SEC = 16000 * 2

#: How many PCM bytes we buffer before sending one JSON envelope upstream.
#: 200 ms of 16 kHz PCM16 mono = 6400 bytes. Larger than Hamsa's 100 ms frame
#: because each frame here pays JSON+base64 overhead, and 200 ms is the frame
#: size used to verify the protocol end to end.
_FRAME_MS = 200
_FRAME_BYTES = 16000 * 2 * _FRAME_MS // 1000


class ElevenLabsRealtimeError(RuntimeError):
    """Raised when the Scribe realtime relay cannot proceed."""


def connect_url() -> str:
    """The wss:// URL with model and commit strategy applied.

    Exposed so a test can assert the composition without opening a socket, and
    so a follow-up feature (keyterms, secondary_languages) can extend it in one
    place rather than by string-concatenation at the call site.
    """
    settings = get_settings()
    base = settings.elevenlabs_stt_realtime_url.rstrip("/")
    model = settings.elevenlabs_stt_realtime_model
    strategy = settings.elevenlabs_stt_realtime_commit_strategy
    return f"{base}?model_id={model}&commit_strategy={strategy}"


def connect_headers() -> dict[str, str]:
    """The auth headers, or raise if the engine is not configured.

    A dedicated function so a bad setup fails BEFORE `websockets.connect` runs
    with a clearer message than the socket's TLS/handshake exception.
    """
    settings = get_settings()
    if not settings.elevenlabs_api_key:
        raise ElevenLabsRealtimeError(
            "elevenlabs-scribe-v2 realtime is not configured -- set ELEVENLABS_API_KEY in .env"
        )
    return {"xi-api-key": settings.elevenlabs_api_key}


async def run(
    websocket: WebSocket,
    session_id: str,
    asr_id: str,
) -> None:
    """Relay one live session between the browser and Scribe Realtime.

    Called by the WS route in `apps/backend_api/routers/transcript.py` after
    the client socket has been accepted and the session has been validated.
    Mirrors the hamsa relay's shape (two tasks under one upstream connection)
    so the two engines behave the same way at the boundaries -- ping cadence,
    idle timeouts, session timeout, close on client disconnect.
    """
    import websockets

    settings = get_settings()
    started = time.perf_counter()
    audio_bytes_sent = 0
    index = 0
    client_done = asyncio.Event()

    try:
        url = connect_url()
        headers = connect_headers()
    except ElevenLabsRealtimeError as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close()
        return

    try:
        async with websockets.connect(
            url,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
            max_size=None,
        ) as upstream:
            # First frame from the server is `session_started`. Wait for it
            # before signalling `ready` so the browser only starts capturing
            # after the upstream has confirmed. Same reason as Hamsa's ack.
            handshake_raw = await asyncio.wait_for(upstream.recv(), timeout=10.0)
            try:
                handshake = json.loads(handshake_raw) if isinstance(handshake_raw, str) else {}
            except json.JSONDecodeError:
                handshake = {}
            if handshake.get("message_type") != "session_started":
                await websocket.send_json(
                    {"type": "error", "message": f"handshake rejected: {str(handshake)[:200]}"}
                )
                return
            await websocket.send_json({"type": "ready", "asrId": asr_id})

            async def pump_audio() -> None:
                """Browser -> ElevenLabs. Re-frames to 200 ms JSON envelopes."""
                nonlocal audio_bytes_sent
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
                            await _send_chunk(upstream, frame, commit=False)
                            audio_bytes_sent += len(frame)
                        continue
                    text = message.get("text")
                    if text and json.loads(text).get("type") == "stop":
                        break
                # Flush any partial frame WITHOUT committing, then send an
                # empty committing chunk. The empty chunk is what tells the
                # server "no more audio, flush the final segment", and it
                # deliberately does not bump audio_bytes_sent -- it carries
                # no samples, and pretending otherwise would move the lag
                # figure by the tail flush's duration.
                if buffer:
                    await _send_chunk(upstream, bytes(buffer), commit=False)
                    audio_bytes_sent += len(buffer)
                await _send_chunk(upstream, b"", commit=True)
                client_done.set()

            async def pump_text() -> None:
                """ElevenLabs -> browser, recording each committed segment.

                Bounded waits, same reason as Hamsa: an unbounded recv() would
                stop this task ever re-checking whether the client finished,
                so a silent server would hang the socket open forever.
                """
                nonlocal index
                while True:
                    try:
                        raw = await asyncio.wait_for(
                            upstream.recv(),
                            timeout=settings.elevenlabs_stt_realtime_idle_timeout_sec,
                        )
                    except asyncio.TimeoutError:
                        if client_done.is_set():
                            return
                        continue
                    if not isinstance(raw, str):
                        continue
                    try:
                        message = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    mt = message.get("message_type")
                    if mt in ("input_error", "error"):
                        # Forward and keep going. Some server errors are
                        # session-fatal (QuotaExceeded,
                        # ScribeInsufficientAudioActivityError,
                        # ScribeChunkSizeExceededError) -- the socket will
                        # close on its own and we surface the reason.
                        await websocket.send_json(
                            {"type": "error", "message": str(message)[:300]}
                        )
                        continue
                    if mt == "partial_transcript":
                        # Forwarded for visual feedback ONLY, not recorded.
                        # See the module docstring for why.
                        text = str(message.get("text") or "").strip()
                        if text:
                            await websocket.send_json(
                                {"type": "partial", "text": text}
                            )
                        continue
                    if mt in ("committed_transcript", "committed_transcript_with_timestamps"):
                        text = str(message.get("text") or "").strip()
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
                            # carries per-word timestamps when the option is
                            # on, plus any entity-detection payloads, which
                            # the adapter necessarily drops.
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

            audio_task = asyncio.create_task(pump_audio())
            text_task = asyncio.create_task(pump_text())
            try:
                await asyncio.wait_for(
                    asyncio.gather(audio_task, text_task),
                    timeout=settings.elevenlabs_stt_realtime_session_timeout_sec,
                )
            finally:
                for task in (audio_task, text_task):
                    if not task.done():
                        task.cancel()
            await websocket.send_json({"type": "done", "chunkCount": index})
    except WebSocketDisconnect:
        logger.info(
            "elevenlabs realtime relay: client closed session %s", session_id
        )
    except Exception as exc:  # noqa: BLE001 - surface the reason, then close
        logger.exception(
            "elevenlabs realtime relay failed for session %s", session_id
        )
        with suppress(RuntimeError):
            await websocket.send_json({"type": "error", "message": str(exc)[:300]})
    finally:
        with suppress(RuntimeError):
            await websocket.close()


async def _send_chunk(upstream: Any, pcm: bytes, *, commit: bool) -> None:
    """Wrap raw PCM in the JSON envelope Scribe requires and send it upstream.

    The one place that knows the wire shape, so the wrong field name
    (`audio_chunk`, `data`, `audio` -- all measured to be rejected with
    `Could not parse the protocol message`) cannot creep in from a caller.
    """
    payload = {
        "message_type": "input_audio_chunk",
        "audio_base_64": base64.b64encode(pcm).decode("ascii"),
        "commit": commit,
        "sample_rate": 16000,
    }
    await upstream.send(json.dumps(payload))


#: Native shape returned by `stream_replay`: the ordered list of `committed_*`
#: frames the server emitted for a full pass over the file. Partials are
#: dropped here for the same reason they are not persisted from a live run --
#: a partial is a hypothesis the model then rewrites.
StreamReplayFrames = list[dict[str, Any]]


async def _stream_replay_async(audio_path: str) -> tuple[StreamReplayFrames, list[int]]:
    """Feed a stored WAV through the Scribe realtime WS at 1x pace.

    Returns `(frames, latencies_ms)` where:
      * `frames` is the list of `committed_transcript*` messages, in arrival
        order. Nothing else: partials are hypotheses, session_started is not
        an outcome, and Info frames are housekeeping.
      * `latencies_ms` is the arrival-lag per committed frame -- wall-clock
        elapsed since session start minus the audio-seconds delivered when
        that frame arrived. Same formula the live relay uses, so a replayed
        row and a read-aloud row measure lag the same way.

    Pacing is 1x realtime (200 ms PCM every 200 ms of wall clock). Do NOT
    remove the sleep. Feeding the whole file as fast as the socket accepts
    makes the server buffer everything and emit its commits back-to-back at
    the end -- which measures nothing about lag and produces a row whose
    latencies read as near-zero. If the goal is speed, use the batch runner
    (whole-file POST to `scribe_v2`); this function measures the streaming
    product (`scribe_v2_realtime`) on stored audio.
    """
    import wave

    import websockets

    with wave.open(audio_path, "rb") as wav:
        if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ElevenLabsRealtimeError(
                f"elevenlabs stream replay expects 16 kHz mono 16-bit PCM; got "
                f"rate={wav.getframerate()} channels={wav.getnchannels()} "
                f"sampwidth={wav.getsampwidth()}"
            )
        pcm = wav.readframes(wav.getnframes())

    url = connect_url()
    headers = connect_headers()

    frames: StreamReplayFrames = []
    latencies_ms: list[int] = []
    started = time.perf_counter()
    audio_bytes_sent = 0

    async with websockets.connect(
        url,
        additional_headers=headers,
        ping_interval=20,
        ping_timeout=20,
        close_timeout=5,
        max_size=None,
    ) as upstream:
        handshake_raw = await asyncio.wait_for(upstream.recv(), timeout=10.0)
        try:
            handshake = json.loads(handshake_raw) if isinstance(handshake_raw, str) else {}
        except json.JSONDecodeError:
            handshake = {}
        if handshake.get("message_type") != "session_started":
            raise ElevenLabsRealtimeError(
                f"elevenlabs stream replay: handshake rejected: {str(handshake)[:200]}"
            )

        done = asyncio.Event()

        async def sender() -> None:
            nonlocal audio_bytes_sent
            for offset in range(0, len(pcm), _FRAME_BYTES):
                frame = pcm[offset:offset + _FRAME_BYTES]
                await _send_chunk(upstream, frame, commit=False)
                audio_bytes_sent += len(frame)
                # 1x pacing. Sleeping for _FRAME_MS/1000 whether or not the
                # frame was full-sized: the last partial frame still occupies
                # its own audio slot on the real clock.
                await asyncio.sleep(_FRAME_MS / 1000)
            # Empty committing chunk -> flush the final segment. Does not
            # bump audio_bytes_sent for the same reason as the live relay:
            # this chunk carries no PCM samples.
            await _send_chunk(upstream, b"", commit=True)
            done.set()

        async def reader() -> None:
            settings = get_settings()
            while True:
                try:
                    raw = await asyncio.wait_for(
                        upstream.recv(),
                        timeout=settings.elevenlabs_stt_realtime_idle_timeout_sec,
                    )
                except asyncio.TimeoutError:
                    if done.is_set():
                        return
                    continue
                if not isinstance(raw, str):
                    continue
                try:
                    message = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                mt = message.get("message_type")
                if mt in ("committed_transcript", "committed_transcript_with_timestamps"):
                    elapsed_ms = int((time.perf_counter() - started) * 1000)
                    audio_ms = int(audio_bytes_sent / _BYTES_PER_SEC * 1000)
                    lag_ms = max(0, elapsed_ms - audio_ms)
                    frames.append(message)
                    latencies_ms.append(lag_ms)

        sender_task = asyncio.create_task(sender())
        reader_task = asyncio.create_task(reader())
        try:
            # Reader exits only via idle timeout after sender is done.
            await sender_task
            # Give the server a moment to flush any trailing commits after
            # our final commit=True. The idle timeout inside reader() bounds
            # this; we just wait for it to naturally exit.
            await asyncio.wait_for(
                reader_task, timeout=get_settings().elevenlabs_stt_realtime_idle_timeout_sec + 2
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
    through Scribe realtime at 1x pace. See `_stream_replay_async`."""
    return asyncio.run(_stream_replay_async(audio_path))
