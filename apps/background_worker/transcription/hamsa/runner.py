"""TryHamsa STT — execution code for the ONLINE transcription mode.

Streams 16 kHz mono PCM to a Hamsa deployment over a WebSocket and returns the
server's own messages untouched; only `adapter.py` is allowed to understand
that shape.

This is a STREAMING protocol, not request/response. Audio is paced to the
server in 100 ms chunks so its VAD can segment speech as it arrives, which
means ASR wall-clock is a function of the recording's LENGTH (~50 % of it at
the default pace), not of model speed. Nothing here can make a long recording
transcribe quickly; that is the protocol, and the UI labels the reported time
accordingly.

Audio LEAVES this host in this mode. Offline mode runs `../cohere/` locally
instead.

Configuration (`.env`, read via `packages/config/settings.py`):
  HAMSA_STT_WS_URL / HAMSA_STT_URL — the wss:// endpoint
  HAMSA_STT_KEY                   — x-api-key header
  HAMSA_STT_BEARER_TOKEN          — optional Bearer token (ADEO deployments)
  HAMSA_STT_SSL_VERIFY            — false for a cert this host doesn't trust
  HAMSA_STT_CHUNK_SLEEP_SEC       — the streaming pace (see above)
  HAMSA_STT_IDLE_TIMEOUT_SEC      — socket silence that marks the end
  HAMSA_STT_SESSION_TIMEOUT_SEC   — hard ceiling on one session
"""

import asyncio
import json
import logging
import ssl
from typing import Any

from packages.audio import ensure_canonical_wav, read_pcm16
from packages.config.settings import get_settings

logger = logging.getLogger(__name__)

#: Hamsa's native shape: the ordered list of messages the server sent.
HamsaRawOutput = list[dict[str, Any]]

#: 100 ms of 16 kHz / 16-bit / mono PCM. The server's VAD is tuned around this
#: chunk size, so it is fixed rather than configurable.
CHUNK_BYTES = 3200


def _ssl_context(ws_url: str, verify: bool) -> ssl.SSLContext | None:
    if not ws_url.startswith("wss://") or verify:
        return None
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    logger.warning("Hamsa TLS verification disabled by HAMSA_STT_SSL_VERIFY=false")
    return ctx


def _handshake(api_key: str, bearer: str | None, sample_rate: int) -> str:
    """The handshake payload, matching the options this deployment was verified
    against. The VAD/EOS values are the server's segmentation contract, not
    tuning knobs to guess at — changing them changes where segment boundaries
    fall and therefore the transcript."""
    return json.dumps(
        {
            "type": "handshake",
            "api_key": api_key,
            "authorization": f"Bearer {bearer or api_key}",
            "options": {
                "silence_timeout": 30,
                "sample_rate": sample_rate,
                "client_logging": False,
                "min_silence_duration_ms": 300,
                "min_speech_ms": 600,
                "vad_threshold": 0.6,
                "eos_enabled": True,
                "eos_threshold": 0.6,
                "audio_type": "PCM",
                "noise_cancellation": True,
            },
        }
    )


async def _stream(pcm: bytes) -> HamsaRawOutput:
    """Send audio and collect messages CONCURRENTLY.

    The obvious implementation — send every chunk, then read replies — is what
    the reference client does and it only works on short clips. Hamsa emits a
    message per detected speech segment while audio is still arriving, so on a
    32-minute file that version leaves ~950 s of server messages unread in the
    socket buffer and never services a ping. Two tasks under one connection
    keeps the socket healthy and memory flat regardless of length.
    """
    import websockets

    settings = get_settings()
    ws_url, kwargs = connect_kwargs()

    messages: HamsaRawOutput = []
    async with websockets.connect(ws_url, **kwargs) as ws:
        await ws.send(handshake_payload())
        ack = json.loads(await asyncio.wait_for(ws.recv(), timeout=10.0))
        if ack.get("type") != "handshake_ack":
            raise RuntimeError(f"Hamsa handshake rejected: {str(ack)[:300]}")

        async def send_audio() -> None:
            pace = settings.hamsa_stt_chunk_sleep_sec
            for offset in range(0, len(pcm), CHUNK_BYTES):
                await ws.send(pcm[offset : offset + CHUNK_BYTES])
                await asyncio.sleep(pace)
            await ws.send(json.dumps({"type": "finalize"}))
            # Silence padding: the server's VAD needs trailing quiet to close
            # and emit the final segment, which `finalize` alone does not
            # always flush.
            await ws.send(b"\x00" * CHUNK_BYTES * 3)

        async def collect() -> None:
            # Hamsa sends no end-of-stream marker, so the run ends when the
            # socket has been quiet for idle_timeout AND the audio is all sent.
            #
            # Every wait is bounded, deliberately. Waiting unbounded while the
            # sender is still going looks equivalent — silence mid-stream is
            # normal, so why time out? — but it deadlocks: once blocked in a
            # timeout-less recv() this task never re-checks whether the sender
            # finished, so a server that stays quiet until `finalize` hangs the
            # job forever. Timing out and re-checking is what makes the exit
            # condition observable.
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=settings.hamsa_stt_idle_timeout_sec)
                except asyncio.TimeoutError:
                    if sender.done():
                        return
                    continue  # still streaming; a gap between segments is not the end
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                messages.append(msg)
                if msg.get("type") == "error":
                    raise RuntimeError(f"Hamsa server error: {str(msg)[:300]}")

        sender = asyncio.create_task(send_audio())
        collector = asyncio.create_task(collect())
        try:
            await asyncio.wait_for(
                asyncio.gather(sender, collector), timeout=settings.hamsa_stt_session_timeout_sec
            )
        finally:
            for task in (sender, collector):
                if not task.done():
                    task.cancel()

    return messages


def connect_kwargs() -> tuple[str, dict[str, Any]]:
    """The URL and connect options for a Hamsa socket, plus validation.

    Extracted from `_stream` so the live relay opens its socket exactly the way a
    stored-file run does — same headers, same TLS decision, same ping settings.
    A second hand-rolled connection setup would be free to drift, and a
    difference there would show up as a difference in the engine's measured
    output.
    """
    settings = get_settings()
    ws_url = settings.hamsa_ws_endpoint
    api_key = settings.hamsa_stt_key
    if not ws_url or not api_key:
        raise RuntimeError("Hamsa STT needs HAMSA_STT_WS_URL and HAMSA_STT_KEY in .env")

    headers = {"x-api-key": api_key}
    if settings.hamsa_stt_bearer_token:
        headers["Authorization"] = f"Bearer {settings.hamsa_stt_bearer_token}"
    kwargs: dict[str, Any] = {
        "additional_headers": headers,
        "ping_interval": 20,
        "ping_timeout": 20,
        "close_timeout": 5,
        "max_size": None,  # a long file's final message can be large
    }
    ssl_ctx = _ssl_context(ws_url, settings.hamsa_stt_ssl_verify)
    if ssl_ctx is not None:
        kwargs["ssl"] = ssl_ctx
    return ws_url, kwargs


def handshake_payload() -> str:
    """The handshake for a Hamsa socket, from settings.

    Shared with the live relay for the same reason as `connect_kwargs`: the
    handshake carries the server's VAD/EOS contract, which determines where
    segment boundaries fall. Two copies could disagree and silently change the
    transcript.
    """
    settings = get_settings()
    return _handshake(
        settings.hamsa_stt_key, settings.hamsa_stt_bearer_token, settings.hamsa_stt_sample_rate
    )


def run(audio_path: str) -> HamsaRawOutput:
    """Transcribe `audio_path` and return Hamsa's native messages untouched."""
    send_path, cleanup = ensure_canonical_wav(audio_path)
    try:
        pcm = read_pcm16(send_path)
    finally:
        if cleanup:
            import os
            from contextlib import suppress

            with suppress(OSError):
                os.unlink(cleanup)
    if not pcm:
        return []
    return asyncio.run(_stream(pcm))
