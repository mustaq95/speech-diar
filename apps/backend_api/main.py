"""FastAPI application entry point.

Run locally (from repo root):  uv run uvicorn apps.backend_api.main:app --reload --port 8010
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.background_worker.transcription import (
    ASR_ENGINES,
    COMPARISON_ASR_IDS,
    DEFAULT_TRANSCRIPTION_MODE,
    MODE_TO_ASR_ID,
    transport_for,
)
from apps.background_worker.tts import COMPARISON_TTS_IDS, TTS_DELIVERY, TTS_ENGINES
from apps.backend_api.routers import evaluations, models, recordings, transcript, tts, upload
from packages.config.logging import configure_logging
from packages.config.settings import get_settings
from packages.database.session import init_db

configure_logging()
settings = get_settings()

app = FastAPI(
    title="Speaker Diarization Evaluation Platform",
    version="0.1.0",
)


@app.on_event("startup")
def _startup() -> None:
    init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(evaluations.router)
app.include_router(models.router)
app.include_router(recordings.router)
app.include_router(transcript.router)
app.include_router(tts.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
def frontend_config() -> dict[str, object]:
    """Runtime settings the frontend can pick up without a rebuild.

    The API's own address (VITE_API_BASE_URL) is the one value that must be
    known before the frontend can call this endpoint at all, so it stays a
    build-time setting; everything else here is .env-only, both to set and
    to change later.

    The transcription block reports, per mode, the engine's display name and
    whether it is configured on this host. The panel uses it to render the
    online/offline toggle and disable a side whose engine has no
    credentials/endpoint, instead of offering a run that is certain to fail.
    `defaultTranscriptionMode` is the mode an upload auto-transcribes in and the
    one the toggle starts on; it is not a fallback order, and the other mode
    never runs unless asked for.

    Which mode PRODUCED a given transcript is a different question, answered on
    each TranscriptResult row and never rewritten by config.

    The `transcript` block serves the transcript-evaluation surface everything it
    would otherwise hardcode: the engines it compares, the recorder's sample
    rate, the chunk-interval bounds, and the script controls' options. None of
    those are literals in the frontend — they are .env values, so moving
    environments stays an .env-only change there too.

    Read from the process-wide `settings` captured at import, so changing an
    engine's credentials in .env takes effect on restart — the same contract as
    every other setting here.
    """
    return {
        "pollIntervalMs": settings.frontend_poll_interval_ms,
        "defaultTranscriptionMode": DEFAULT_TRANSCRIPTION_MODE,
        "transcriptionModes": {
            mode: {
                "asrName": ASR_ENGINES[asr_id].name,
                "configured": ASR_ENGINES[asr_id].configured(settings),
            }
            for mode, asr_id in MODE_TO_ASR_ID.items()
        },
        "transcript": {
            # One entry per compared engine, in display order. `transport` is
            # here because the UI must label every figure with it: a streaming
            # engine and a chunked one are not measuring the same thing, and the
            # chunk statistics of the two are not comparable at all.
            "engines": [
                {
                    "asrId": asr_id,
                    "name": ASR_ENGINES[asr_id].name,
                    "mode": ASR_ENGINES[asr_id].mode,
                    "transport": transport_for(asr_id),
                    "configured": ASR_ENGINES[asr_id].configured(settings),
                }
                for asr_id in COMPARISON_ASR_IDS
                if asr_id in ASR_ENGINES
            ],
            "recordSampleRate": settings.live_record_sample_rate,
            "recordBlockSamples": settings.live_record_block_samples,
            "chunkIntervalSec": settings.live_chunk_default_sec,
            "chunkIntervalMinSec": settings.live_chunk_min_sec,
            "chunkIntervalMaxSec": settings.live_chunk_max_sec,
            "socketOpenTimeoutSec": settings.live_socket_open_timeout_sec,
            "scriptLengthsMin": settings.script_length_options,
            "scriptLanguageMixes": settings.script_language_mix_options,
            "scriptHardCases": settings.script_hard_case_options,
            # Null when no script gateway is configured: the UI then offers the
            # paste-a-reference path only, instead of a Generate button that
            # cannot work.
            "scriptModel": settings.llm_model if settings.llm_chat_url else None,
        },
        "tts": {
            "engines": [
                {
                    "ttsId": tts_id,
                    "name": TTS_ENGINES[tts_id].name,
                    "delivery": TTS_DELIVERY[tts_id],
                    "configured": TTS_ENGINES[tts_id].configured(settings),
                    "voices": TTS_ENGINES[tts_id].voices(settings),
                    "defaultVoice": TTS_ENGINES[tts_id].default_voice(settings),
                    # Engine-level synthesis settings, true of every voice this
                    # engine offers. The UI shows them beside the voice picker;
                    # neither gateway exposes per-voice metadata (no
                    # list-voices endpoint exists), so nothing here is
                    # per-voice and the UI must not present it as such.
                    "synthesisParams": TTS_ENGINES[tts_id].synthesis_params(settings),
                }
                for tts_id in COMPARISON_TTS_IDS
                if tts_id in TTS_ENGINES
            ],
            "maxInputChars": settings.tts_max_input_chars,
        },
    }
