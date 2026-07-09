"""Azure Speech diarization — execution code.

Uses the Speech SDK's ConversationTranscriber (real-time diarization streamed
from the audio file). Chosen over the fast-transcription REST API because fast
transcription is not available in every region (e.g. uaenorth), while
ConversationTranscriber works with just a key + region and no blob storage.

Returns a NATIVE payload of raw SDK phrase events; only ``adapter.py``
understands that shape.

Configuration (`.env` at repo root, read via `packages/config/settings.py`):
  AZURE_SPEECH_KEY     — Cognitive Services key (required)
  AZURE_SPEECH_REGION  — resource region, e.g. uaenorth (required)
"""

import threading
import wave
from contextlib import suppress
from typing import Any

from packages.config.settings import get_settings

from ..base_model import ModelRunner

# Native shape produced by this runner:
# {"audio_duration_sec": float | None,
#  "phrases": [{"speaker_id": "Guest-1", "offset_ticks": int, "duration_ticks": int, "text": str}]}
AzureSpeechRawOutput = dict[str, Any]


def _wav_duration_sec(audio_path: str) -> float | None:
    with suppress(Exception):
        with wave.open(audio_path, "rb") as wav:
            return wav.getnframes() / wav.getframerate()
    return None


class AzureSpeechRunner(ModelRunner[AzureSpeechRawOutput]):
    model_id = "azure"

    def run(self, audio_path: str, params: dict[str, Any] | None = None) -> AzureSpeechRawOutput:
        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError as exc:
            raise RuntimeError("pip install azure-cognitiveservices-speech") from exc

        settings = get_settings()
        if not settings.azure_speech_key or not settings.azure_speech_region:
            raise RuntimeError("AZURE_SPEECH_KEY and AZURE_SPEECH_REGION must be set")

        params = params or {}
        config = speechsdk.SpeechConfig(subscription=settings.azure_speech_key, region=settings.azure_speech_region)
        config.speech_recognition_language = params.get("language", settings.locale)
        audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
        transcriber = speechsdk.transcription.ConversationTranscriber(config, audio_config)

        phrases: list[dict[str, Any]] = []
        errors: list[str] = []
        done = threading.Event()

        def on_transcribed(evt: Any) -> None:
            result = evt.result
            if result.reason == speechsdk.ResultReason.RecognizedSpeech and result.text:
                phrases.append(
                    {
                        "speaker_id": result.speaker_id or "Unknown",
                        "offset_ticks": result.offset,
                        "duration_ticks": result.duration,
                        "text": result.text,
                    }
                )

        def on_canceled(evt: Any) -> None:
            details = evt.result.cancellation_details
            # EndOfStream is the normal end-of-file signal, not an error.
            if details.reason != speechsdk.CancellationReason.EndOfStream:
                errors.append(f"{details.reason}: {details.error_details}")
            done.set()

        transcriber.transcribed.connect(on_transcribed)
        transcriber.canceled.connect(on_canceled)
        transcriber.session_stopped.connect(lambda evt: done.set())

        transcriber.start_transcribing_async().get()
        finished = done.wait(settings.azure_realtime_transcribe_timeout_sec)
        transcriber.stop_transcribing_async().get()

        if errors:
            raise RuntimeError(f"Azure transcription failed: {'; '.join(errors)}")
        if not finished:
            raise RuntimeError(f"Azure transcription timed out after {settings.azure_realtime_transcribe_timeout_sec}s")

        return {
            "audio_duration_sec": _wav_duration_sec(audio_path),
            "phrases": phrases,
        }
