"""WhisperX-specific execution code.

Wire the real pipeline here (whisperx.load_model → align → assign speakers).
The runner returns WhisperX's NATIVE aligned-segment output; only
``adapter.py`` is allowed to understand that shape.
"""

from typing import Any

from ..base_model import ModelRunner

# WhisperX native shape: {"segments": [{"start", "end", "speaker": "SPEAKER_00", ...}]}
WhisperXRawOutput = dict[str, Any]


class WhisperXRunner(ModelRunner[WhisperXRawOutput]):
    model_id = "whisperx"
    available = False

    def run(self, audio_path: str, params: dict[str, Any] | None = None) -> WhisperXRawOutput:
        raise NotImplementedError(
            "Install whisperx and implement: transcribe -> align -> diarize, "
            "returning the native result dict untouched."
        )
