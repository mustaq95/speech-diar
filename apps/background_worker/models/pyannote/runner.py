"""PyAnnote-specific execution code.

Wire the real pipeline here (pyannote.audio Pipeline.from_pretrained).
The runner returns PyAnnote's NATIVE Annotation-like output; only
``adapter.py`` is allowed to understand that shape.
"""

from typing import Any

from ..base_model import ModelRunner

# PyAnnote native shape (serialized): [{"start", "end", "label": "SPEAKER_00"}]
PyAnnoteRawOutput = list[dict[str, Any]]


class PyAnnoteRunner(ModelRunner[PyAnnoteRawOutput]):
    model_id = "pyannote"
    available = False

    def run(self, audio_path: str, params: dict[str, Any] | None = None) -> PyAnnoteRawOutput:
        raise NotImplementedError(
            "Install pyannote.audio and implement: pipeline(audio_path) -> "
            "[{'start': t.start, 'end': t.end, 'label': label} "
            "for t, _, label in annotation.itertracks(yield_label=True)]"
        )
