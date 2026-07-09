"""Model registry: every pluggable diarization model, keyed by id.

Adding a new engine = new folder with runner.py + adapter.py, one line here.
Nothing else in the platform changes.
"""

from .azure_batch.adapter import AzureBatchAdapter
from .azure_batch.runner import AzureBatchRunner
from .azure_speech.adapter import AzureSpeechAdapter
from .azure_speech.runner import AzureSpeechRunner
from .base_model import DiarizationModel
from .pyannote.adapter import PyAnnoteAdapter
from .pyannote.runner import PyAnnoteRunner
from .whisperx.adapter import WhisperXAdapter
from .whisperx.runner import WhisperXRunner

REGISTRY: dict[str, DiarizationModel] = {
    model.model_id: model
    for model in (
        DiarizationModel(AzureSpeechRunner(), AzureSpeechAdapter()),
        DiarizationModel(AzureBatchRunner(), AzureBatchAdapter()),
        DiarizationModel(WhisperXRunner(), WhisperXAdapter()),
        DiarizationModel(PyAnnoteRunner(), PyAnnoteAdapter()),
    )
}
