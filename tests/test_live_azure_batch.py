"""Opt-in live smoke test: runs the real `azure-batch` model against a real,
public two-speaker sample audio file over the actual Azure Speech batch
transcription API.

Skipped by default. To run it (needs AZURE_SPEECH_KEY + AZURE_SPEECH_REGION
or AZURE_SPEECH_ENDPOINT in `.env`):

    RUN_LIVE_TESTS=1 uv run pytest tests/test_live_azure_batch.py -v -m live
"""

import pytest

from apps.background_worker.models import REGISTRY

# Microsoft's own public sample: a real two-person ("Katie" and "Steve")
# conversation used in the Azure Speech SDK docs/samples for diarization.
KATIESTEVE_URL = "https://github.com/Azure-Samples/cognitive-services-speech-sdk/raw/master/sampledata/audiofiles/katiesteve.wav"


@pytest.mark.live
def test_azure_batch_diarizes_real_public_sample_audio() -> None:
    model = REGISTRY["azure-batch"]
    run = model.process(KATIESTEVE_URL)

    assert run.id == "azure-batch"
    assert run.num_spk >= 2  # katiesteve.wav is a real two-speaker conversation
    assert len(run.segs) > 0
    for seg in run.segs:
        assert seg.e > seg.s
