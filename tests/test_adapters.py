"""Unit tests for model adapters: pure functions mapping each engine's native
output to the shared contract. No merging, no fabrication — one segment per
raw phrase/turn the engine actually reported."""

from apps.background_worker.models.azure_batch.adapter import AzureBatchAdapter
from apps.background_worker.models.azure_speech.adapter import AzureSpeechAdapter
from apps.background_worker.models.pyannote.adapter import PyAnnoteAdapter
from apps.background_worker.models.whisperx.adapter import WhisperXAdapter


def test_azure_speech_adapter_maps_speaker_ids_by_first_appearance() -> None:
    ticks_per_sec = 10_000_000
    raw = {
        "audio_duration_sec": 12.0,
        "phrases": [
            {"speaker_id": "Guest-1", "offset_ticks": 0, "duration_ticks": 2 * ticks_per_sec, "text": "hi"},
            {"speaker_id": "Guest-2", "offset_ticks": 2 * ticks_per_sec, "duration_ticks": 3 * ticks_per_sec, "text": "hey"},
            {"speaker_id": "Guest-1", "offset_ticks": 5 * ticks_per_sec, "duration_ticks": 1 * ticks_per_sec, "text": "ok"},
        ],
    }
    run = AzureSpeechAdapter().adapt(raw)
    assert run.id == "azure"
    assert run.num_spk == 2
    assert [seg.spk for seg in run.segs] == [0, 1, 0]
    assert run.segs[1].s == 2.0
    assert run.segs[1].e == 5.0


def test_azure_speech_adapter_sorts_out_of_order_phrases_by_offset() -> None:
    ticks_per_sec = 10_000_000
    raw = {
        "phrases": [
            {"speaker_id": "Guest-2", "offset_ticks": 5 * ticks_per_sec, "duration_ticks": ticks_per_sec, "text": "b"},
            {"speaker_id": "Guest-1", "offset_ticks": 0, "duration_ticks": ticks_per_sec, "text": "a"},
        ],
    }
    run = AzureSpeechAdapter().adapt(raw)
    assert [seg.s for seg in run.segs] == [0.0, 5.0]
    assert run.segs[0].spk == 0  # Guest-1 seen first despite appearing second in the raw list


def test_azure_batch_adapter_rebase_speakers_to_zero_based() -> None:
    ticks_per_sec = 10_000_000
    raw = {
        "recognizedPhrases": [
            {"speaker": 1, "offsetInTicks": 0, "durationInTicks": 2 * ticks_per_sec},
            {"speaker": 2, "offsetInTicks": 2 * ticks_per_sec, "durationInTicks": 2 * ticks_per_sec},
            {"speaker": 1, "offsetInTicks": 4 * ticks_per_sec, "durationInTicks": 1 * ticks_per_sec},
        ]
    }
    run = AzureBatchAdapter().adapt(raw)
    assert run.id == "azure-batch"
    assert run.num_spk == 2
    assert [seg.spk for seg in run.segs] == [0, 1, 0]


def test_azure_batch_adapter_no_merging_of_adjacent_same_speaker_phrases() -> None:
    ticks_per_sec = 10_000_000
    raw = {
        "recognizedPhrases": [
            {"speaker": 1, "offsetInTicks": 0, "durationInTicks": 1 * ticks_per_sec},
            {"speaker": 1, "offsetInTicks": 1 * ticks_per_sec, "durationInTicks": 1 * ticks_per_sec},
        ]
    }
    run = AzureBatchAdapter().adapt(raw)
    assert len(run.segs) == 2


def test_azure_batch_adapter_processing_ms_from_job_timing() -> None:
    raw = {
        "recognizedPhrases": [],
        "_azureJobTiming": {
            "createdDateTime": "2026-01-01T00:00:00Z",
            "lastActionDateTime": "2026-01-01T00:00:05Z",
        },
    }
    assert AzureBatchAdapter().processing_ms(raw) == 5000


def test_azure_batch_adapter_processing_ms_none_when_timing_missing() -> None:
    assert AzureBatchAdapter().processing_ms({"recognizedPhrases": []}) is None


def test_static_metadata_present_for_get_models_registry() -> None:
    for adapter in (AzureSpeechAdapter(), AzureBatchAdapter(), PyAnnoteAdapter(), WhisperXAdapter()):
        assert adapter.name and adapter.short and adapter.description
