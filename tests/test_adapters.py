"""Unit tests for model adapters: pure functions mapping each engine's native
output to the shared contract. No merging, no fabrication — one segment per
raw phrase/turn the engine actually reported."""

from apps.background_worker.models import _nim_shared
from apps.background_worker.models.azure_batch.adapter import AzureBatchAdapter
from apps.background_worker.models.azure_speech.adapter import AzureSpeechAdapter
from apps.background_worker.models.diarizen.adapter import DiarizenAdapter
from apps.background_worker.models.nemo_clustering.adapter import NemoClusteringAdapter
from apps.background_worker.models.nim_sortformer_ofl.adapter import NimSortformerOflAdapter
from apps.background_worker.models.nim_sortformer_str.adapter import NimSortformerStrAdapter
from apps.background_worker.models.pyannote.adapter import PyAnnoteAdapter
from apps.background_worker.models.sherpa.adapter import SherpaAdapter
from apps.background_worker.models.speaker3d_clustering.adapter import Speaker3dClusteringAdapter
from apps.background_worker.models.vibevoice.adapter import VibeVoiceAdapter


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


def test_pyannote_adapter_maps_tracks_and_rebases_speakers_by_first_appearance() -> None:
    raw = [
        {"start": 0.0, "end": 0.4, "label": "SPEAKER_01"},
        {"start": 0.5, "end": 0.7, "label": "SPEAKER_00"},
        {"start": 0.8, "end": 1.0, "label": "SPEAKER_01"},
    ]
    run = PyAnnoteAdapter().adapt(raw)
    assert run.id == "pyannote"
    assert run.num_spk == 2
    assert [seg.spk for seg in run.segs] == [0, 1, 0]  # first-appearance order, not label sort order
    assert run.segs[0].s == 0.0 and run.segs[0].e == 0.4


def test_pyannote_adapter_no_merging_of_adjacent_same_speaker_tracks() -> None:
    raw = [
        {"start": 0.0, "end": 1.0, "label": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "label": "SPEAKER_00"},
    ]
    run = PyAnnoteAdapter().adapt(raw)
    assert len(run.segs) == 2


def test_sherpa_adapter_maps_segments_and_rebases_speakers_by_first_appearance() -> None:
    raw = [
        {"start": 0.0, "end": 0.4, "speaker": 1},
        {"start": 0.5, "end": 0.7, "speaker": 0},
        {"start": 0.8, "end": 1.0, "speaker": 1},
    ]
    run = SherpaAdapter().adapt(raw)
    assert run.id == "sherpa"
    assert run.num_spk == 2
    assert [seg.spk for seg in run.segs] == [0, 1, 0]  # first-appearance order, not label sort order
    assert run.segs[0].s == 0.0 and run.segs[0].e == 0.4


def test_sherpa_adapter_no_merging_of_adjacent_same_speaker_segments() -> None:
    raw = [
        {"start": 0.0, "end": 1.0, "speaker": 0},
        {"start": 1.0, "end": 2.0, "speaker": 0},
    ]
    run = SherpaAdapter().adapt(raw)
    assert len(run.segs) == 2


def test_static_metadata_present_for_get_models_registry() -> None:
    for adapter in (
        AzureSpeechAdapter(),
        AzureBatchAdapter(),
        PyAnnoteAdapter(),
        SherpaAdapter(),
        NimSortformerOflAdapter(),
        NimSortformerStrAdapter(),
        NemoClusteringAdapter(),
        Speaker3dClusteringAdapter(),
        DiarizenAdapter(),
        VibeVoiceAdapter(),
    ):
        assert adapter.name and adapter.short and adapter.description


def test_nim_group_words_into_turns_groups_consecutive_same_speaker() -> None:
    words = [
        {"speaker_tag": 0, "start_ms": 0, "end_ms": 200, "word": "hi"},
        {"speaker_tag": 0, "start_ms": 200, "end_ms": 400, "word": "there"},
        {"speaker_tag": 1, "start_ms": 500, "end_ms": 700, "word": "hey"},
    ]
    turns = _nim_shared.group_words_into_turns(words)
    assert turns == [(0, 0.0, 0.4), (1, 0.5, 0.7)]


def test_nim_group_words_into_turns_never_recoalesces_separated_runs() -> None:
    """Speaker 0 appears again after speaker 1 -- must stay two separate
    turns, not merge back into the earlier speaker-0 run (never coalesce)."""
    words = [
        {"speaker_tag": 0, "start_ms": 0, "end_ms": 100, "word": "a"},
        {"speaker_tag": 1, "start_ms": 100, "end_ms": 200, "word": "b"},
        {"speaker_tag": 0, "start_ms": 200, "end_ms": 300, "word": "c"},
    ]
    turns = _nim_shared.group_words_into_turns(words)
    assert [tag for tag, _, _ in turns] == [0, 1, 0]
    assert len(turns) == 3


def test_nim_group_words_into_turns_sorts_out_of_order_words() -> None:
    words = [
        {"speaker_tag": 1, "start_ms": 500, "end_ms": 700, "word": "b"},
        {"speaker_tag": 0, "start_ms": 0, "end_ms": 200, "word": "a"},
    ]
    turns = _nim_shared.group_words_into_turns(words)
    assert [s for _, s, _ in turns] == [0.0, 0.5]


def test_nim_group_words_into_turns_empty_input() -> None:
    assert _nim_shared.group_words_into_turns([]) == []


def test_nim_sortformer_ofl_adapter_maps_turns_and_rebases_speakers() -> None:
    raw = {
        "audio_duration_sec": 1.0,
        "words": [
            {"speaker_tag": 5, "start_ms": 0, "end_ms": 200, "word": "hi"},
            {"speaker_tag": 5, "start_ms": 200, "end_ms": 400, "word": "there"},
            {"speaker_tag": 2, "start_ms": 500, "end_ms": 700, "word": "hey"},
        ],
    }
    run = NimSortformerOflAdapter().adapt(raw)
    assert run.id == "nim-sortformer-ofl"
    assert run.num_spk == 2
    assert [seg.spk for seg in run.segs] == [0, 1]  # first-appearance order, not raw tag values
    assert run.segs[0].s == 0.0 and run.segs[0].e == 0.4
    assert NimSortformerOflAdapter().audio_duration_sec(raw) == 1.0


def test_nim_sortformer_str_adapter_maps_turns_and_rebases_speakers() -> None:
    raw = {
        "audio_duration_sec": 1.0,
        "words": [
            {"speaker_tag": 0, "start_ms": 0, "end_ms": 200, "word": "hi"},
            {"speaker_tag": 1, "start_ms": 200, "end_ms": 400, "word": "there"},
        ],
    }
    run = NimSortformerStrAdapter().adapt(raw)
    assert run.id == "nim-sortformer-str"
    assert run.num_spk == 2
    assert [seg.spk for seg in run.segs] == [0, 1]


def test_nemo_clustering_adapter_parses_rttm_and_rebases_speakers_by_first_appearance() -> None:
    raw = {
        "audio_duration_sec": 10.0,
        "rttm": (
            "SPEAKER rec 1 0.00 2.00 <NA> <NA> speaker_3 <NA> <NA>\n"
            "SPEAKER rec 1 2.00 3.00 <NA> <NA> speaker_1 <NA> <NA>\n"
            "SPEAKER rec 1 5.00 1.50 <NA> <NA> speaker_3 <NA> <NA>\n"
        ),
    }
    run = NemoClusteringAdapter().adapt(raw)
    assert run.id == "nemo-clustering"
    assert run.num_spk == 2
    assert [seg.spk for seg in run.segs] == [0, 1, 0]  # speaker_3 seen first, despite label ">" speaker_1
    assert run.segs[0].s == 0.0 and run.segs[0].e == 2.0
    assert NemoClusteringAdapter().audio_duration_sec(raw) == 10.0


def test_nemo_clustering_adapter_no_merging_of_adjacent_same_speaker_lines() -> None:
    raw = {
        "rttm": (
            "SPEAKER rec 1 0.00 1.00 <NA> <NA> speaker_0 <NA> <NA>\n"
            "SPEAKER rec 1 1.00 1.00 <NA> <NA> speaker_0 <NA> <NA>\n"
        )
    }
    run = NemoClusteringAdapter().adapt(raw)
    assert len(run.segs) == 2


def test_nemo_clustering_adapter_empty_rttm_yields_no_segments() -> None:
    run = NemoClusteringAdapter().adapt({"rttm": ""})
    assert run.segs == []
    assert run.num_spk == 0


def test_speaker3d_clustering_adapter_parses_rttm_and_rebases_speakers_by_first_appearance() -> None:
    raw = {
        "audio_duration_sec": 10.0,
        "rttm": (
            "SPEAKER rec 0 0.00 2.00 <NA> <NA> 3 <NA> <NA>\n"
            "SPEAKER rec 0 2.00 3.00 <NA> <NA> 1 <NA> <NA>\n"
            "SPEAKER rec 0 5.00 1.50 <NA> <NA> 3 <NA> <NA>\n"
        ),
    }
    run = Speaker3dClusteringAdapter().adapt(raw)
    assert run.id == "3d-speaker-clustering"
    assert run.num_spk == 2
    assert [seg.spk for seg in run.segs] == [0, 1, 0]  # cluster "3" seen first, despite label "1" < "3"
    assert run.segs[0].s == 0.0 and run.segs[0].e == 2.0
    assert Speaker3dClusteringAdapter().audio_duration_sec(raw) == 10.0


def test_speaker3d_clustering_adapter_no_merging_of_adjacent_same_speaker_lines() -> None:
    raw = {
        "rttm": (
            "SPEAKER rec 0 0.00 1.00 <NA> <NA> 0 <NA> <NA>\n"
            "SPEAKER rec 0 1.00 1.00 <NA> <NA> 0 <NA> <NA>\n"
        )
    }
    run = Speaker3dClusteringAdapter().adapt(raw)
    assert len(run.segs) == 2


def test_speaker3d_clustering_adapter_empty_rttm_yields_no_segments() -> None:
    run = Speaker3dClusteringAdapter().adapt({"rttm": ""})
    assert run.segs == []
    assert run.num_spk == 0


def test_diarizen_adapter_parses_rttm_and_rebases_speakers_by_first_appearance() -> None:
    raw = {
        "audio_duration_sec": 10.0,
        "rttm": (
            "SPEAKER rec 1 0.00 2.00 <NA> <NA> speaker_3 <NA> <NA>\n"
            "SPEAKER rec 1 2.00 3.00 <NA> <NA> speaker_1 <NA> <NA>\n"
            "SPEAKER rec 1 5.00 1.50 <NA> <NA> speaker_3 <NA> <NA>\n"
        ),
    }
    run = DiarizenAdapter().adapt(raw)
    assert run.id == "diarizen"
    assert run.num_spk == 2
    assert [seg.spk for seg in run.segs] == [0, 1, 0]  # speaker_3 seen first, despite label ">" speaker_1
    assert run.segs[0].s == 0.0 and run.segs[0].e == 2.0
    assert DiarizenAdapter().audio_duration_sec(raw) == 10.0


def test_diarizen_adapter_no_merging_of_adjacent_same_speaker_lines() -> None:
    raw = {
        "rttm": (
            "SPEAKER rec 1 0.00 1.00 <NA> <NA> speaker_0 <NA> <NA>\n"
            "SPEAKER rec 1 1.00 1.00 <NA> <NA> speaker_0 <NA> <NA>\n"
        )
    }
    run = DiarizenAdapter().adapt(raw)
    assert len(run.segs) == 2


def test_diarizen_adapter_empty_rttm_yields_no_segments() -> None:
    run = DiarizenAdapter().adapt({"rttm": ""})
    assert run.segs == []
    assert run.num_spk == 0


def test_vibevoice_adapter_drops_transcript_and_rebases_speakers_by_first_appearance() -> None:
    # VibeVoice emits speaker ids as generated *text*, so they need not start
    # at 0 or run contiguously — hence the rebasing, and hence a fixture that
    # starts at 2. `Content` is dropped: the contract is diarization-only.
    raw = {
        "audio_duration_sec": 10.0,
        "segments": [
            {"Start": 0.0, "End": 2.0, "Speaker": 2, "Content": "Hello there."},
            {"Start": 2.0, "End": 5.0, "Speaker": 1, "Content": "Hi Alex."},
            {"Start": 5.0, "End": 6.5, "Speaker": 2, "Content": "Shall we begin?"},
        ],
    }
    run = VibeVoiceAdapter().adapt(raw)
    assert run.id == "vibevoice"
    assert run.num_spk == 2
    assert [seg.spk for seg in run.segs] == [0, 1, 0]  # Speaker 2 seen first, so it becomes 0
    assert run.segs[0].s == 0.0 and run.segs[0].e == 2.0
    assert VibeVoiceAdapter().audio_duration_sec(raw) == 10.0


def test_vibevoice_adapter_no_merging_of_adjacent_same_speaker_segments() -> None:
    raw = {
        "segments": [
            {"Start": 0.0, "End": 1.0, "Speaker": 0, "Content": "One."},
            {"Start": 1.0, "End": 2.0, "Speaker": 0, "Content": "Two."},
        ],
    }
    run = VibeVoiceAdapter().adapt(raw)
    assert len(run.segs) == 2


def test_vibevoice_adapter_empty_segments_yields_no_segments() -> None:
    run = VibeVoiceAdapter().adapt({"segments": []})
    assert run.segs == []
    assert run.num_spk == 0


def test_vibevoice_adapter_skips_speakerless_non_speech_events() -> None:
    # Verbatim tail of a real run against tests/samples/katiesteve.wav: the
    # model tags non-speech intervals ([Silence], [Music], [Noise], ...) rather
    # than hallucinating words over them, and those entries carry no "Speaker"
    # key at all. They are not speech turns, so they produce no segment — and
    # must not mint a speaker index.
    raw = {
        "audio_duration_sec": 29.4875,
        "segments": [
            {"Start": 0.0, "End": 1.5, "Speaker": 0, "Content": "Good morning, Steve."},
            {"Start": 1.0, "End": 11.41, "Speaker": 1, "Content": "Good morning, Katie."},
            {"Start": 24.09, "End": 29.49, "Content": "[Silence]"},
        ],
    }
    run = VibeVoiceAdapter().adapt(raw)
    assert len(run.segs) == 2
    assert run.num_spk == 2  # the [Silence] event is not a third speaker
