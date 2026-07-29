"""Transcript adapters: the only code allowed to understand each engine's
native shape, so these are the tests that pin those shapes.

Pure functions, no infrastructure — the same split as `test_adapters.py` does
for the diarization side.
"""

from apps.background_worker.transcription.cohere import adapter as cohere_adapter
from apps.background_worker.transcription.ctc_aligner import adapter as aligner_adapter
from apps.background_worker.transcription.hamsa import adapter as hamsa_adapter


def test_hamsa_adapter_joins_segments_in_arrival_order() -> None:
    """Hamsa emits one message per detected speech segment, in the order the
    audio was consumed — so arrival order IS chronological order."""
    raw = [
        {"type": "handshake_ack"},
        {"type": "transcription", "data": {"transcription": "مرحبا بكم"}},
        {"type": "transcription", "data": {"transcription": "في الاجتماع"}},
    ]
    assert hamsa_adapter.adapt(raw) == "مرحبا بكم في الاجتماع"


def test_hamsa_adapter_accepts_the_flat_payload_shape() -> None:
    """Some deployments put the transcript at the top level rather than under
    `data`; both are real and the reference client tolerates both."""
    raw = [{"type": "transcription", "text": "hello there"}]
    assert hamsa_adapter.adapt(raw) == "hello there"


def test_hamsa_adapter_skips_non_transcription_and_empty_messages() -> None:
    """Status/ack frames carry no speech, and an empty segment must not become
    stray whitespace in the text handed to the aligner."""
    raw = [
        {"type": "handshake_ack"},
        {"type": "status", "data": {"transcription": "ignored"}},
        {"type": "transcription", "data": {"transcription": "   "}},
        {"type": "transcription", "data": {"transcription": "real words"}},
    ]
    assert hamsa_adapter.adapt(raw) == "real words"


def test_hamsa_adapter_returns_empty_string_when_nothing_was_said() -> None:
    """Silence is a legitimate outcome, not a failure."""
    assert hamsa_adapter.adapt([{"type": "handshake_ack"}]) == ""


def test_cohere_adapter_takes_the_text_and_drops_usage() -> None:
    """vLLM's transcription response carries token accounting alongside the
    text; only the speech is part of this platform's contract."""
    raw = {"text": " مرحبا بكم ", "usage": {"type": "duration", "seconds": 1943}}
    assert cohere_adapter.adapt(raw) == "مرحبا بكم"


def test_cohere_adapter_handles_a_missing_text_field() -> None:
    assert cohere_adapter.adapt({}) == ""


def test_aligner_adapter_maps_native_spans_to_contract_words() -> None:
    raw = [
        {"text": "hello", "start": 0.5, "end": 0.9, "score": 0.87},
        {"text": "world", "start": 1.0, "end": 1.4, "score": 0.91},
    ]
    words = aligner_adapter.adapt(raw)

    assert [w.w for w in words] == ["hello", "world"]
    assert words[0].s == 0.5
    assert words[0].e == 0.9
    assert words[0].score == 0.87


def test_aligner_adapter_carries_unaligned_words_instead_of_dropping_them() -> None:
    """A word the aligner could not place keeps its position with no timing.
    Dropping it would silently shorten the transcript; interpolating a time
    would fabricate a measurement. It is carried, unplaced."""
    raw = [
        {"text": "hello", "start": 0.5, "end": 0.9, "score": 0.9},
        {"text": "Kubernetes", "start": None, "end": None, "score": None},
        {"text": "world", "start": 1.0, "end": 1.4, "score": 0.9},
    ]
    words = aligner_adapter.adapt(raw)

    assert [w.w for w in words] == ["hello", "Kubernetes", "world"]
    assert words[1].s is None
    assert words[1].e is None


def test_aligner_adapter_skips_star_gap_markers() -> None:
    """`<star>` is the library's own marker for a gap between aligned regions,
    not a spoken word."""
    raw = [
        {"text": "hello", "start": 0.0, "end": 0.4, "score": 0.9},
        {"text": "<star>", "start": 0.4, "end": 2.0, "score": 0.0},
        {"text": "world", "start": 2.0, "end": 2.4, "score": 0.9},
    ]
    assert [w.w for w in aligner_adapter.adapt(raw)] == ["hello", "world"]


def test_aligner_adapter_on_empty_input() -> None:
    assert aligner_adapter.adapt([]) == []
