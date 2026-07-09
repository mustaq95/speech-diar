"""Unit tests for packages/shared_contracts/schemas.py.

normalize_model_run must only sort segments and recompute num_spk — it must
never merge or drop a segment a model actually produced (that's the whole
point of the "raw segments" requirement for KPI evaluation).
"""

from packages.shared_contracts.schemas import DiarizationModelRun, DiarizationSegment, normalize_model_run


def _run(segs: list[tuple[int, float, float]]) -> DiarizationModelRun:
    return DiarizationModelRun(
        id="test",
        name="Test Model",
        short="Test",
        description="unit test fixture",
        segs=[DiarizationSegment(spk=spk, s=s, e=e) for spk, s, e in segs],
    )


def test_normalize_sorts_by_start_time() -> None:
    run = _run([(0, 10.0, 20.0), (1, 0.0, 5.0), (0, 5.0, 10.0)])
    normalized = normalize_model_run(run)
    starts = [seg.s for seg in normalized.segs]
    assert starts == sorted(starts)


def test_normalize_recomputes_num_spk_from_max_speaker_index() -> None:
    run = _run([(0, 0.0, 1.0), (2, 1.0, 2.0)])
    normalized = normalize_model_run(run)
    assert normalized.num_spk == 3  # max spk (2) + 1, even though spk=1 never appears


def test_normalize_never_merges_adjacent_same_speaker_segments() -> None:
    """Two same-speaker segments 0.1s apart must both survive — this is the
    KPI requirement: raw model turn-taking, not human-friendly merged turns."""
    run = _run([(0, 0.0, 1.0), (0, 1.1, 2.0)])
    normalized = normalize_model_run(run)
    assert len(normalized.segs) == 2
    assert [(seg.s, seg.e) for seg in normalized.segs] == [(0.0, 1.0), (1.1, 2.0)]


def test_normalize_does_not_drop_zero_length_or_overlapping_segments() -> None:
    run = _run([(0, 1.0, 1.0), (1, 0.5, 1.5)])
    normalized = normalize_model_run(run)
    assert len(normalized.segs) == 2


def test_normalize_empty_segments_gives_zero_speakers() -> None:
    normalized = normalize_model_run(_run([]))
    assert normalized.segs == []
    assert normalized.num_spk == 0


def test_wire_format_is_camel_case() -> None:
    run = _run([(0, 0.0, 1.0)])
    dumped = run.model_dump(by_alias=True, mode="json")
    assert "numSpk" in dumped
    assert "num_spk" not in dumped
    assert dumped["segs"][0]["spk"] == 0


def test_round_trip_preserves_snake_case_python_access() -> None:
    payload = {"id": "x", "name": "X", "short": "X", "description": "d", "segs": [], "numSpk": 0}
    run = DiarizationModelRun.model_validate(payload)
    assert run.num_spk == 0  # snake_case attribute access in Python
