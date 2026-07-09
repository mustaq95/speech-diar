"""Unit tests for apps/background_worker/lanes.py — the sole place that
decides which storage lane (MinIO vs Azure Blob) a model belongs to."""

from apps.background_worker.lanes import lane_for, split_by_lane


def test_local_models_map_to_local_lane() -> None:
    for model_id in ("pyannote", "whisperx", "azure"):
        assert lane_for(model_id) == "local"


def test_azure_batch_maps_to_azure_lane() -> None:
    assert lane_for("azure-batch") == "azure"


def test_unknown_model_has_no_lane() -> None:
    assert lane_for("no-such-model") is None


def test_split_by_lane_separates_and_preserves_order() -> None:
    local_ids, azure_ids = split_by_lane(["pyannote", "azure-batch", "azure", "whisperx"])
    assert local_ids == ["pyannote", "azure", "whisperx"]
    assert azure_ids == ["azure-batch"]


def test_split_by_lane_drops_unknown_ids_from_both() -> None:
    local_ids, azure_ids = split_by_lane(["pyannote", "not-a-model"])
    assert local_ids == ["pyannote"]
    assert azure_ids == []


def test_split_by_lane_empty_input() -> None:
    assert split_by_lane([]) == ([], [])
