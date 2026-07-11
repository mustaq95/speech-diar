"""The single source of truth for which storage lane each model belongs to.

Local lane = MinIO, run by `pipelines/local_pipeline.py`.
Azure lane = Blob, run by `pipelines/azure_pipeline.py`, and only ever used
for `azure-batch` — the one engine that is URL/Blob-based. No model may
belong to both lanes; nothing here ever copies bytes between the two stores.
"""

from typing import Literal

Lane = Literal["local", "azure"]

LANE_MAP: dict[str, Lane] = {
    "pyannote": "local",
    "whisperx": "local",
    "azure": "local",
    "azure-batch": "azure",
    "nim-sortformer-str": "local",
    "nim-sortformer-ofl": "local",
    "nemo-clustering": "local",
    "3d-speaker-clustering": "local",
    "diarizen": "local",
}


def lane_for(model_id: str) -> Lane | None:
    return LANE_MAP.get(model_id)


def split_by_lane(model_ids: list[str]) -> tuple[list[str], list[str]]:
    """Split requested model ids into (local_ids, azure_ids), preserving order.

    Unknown model ids are dropped from both; callers that need to validate
    against the model registry should do so separately.
    """
    local_ids = [m for m in model_ids if LANE_MAP.get(m) == "local"]
    azure_ids = [m for m in model_ids if LANE_MAP.get(m) == "azure"]
    return local_ids, azure_ids
