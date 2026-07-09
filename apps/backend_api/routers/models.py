"""GET /models — the honest engine registry.

Unimplemented engines (pyannote, whisperx) are listed but `available: false`;
the frontend disables them instead of hiding them or faking output.
"""

from fastapi import APIRouter

from apps.background_worker.models import REGISTRY
from packages.shared_contracts.schemas import ModelMetadata

router = APIRouter(tags=["models"])


@router.get("/models", response_model=list[ModelMetadata], response_model_by_alias=True)
def list_models() -> list[ModelMetadata]:
    return [
        ModelMetadata(
            id=model.model_id,
            name=model.adapter.name,
            short=model.adapter.short,
            description=model.adapter.description,
            available=model.available,
        )
        for model in REGISTRY.values()
    ]
