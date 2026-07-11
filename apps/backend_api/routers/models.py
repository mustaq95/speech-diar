"""GET /models — the honest engine registry.

Every registered engine is currently implemented. An engine that lands
unimplemented is listed with `available: false` rather than hidden, and the
frontend disables it instead of faking output.
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
