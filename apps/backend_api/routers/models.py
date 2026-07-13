"""GET /models — the honest engine registry.

Every registered engine is currently implemented. An engine that lands
unimplemented is listed with `available: false` rather than hidden, and the
frontend disables it instead of faking output.

GET /models/status — real-time GPU-residency lifecycle state for every
managed model (see apps.background_worker.supervisor), platform-wide and
independent of any single evaluation. Deliberately a separate endpoint from
GET /evaluations/{id}: that endpoint only polls while ITS OWN models are
in-flight, but a model's residency state can change because of a
completely different evaluation's job — see the GPU-supervisor plan doc.
The payload is DERIVED per request from Docker + RQ + the claim rows
(supervisor/state.py), never read from a stored lifecycle mirror.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from apps.backend_api.dependencies import get_db
from apps.background_worker.models import REGISTRY
from apps.background_worker.supervisor.state import derive_statuses, snapshot_all
from packages.shared_contracts.schemas import ModelContainerStatus, ModelMetadata

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


@router.get("/models/status", response_model=list[ModelContainerStatus], response_model_by_alias=True)
def list_model_status(db: Session = Depends(get_db)) -> list[ModelContainerStatus]:
    return derive_statuses(snapshot_all(db))
