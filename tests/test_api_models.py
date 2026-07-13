"""Integration test for GET /models — the honest engine registry."""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from apps.background_worker.models import REGISTRY
from apps.background_worker.supervisor import state
from packages.database.models import ModelContainerState


def test_get_models_lists_every_registered_engine_with_availability(client: TestClient) -> None:
    response = client.get("/models")
    assert response.status_code == 200
    body = {entry["id"]: entry for entry in response.json()}

    assert set(body.keys()) == set(REGISTRY.keys())
    for model_id, model in REGISTRY.items():
        assert body[model_id]["available"] is model.available
        assert body[model_id]["name"] == model.adapter.name
        assert body[model_id]["short"] == model.adapter.short


def test_every_registered_engine_is_available(client: TestClient) -> None:
    """No stubs left in the registry. whisperx was the last engine carrying
    `available = False`, and vibevoice replaced it with a real runner. The
    flag itself stays on ModelRunner — a future engine may land unimplemented,
    and the UI must list-but-disable it rather than hide or fake it."""
    response = client.get("/models")
    body = {entry["id"]: entry for entry in response.json()}
    assert all(entry["available"] for entry in body.values())
    assert body["vibevoice"]["available"] is True
    assert body["pyannote"]["available"] is True
    assert body["azure-batch"]["available"] is True


def test_get_models_status_derives_lifecycle_state_from_docker_and_rq(
    client: TestClient, db_session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The status payload is derived per request from Docker + RQ + the
    claim rows (supervisor/state.py), never read from a stored mirror —
    stub both snapshot sources so the route never touches real
    docker/Redis in the suite."""
    monkeypatch.setattr(state, "live_busy_model_ids", lambda: {"nemo-clustering"})
    monkeypatch.setattr(state.containers, "running_containers", lambda: {"nemo-clustering", "diarizen"})
    monkeypatch.setattr(state, "queued_counts", lambda: {"vibevoice": 2})
    with db_session_factory() as session:
        session.add(ModelContainerState(model_id="nemo-clustering", active_job_count=1))
        session.add(ModelContainerState(model_id="diarizen"))  # running, idle
        session.add(ModelContainerState(model_id="vibevoice", starting_since=datetime.now(timezone.utc)))
        session.commit()

    response = client.get("/models/status")
    assert response.status_code == 200
    body = {entry["modelId"]: entry for entry in response.json()}

    assert body["nemo-clustering"]["state"] == "in_use"
    assert body["nemo-clustering"]["activeJobCount"] == 1
    assert body["diarizen"]["state"] == "ready"  # container running, no jobs
    # vibevoice's claim has no live busy worker behind it -> treated as
    # absent, and its container isn't running -> unloaded, with its queue
    # depth still reported from RQ.
    assert body["vibevoice"]["state"] == "unloaded"
    assert body["vibevoice"]["queuedJobCount"] == 2
    # pyannote has no ModelContainerState row (in-process, no container) --
    # never fabricate a lifecycle state for it.
    assert "pyannote" not in body
