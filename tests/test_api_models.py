"""Integration test for GET /models — the honest engine registry."""

from fastapi.testclient import TestClient

from apps.background_worker.models import REGISTRY


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
