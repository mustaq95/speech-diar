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


def test_unimplemented_engines_are_listed_but_marked_unavailable(client: TestClient) -> None:
    response = client.get("/models")
    body = {entry["id"]: entry for entry in response.json()}
    assert body["whisperx"]["available"] is False
    assert body["pyannote"]["available"] is True
    assert body["azure-batch"]["available"] is True
