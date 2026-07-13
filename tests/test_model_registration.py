"""Consistency checks across the three hand-maintained per-model tables.

Adding a model touches REGISTRY (models/__init__.py), LANE_MAP (lanes.py)
and, for containerized models, the supervisor registry. Each table forgotten
fails silently at runtime (dropped at upload, or run in-process without a
container); these tests turn every mismatch into a loud pytest failure.
"""

from apps.background_worker.lanes import LANE_MAP
from apps.background_worker.models import REGISTRY
from apps.background_worker.supervisor.registry import _registry
from packages.config.settings import get_settings


def test_registry_and_lane_map_list_the_same_models() -> None:
    missing_lane = set(REGISTRY) - set(LANE_MAP)
    missing_registry = set(LANE_MAP) - set(REGISTRY)
    assert not missing_lane, f"In REGISTRY but not LANE_MAP (upload will reject them): {sorted(missing_lane)}"
    assert not missing_registry, f"In LANE_MAP but not REGISTRY (jobs will fail 'Unknown model id'): {sorted(missing_registry)}"


def test_every_managed_container_model_is_a_registered_local_model() -> None:
    for model_id in _registry():
        assert model_id in REGISTRY, f"Supervisor manages {model_id!r} but it is not in REGISTRY"
        assert LANE_MAP.get(model_id) == "local", f"Supervisor-managed {model_id!r} must be in the local lane"


def test_queue_job_timeout_exceeds_every_cold_start_timeout() -> None:
    """Necessary condition only: the full invariant is queue_job_timeout_sec >=
    max(cold-start + inference) across models, but inference timeouts have no
    uniform settings shape — checking that part stays a manual step (see the
    CLAUDE.md recipe)."""
    settings = get_settings()
    for container in _registry().values():
        assert settings.queue_job_timeout_sec > container.cold_start_timeout_sec, (
            f"QUEUE_JOB_TIMEOUT_SEC ({settings.queue_job_timeout_sec}) must exceed "
            f"{container.model_id!r}'s cold-start timeout ({container.cold_start_timeout_sec}) "
            "or RQ kills the job before inference even starts"
        )
