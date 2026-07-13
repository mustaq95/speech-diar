"""Static per-model container config — the single source of truth for which
models this supervisor manages, mirroring the `LANE_MAP` pattern in
`apps/background_worker/lanes.py`.

`pyannote` is deliberately absent: it runs in-process in the worker (see
`apps/background_worker/models/pyannote/runner.py`), not in a container the
supervisor can start/stop, so it is never subject to the residency cap.
`azure`/`azure-batch` are cloud APIs with no local GPU footprint and are
likewise absent. `vibevoice` drops out of the managed set when
VIBEVOICE_BASEURL points it at a remote inference proxy (see `_registry`).
"""

from dataclasses import dataclass

from packages.config.settings import get_settings


@dataclass(frozen=True)
class ManagedContainer:
    model_id: str
    container_name: str
    health_url: str
    cold_start_timeout_sec: int


def _registry() -> dict[str, ManagedContainer]:
    settings = get_settings()
    registry = {
        "nemo-clustering": ManagedContainer(
            model_id="nemo-clustering",
            container_name="nemo-clustering",
            health_url="http://localhost:9020/health/ready",
            cold_start_timeout_sec=settings.nemo_clustering_timeout_sec,
        ),
        "3d-speaker-clustering": ManagedContainer(
            model_id="3d-speaker-clustering",
            container_name="3d-speaker-clustering",
            health_url="http://localhost:9021/health/ready",
            cold_start_timeout_sec=settings.speaker3d_clustering_timeout_sec,
        ),
        "diarizen": ManagedContainer(
            model_id="diarizen",
            container_name="diarizen",
            health_url="http://localhost:9022/health/ready",
            cold_start_timeout_sec=settings.diarizen_timeout_sec,
        ),
        "vibevoice": ManagedContainer(
            model_id="vibevoice",
            container_name="vibevoice",
            health_url="http://localhost:9023/health/ready",
            cold_start_timeout_sec=settings.vibevoice_cold_start_timeout_sec,
        ),
        "nim-sortformer-str": ManagedContainer(
            model_id="nim-sortformer-str",
            container_name="parakeet-nim-str",
            health_url=f"http://localhost:{settings.nim_str_health_port}/v1/health/ready",
            cold_start_timeout_sec=settings.nim_cold_start_timeout_sec,
        ),
        "nim-sortformer-ofl": ManagedContainer(
            model_id="nim-sortformer-ofl",
            container_name="parakeet-nim-ofl",
            health_url=f"http://localhost:{settings.nim_ofl_health_port}/v1/health/ready",
            cold_start_timeout_sec=settings.nim_cold_start_timeout_sec,
        ),
    }
    if settings.vibevoice_baseurl:
        # Remote inference proxy configured: vibevoice runs off-host, has no
        # local container to manage, and must not consume a GPU residency
        # slot -- treated exactly like the cloud-lane models. Its runner
        # switches to the remote endpoint on the same setting.
        del registry["vibevoice"]
    return registry


def managed_container(model_id: str) -> ManagedContainer | None:
    return _registry().get(model_id)


def is_managed(model_id: str) -> bool:
    return model_id in _registry()


def all_managed_model_ids() -> list[str]:
    return list(_registry().keys())
