"""`docker` CLI wrapper for starting/stopping already-provisioned model
containers.

Shells out to the `docker` CLI rather than adding a `docker-py` dependency,
matching this repo's existing all-bash deploy story (`deploy/*/*_up.sh`).
The worker runs as a bare host process (not itself containerized), so
`docker` is directly on PATH with no socket-mounting needed.

Every argument passed to `docker` is either a hardcoded subcommand or a
`container_name` sourced from `registry.py`'s static dict — never user
input, never database-sourced free text — and every call uses
`subprocess.run([...], shell=False)`, so there is no shell-injection
surface.

This module only ever `start`s or `stop`s a container that already exists
(created once by the corresponding `deploy/<model>/*_up.sh` script via
`docker compose up`). It never creates, removes, or reconfigures a
container.
"""

import logging
import subprocess
import time

import httpx

from packages.config.settings import get_settings

logger = logging.getLogger(__name__)

_DOCKER_CLI_TIMEOUT_SEC = 30
_HEALTH_POLL_INTERVAL_SEC = 5


class ContainerNotReadyError(Exception):
    """Raised when a container fails to report healthy within its cold-start budget."""


def _run(args: list[str], timeout: int = _DOCKER_CLI_TIMEOUT_SEC) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        shell=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )


def start_container(container_name: str) -> None:
    logger.info("docker start %s", container_name)
    _run(["docker", "start", container_name])


def stop_container(container_name: str) -> None:
    grace = get_settings().container_stop_grace_sec
    logger.info("docker stop --time %s %s", grace, container_name)
    # The CLI call must OUTLIVE the container's own graceful-shutdown window:
    # `docker stop --time N` lets the container take up to N seconds before
    # SIGKILL, and a heavyweight NIM/Triton container genuinely uses all of
    # it — with a flat CLI timeout equal to the grace, the client timed out
    # at the same instant the stop would have completed, and the
    # TimeoutExpired killed the evicting job (caught live: DiariZen and
    # VibeVoice stuck "Queued" forever after trying to evict parakeet).
    _run(["docker", "stop", "--time", str(grace), container_name], timeout=grace + _DOCKER_CLI_TIMEOUT_SEC)


def running_containers() -> set[str]:
    """Names of all currently-running containers, in one `docker ps` call —
    admission and status derivation need the running set for every managed
    model at once, and one subprocess beats six `docker inspect`s. Callers
    intersect with the static registry's names; unmanaged containers in the
    result are ignored, not an error."""
    try:
        result = _run(["docker", "ps", "--format", "{{.Names}}"])
    except (subprocess.CalledProcessError, subprocess.SubprocessError):
        logger.exception("docker ps failed; treating all containers as not running")
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def is_healthy(health_url: str) -> bool:
    try:
        response = httpx.get(health_url, timeout=5.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def ensure_ready(health_url: str, timeout_sec: int) -> None:
    """Block the calling job until `health_url` reports healthy, or raise
    `ContainerNotReadyError` once `timeout_sec` elapses. Called OUTSIDE any
    DB session, mirroring how `model.runner.run()` is already outside one
    in `pipelines/local_pipeline.py` — blocks only the one job actually
    about to use this model's slot, never a whole worker's DB connection.
    """
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if is_healthy(health_url):
            return
        time.sleep(_HEALTH_POLL_INTERVAL_SEC)
    raise ContainerNotReadyError(f"{health_url} did not become healthy within {timeout_sec}s")
