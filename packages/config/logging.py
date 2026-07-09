"""Central logging setup — call once per process (API startup, worker entrypoint).

Plain `logging.basicConfig` so behavior is identical under uvicorn, `rq
worker`, or a bare script. Idempotent: does nothing if the root logger
already has a handler (e.g. the `rq worker` CLI configures its own before
importing job code) so this never double-attaches handlers or clobbers
another entrypoint's setup.
"""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    if logging.root.handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
