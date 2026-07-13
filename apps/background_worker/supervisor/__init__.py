"""GPU container-lifecycle supervisor.

Caps how many local-lane model containers may be GPU-resident (started) at
once on a single, memory-constrained host, queues requests for models
beyond that cap instead of erroring, and unloads idle containers
automatically. Never touches a model with an in-flight job.

- `registry.py` — static per-model container config (the single source of
  truth for which models this supervisor manages).
- `containers.py` — `docker` CLI wrapper: start/stop/health-probe.
- `state.py` — query helpers against the `ModelContainerState` table.
- `admission.py` — `try_acquire`/`release`, called synchronously from
  `pipelines/local_pipeline.py` around each model call.
- `daemon.py` — standalone sweep loop (idle-unload + staleness detection),
  run as its own process via the `supervisor` Procfile line.
"""
