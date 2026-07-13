web: uv run uvicorn apps.backend_api.main:app --port 8010
worker: ./scripts/run_workers.sh
supervisor: uv run python -m apps.background_worker.supervisor.daemon
