web: uv run uvicorn apps.backend_api.main:app --port 8010
worker: ./scripts/run_workers.sh
supervisor: uv run python -m apps.background_worker.supervisor.daemon
recording: uv run uvicorn tools.recording_api.server:app --port 8215
# The cohere-transcribe container is deliberately NOT here any more. It is
# `docker compose up -d` infrastructure now, beside postgres/redis/minio, via the
# `include` at the top of docker-compose.yml.
#
# Honcho used to own it, and that ownership is what put "Not configured: Cohere"
# on the Transcript surface at every boot: honcho stops the container on
# shutdown, so each `honcho start` began from a cold container that needs ~110s
# to load the model, while the API answers /config in milliseconds. The engine
# was therefore genuinely unconfigured for the first two minutes of every run.
#
# Under docker compose with `restart: unless-stopped` it simply stays up across
# app restarts and reboots, so it is warm before the API asks. Ctrl+C on honcho
# no longer takes the GPU container down with it.
#
# deploy/cohere-transcribe/cohere_transcribe_honcho.sh is left in place for a host
# that would rather tie the container's lifetime to the app; it is not referenced
# from here. Running both owners at once is a container-name conflict that kills
# the whole honcho formation, so pick one.
