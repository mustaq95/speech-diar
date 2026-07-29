#!/usr/bin/env bash
# Honcho-managed lifecycle for the cohere-transcribe container (the OFFLINE
# transcription engine). Started by the `cohere` line in the repo-root Procfile.
#
# Why this exists alongside cohere_transcribe_up.sh: that script runs
# `docker compose up -d` and EXITS once the container is healthy, which is the
# right shape for a one-off manual start but the wrong shape for honcho.
# Honcho tears down the whole formation when any process exits, so a detached
# starter would kill the API and workers seconds after boot. This script stays
# in the foreground for as long as the container runs.
#
# Lifecycle:
#   honcho start  -> builds if needed, starts the container, streams its logs
#   Ctrl+C / stop -> SIGTERM reaches the trap below, which stops the container
#
# Gated on this folder's own env file being set up (a real HF_TOKEN): the
# offline engine is now selectable per run from the Live Speech panel, so the
# container should be available whenever an operator has configured it. A host
# that never wants offline transcription simply leaves cohere-transcribe.env
# unset, and this script idles (below) instead of holding the GPU.
#
# Note the container keeps `restart: unless-stopped` in its compose file, so it
# recovers from a crash on its own. Because honcho's shutdown path issues an
# explicit `stop`, Docker records it as stopped and will NOT bring it back until
# honcho starts it again. The one case that escapes honcho's control is a Docker
# daemon restart, which resumes it independently.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$ROOT/cohere-transcribe.env"
COMPOSE=(docker compose -f "$ROOT/docker-compose.cohere-transcribe.yml")

# Run only when this container has been configured on this host. Cohere is the
# offline engine; a host that only ever uses online transcription (TryHamsa, a
# remote service that needs nothing local) never creates this env file, and the
# container then stays down so it does not hold ~5GB of GPU that nothing calls.
if [[ ! -f "$ENV_FILE" ]]; then
  echo "[cohere] $ENV_FILE is missing — offline transcription is not configured"
  echo "[cohere] on this host, so the container stays down and the GPU stays free."
  echo "[cohere] To enable it: cp $ROOT/cohere-transcribe.env.example $ENV_FILE"
  echo "[cohere] and set HF_TOKEN (the model repo is gated), then restart honcho."
  # Idle instead of exiting: honcho kills every other process when one exits, so
  # returning here would take the API and workers down with it.
  exec sleep infinity
fi

# shellcheck disable=SC1090
set -a && source "$ENV_FILE" && set +a

if [[ -z "${HF_TOKEN:-}" || "${HF_TOKEN}" == "hf_..." ]]; then
  echo "[cohere] HF_TOKEN is not set in $ENV_FILE — the model repo is gated."
  echo "[cohere] Sleeping so the rest of the stack stays up."
  exec sleep infinity
fi

# Stop the container whenever this process goes away, so `honcho stop` and a
# plain Ctrl+C both leave the GPU free. `docker compose up` handles SIGTERM
# itself, but the explicit stop also covers the case where this script is
# killed harder than compose expects.
cleanup() {
  echo "[cohere] stopping container..."
  "${COMPOSE[@]}" stop >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "[cohere] building image if needed..."
"${COMPOSE[@]}" build

echo "[cohere] starting (first run downloads ~4GB of weights; port ${COHERE_TRANSCRIBE_PORT:-9025})"
# Attached, NOT -d: this is what keeps the process in the foreground for honcho
# and streams vLLM's logs into the honcho output like every other service here.
exec "${COMPOSE[@]}" up --no-recreate
