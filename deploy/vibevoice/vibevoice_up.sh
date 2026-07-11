#!/usr/bin/env bash
# Build and run the VibeVoice-ASR container (8B single-pass joint ASR +
# diarization + timestamping) on this GPU host.
#
# The first build pulls the NGC PyTorch base and installs transformers; the
# first *startup* then downloads ~16GB of BF16 weights from Hugging Face into
# the vibevoice-cache volume before the model is ready to serve. Both are
# slow, and unlike the other services here the weights load at startup rather
# than on first request -- so the readiness wait below is generous on purpose.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$ROOT/vibevoice.env"
COMPOSE="docker compose -f $ROOT/docker-compose.vibevoice.yml"

wait_for_ready() {
  local url="$1"
  local attempts="${2:-120}"
  echo "==> Waiting for vibevoice on $url (first run downloads ~16GB of weights) ..."
  for _ in $(seq 1 "$attempts"); do
    if curl -sf "$url" 2>/dev/null | grep -q ready; then
      echo "vibevoice is ready."
      return 0
    fi
    sleep 15
  done
  return 1
}

if [[ ! -f "$ENV_FILE" ]]; then
  cp "$ROOT/vibevoice.env.example" "$ENV_FILE"
  echo "Created $ENV_FILE from the example (defaults are fine to start)."
fi

# shellcheck disable=SC1090
set -a && source "$ENV_FILE" && set +a

echo "==> Building vibevoice image..."
$COMPOSE build

echo "==> Starting vibevoice..."
$COMPOSE up -d

PORT="${VIBEVOICE_PORT:-9023}"
if wait_for_ready "http://localhost:${PORT}/health/ready" 120; then
  echo ""
  echo "vibevoice ready: http://localhost:${PORT}"
  echo "Test: curl -F file=@sample.wav http://localhost:${PORT}/diarize"
else
  echo "Timed out waiting for readiness. Check logs:"
  echo "  $COMPOSE logs -f vibevoice"
  exit 1
fi
