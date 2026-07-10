#!/usr/bin/env bash
# Build and run the NeMo Clustering Diarizer container (MarbleNet VAD +
# TitaNet + spectral clustering) on this GPU host.
#
# Unlike the Parakeet NIMs (deploy/parakeet/), there is no turnkey NVIDIA NIM
# for the cascaded clustering pipeline, so this builds a custom image from
# the official NGC NeMo container (nvcr.io/nvidia/nemo:26.02) -- the first
# build pulls that image (~20GB) and can take a while.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$ROOT/nemo-clustering.env"
COMPOSE="docker compose -f $ROOT/docker-compose.nemo-clustering.yml"

die() { echo "ERROR: $*" >&2; exit 1; }

wait_for_ready() {
  local url="$1"
  local attempts="${2:-80}"
  echo "==> Waiting for nemo-clustering on $url ..."
  for _ in $(seq 1 "$attempts"); do
    if curl -sf "$url" 2>/dev/null | grep -q ready; then
      echo "nemo-clustering is ready."
      return 0
    fi
    sleep 15
  done
  return 1
}

if [[ ! -f "$ENV_FILE" ]]; then
  cp "$ROOT/nemo-clustering.env.example" "$ENV_FILE"
  echo "Created $ENV_FILE from the example (defaults are fine to start)."
fi

# shellcheck disable=SC1090
set -a && source "$ENV_FILE" && set +a

if [[ -n "${NGC_API_KEY:-}" ]]; then
  echo "==> Logging in to nvcr.io..."
  echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
fi

echo "==> Building nemo-clustering image (pulls nvcr.io/nvidia/nemo:26.02 on first build)..."
$COMPOSE build

echo "==> Starting nemo-clustering..."
$COMPOSE up -d

PORT="${NEMO_CLUSTERING_PORT:-9020}"
if wait_for_ready "http://localhost:${PORT}/health/ready" 80; then
  echo ""
  echo "nemo-clustering ready: http://localhost:${PORT}"
  echo "Test: curl -F file=@sample.wav http://localhost:${PORT}/diarize"
else
  echo "Timed out waiting for readiness. Check logs:"
  echo "  $COMPOSE logs -f nemo-clustering"
  exit 1
fi
