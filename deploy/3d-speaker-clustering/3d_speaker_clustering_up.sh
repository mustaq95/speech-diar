#!/usr/bin/env bash
# Build and run the 3D-Speaker CAM++ Clustering Diarizer container (FSMN VAD
# + CAM++ speaker embeddings + clustering, no ASR/transcription step) on
# this GPU host.
#
# No turnkey NIM or vendor image exists for this pipeline, so this builds a
# custom image by cloning modelscope/3D-Speaker at a pinned commit -- the
# first build pulls the base PyTorch image and installs 3D-Speaker's deps,
# and the first request downloads the ModelScope VAD/CAM++ checkpoints, so
# both the first build and the first /diarize call can take a while.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$ROOT/3d-speaker-clustering.env"
COMPOSE="docker compose -f $ROOT/docker-compose.3d-speaker-clustering.yml"

wait_for_ready() {
  local url="$1"
  local attempts="${2:-80}"
  echo "==> Waiting for 3d-speaker-clustering on $url ..."
  for _ in $(seq 1 "$attempts"); do
    if curl -sf "$url" 2>/dev/null | grep -q ready; then
      echo "3d-speaker-clustering is ready."
      return 0
    fi
    sleep 15
  done
  return 1
}

if [[ ! -f "$ENV_FILE" ]]; then
  cp "$ROOT/3d-speaker-clustering.env.example" "$ENV_FILE"
  echo "Created $ENV_FILE from the example (defaults are fine to start)."
fi

# shellcheck disable=SC1090
set -a && source "$ENV_FILE" && set +a

echo "==> Building 3d-speaker-clustering image (clones modelscope/3D-Speaker on first build)..."
$COMPOSE build

echo "==> Starting 3d-speaker-clustering..."
$COMPOSE up -d

PORT="${SPEAKER3D_CLUSTERING_PORT:-9021}"
if wait_for_ready "http://localhost:${PORT}/health/ready" 80; then
  echo ""
  echo "3d-speaker-clustering ready: http://localhost:${PORT}"
  echo "Test: curl -F file=@sample.wav http://localhost:${PORT}/diarize"
else
  echo "Timed out waiting for readiness. Check logs:"
  echo "  $COMPOSE logs -f 3d-speaker-clustering"
  exit 1
fi
