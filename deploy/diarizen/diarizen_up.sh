#!/usr/bin/env bash
# Build and run the DiariZen container (WavLM-Large + Conformer local
# end-to-end diarization followed by global clustering) on this GPU host.
#
# No turnkey NIM or vendor image exists for this pipeline, so this builds a
# custom image by cloning BUTSpeechFIT/DiariZen at a pinned commit -- the
# first build pulls the base PyTorch image and installs DiariZen's deps, and
# the first request downloads the pretrained weights from Hugging Face, so
# both the first build and the first /diarize call can take a while.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$ROOT/diarizen.env"
COMPOSE="docker compose -f $ROOT/docker-compose.diarizen.yml"

wait_for_ready() {
  local url="$1"
  local attempts="${2:-80}"
  echo "==> Waiting for diarizen on $url ..."
  for _ in $(seq 1 "$attempts"); do
    if curl -sf "$url" 2>/dev/null | grep -q ready; then
      echo "diarizen is ready."
      return 0
    fi
    sleep 15
  done
  return 1
}

if [[ ! -f "$ENV_FILE" ]]; then
  cp "$ROOT/diarizen.env.example" "$ENV_FILE"
  echo "Created $ENV_FILE from the example (defaults are fine to start)."
fi

# shellcheck disable=SC1090
set -a && source "$ENV_FILE" && set +a

echo "==> Building diarizen image (clones BUTSpeechFIT/DiariZen on first build)..."
$COMPOSE build

echo "==> Starting diarizen..."
$COMPOSE up -d

PORT="${DIARIZEN_PORT:-9022}"
if wait_for_ready "http://localhost:${PORT}/health/ready" 80; then
  echo ""
  echo "diarizen ready: http://localhost:${PORT}"
  echo "Test: curl -F file=@sample.wav http://localhost:${PORT}/diarize"
else
  echo "Timed out waiting for readiness. Check logs:"
  echo "  $COMPOSE logs -f diarizen"
  exit 1
fi
