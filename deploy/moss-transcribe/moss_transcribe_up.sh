#!/usr/bin/env bash
# Run the MOSS-Transcribe-Diarize container (0.9B joint ASR + diarization +
# timestamping) on this GPU host, served through vLLM.
#
# The build is thin -- MOSS is registered in vLLM upstream, so the image only
# adds vLLM's optional audio-decode deps to the stock vllm/vllm-openai base
# (see Dockerfile). The first startup then downloads ~2GB of weights from
# Hugging Face into the moss-transcribe-cache volume and pays vLLM's
# torch.compile cost before the model is ready; both are cached for subsequent
# starts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$ROOT/moss-transcribe.env"
COMPOSE="docker compose -f $ROOT/docker-compose.moss-transcribe.yml"

wait_for_ready() {
  local url="$1"
  local attempts="${2:-80}"
  echo "==> Waiting for moss-transcribe on $url (first run downloads ~2GB of weights) ..."
  for _ in $(seq 1 "$attempts"); do
    if curl -sf -o /dev/null "$url" 2>/dev/null; then
      echo "moss-transcribe is ready."
      return 0
    fi
    sleep 15
  done
  return 1
}

if [[ ! -f "$ENV_FILE" ]]; then
  cp "$ROOT/moss-transcribe.env.example" "$ENV_FILE"
  echo "Created $ENV_FILE from the example (defaults are fine to start)."
fi

# shellcheck disable=SC1090
set -a && source "$ENV_FILE" && set +a

echo "==> Building moss-transcribe image..."
$COMPOSE build

echo "==> Starting moss-transcribe..."
$COMPOSE up -d

PORT="${MOSS_TRANSCRIBE_PORT:-9024}"
if wait_for_ready "http://localhost:${PORT}/health" 80; then
  echo ""
  echo "moss-transcribe ready: http://localhost:${PORT}"
  echo "Test: curl -s http://localhost:${PORT}/v1/audio/transcriptions \\"
  echo "        -F model=moss-transcribe -F file=@tests/samples/katiesteve.wav \\"
  echo "        -F response_format=json -F temperature=0 -F max_completion_tokens=65536"
else
  echo "Timed out waiting for readiness. Check logs:"
  echo "  $COMPOSE logs -f moss-transcribe"
  exit 1
fi
