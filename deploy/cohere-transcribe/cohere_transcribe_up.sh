#!/usr/bin/env bash
# Run the Cohere Transcribe Arabic container (2B Arabic/English speech-to-text)
# on this GPU host, served through vLLM. This is the OFFLINE transcription
# engine, selectable per run from the Live Speech panel's online/offline toggle.
#
# The build is thin -- the architecture is registered in vLLM upstream, so the
# image only adds vLLM's optional audio-decode deps to the stock
# vllm/vllm-openai base (see Dockerfile). The first startup then downloads ~4GB
# of weights from Hugging Face into the cohere-transcribe-cache volume and pays
# vLLM's torch.compile cost before the model is ready; both are cached for
# subsequent starts.
#
# Unlike the diarization containers, this one is not GPU-supervisor-managed:
# nothing starts or stops it automatically, so run this once and leave it up.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$ROOT/cohere-transcribe.env"
# The REPO-ROOT compose file, which `include`s this folder's, so this script and
# `docker compose up -d` produce the same container in the same project. Pointing
# it at the local file instead creates a second project that owns a container of
# the same name, and whichever runs second dies on a name conflict.
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
COMPOSE="docker compose -f $REPO_ROOT/docker-compose.yml"
SERVICE="cohere-transcribe"

wait_for_ready() {
  local url="$1"
  local attempts="${2:-80}"
  echo "==> Waiting for cohere-transcribe on $url (first run downloads ~4GB of weights) ..."
  for _ in $(seq 1 "$attempts"); do
    if curl -sf -o /dev/null "$url" 2>/dev/null; then
      echo "cohere-transcribe is ready."
      return 0
    fi
    sleep 15
  done
  return 1
}

if [[ ! -f "$ENV_FILE" ]]; then
  cp "$ROOT/cohere-transcribe.env.example" "$ENV_FILE"
  echo "Created $ENV_FILE from the example."
  echo "The model repo is GATED -- set HF_TOKEN in that file before rerunning."
  exit 1
fi

# shellcheck disable=SC1090
set -a && source "$ENV_FILE" && set +a

if [[ -z "${HF_TOKEN:-}" || "${HF_TOKEN}" == "hf_..." ]]; then
  echo "HF_TOKEN is not set in $ENV_FILE."
  echo "CohereLabs/cohere-transcribe-arabic-07-2026 is a gated repo: accept its"
  echo "conditions on Hugging Face, then paste a read token into that file."
  exit 1
fi

echo "==> Building cohere-transcribe image..."
$COMPOSE build "$SERVICE"

echo "==> Starting cohere-transcribe..."
$COMPOSE up -d "$SERVICE"

PORT="${COHERE_TRANSCRIBE_PORT:-9025}"
if wait_for_ready "http://localhost:${PORT}/health" 80; then
  echo ""
  echo "cohere-transcribe ready: http://localhost:${PORT}"
  echo "Test: curl -s http://localhost:${PORT}/v1/audio/transcriptions \\"
  echo "        -F model=cohere-transcribe -F file=@tests/samples/3-two-speakers-en.wav \\"
  echo "        -F response_format=json -F temperature=0"
else
  echo "Timed out waiting for readiness. Check logs:"
  echo "  $COMPOSE logs -f $SERVICE"
  exit 1
fi
