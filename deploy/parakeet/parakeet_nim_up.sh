#!/usr/bin/env bash
# Deploy Parakeet 1.1B RNNT Multilingual NIM on DGX Spark (official NVIDIA path).
# Starts two containers: streaming (mode=str) and offline batch (mode=ofl).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$ROOT/deploy/parakeet-nim.env"
COMPOSE="docker compose -f $ROOT/deploy/docker-compose.parakeet-nim.yml"
STR_CACHE_VOLUME=parakeet-nim-str-cache
OFL_CACHE_VOLUME=parakeet-nim-ofl-cache
LEGACY_CACHE_VOLUME=parakeet-nim-cache

die() { echo "ERROR: $*" >&2; exit 1; }

validate_key() {
  if [[ -z "${NGC_API_KEY:-}" ]]; then
    die "NGC_API_KEY is empty in $ENV_FILE"
  fi
}

ensure_cache_volume_writable() {
  local volume="$1"
  docker volume inspect "$volume" >/dev/null 2>&1 || docker volume create "$volume" >/dev/null
  docker run --rm -v "$volume":/cache alpine chown -R 1000:1000 /cache
}

migrate_and_prune_legacy_cache() {
  if ! docker volume inspect "$LEGACY_CACHE_VOLUME" >/dev/null 2>&1; then
    return 0
  fi

  echo "==> Migrating mode=str cache from legacy volume and pruning incomplete profiles..."
  ensure_cache_volume_writable "$STR_CACHE_VOLUME"

  docker run --rm \
    -v "$LEGACY_CACHE_VOLUME":/old:ro \
    -v "$STR_CACHE_VOLUME":/str \
    alpine sh -ec '
      HUB=models--nim--nvidia--parakeet-1-1b-rnnt-multilingual
      SRC=/old/ngc/hub/$HUB
      DST=/str/ngc/hub/$HUB
      STR_SNAP=dgx-sparkx1-str-multi-silero-sortformer-26.02.3-fp16-zhuvd6eg-q

      if [ ! -d "$SRC" ]; then
        echo "    No legacy model cache found; skipping migration."
        exit 0
      fi

      mkdir -p "$DST/snapshots" "$DST/blobs" "$DST/refs"

      if [ -d "$SRC/snapshots/$STR_SNAP" ]; then
        echo "    Keeping complete profile: $STR_SNAP"
        cp -a "$SRC/snapshots/$STR_SNAP" "$DST/snapshots/"
        if [ -f "$SRC/refs/$STR_SNAP" ]; then
          cp -a "$SRC/refs/$STR_SNAP" "$DST/refs/"
        fi
        for blob in \
          1e2d983dca10597fbb9061f8d40c6db2-13 \
          9a7f36cd95b4d533d04675930b3bcc53 \
          70ed85888b944b5fce7a42f9ecb9d065; do
          if [ -f "$SRC/blobs/$blob" ]; then
            cp -a "$SRC/blobs/$blob" "$DST/blobs/"
          fi
        done
      else
        echo "    No complete mode=str snapshot in legacy cache."
      fi

      echo "    Dropping incomplete profiles (str-indic), str-thr, and download temp files."
      chown -R 1000:1000 /str
    '

  echo "==> Removing legacy cache volume ($LEGACY_CACHE_VOLUME)..."
  docker volume rm "$LEGACY_CACHE_VOLUME" >/dev/null 2>&1 || true
}

wait_for_ready() {
  local name="$1"
  local url="$2"
  local attempts="${3:-120}"
  echo "==> Waiting for $name on $url ..."
  for _ in $(seq 1 "$attempts"); do
    if curl -sf "$url" 2>/dev/null | grep -q ready; then
      echo "$name is ready."
      return 0
    fi
    sleep 15
  done
  return 1
}

if [[ ! -f "$ENV_FILE" ]]; then
  cp "$ROOT/deploy/parakeet-nim.env.example" "$ENV_FILE"
  die "Created $ENV_FILE — set NGC_API_KEY, then re-run this script."
fi

# shellcheck disable=SC1090
set -a && source "$ENV_FILE" && set +a
validate_key

echo "==> Stopping legacy single-container deployment (if present)..."
docker stop parakeet-nim >/dev/null 2>&1 || true
docker rm parakeet-nim >/dev/null 2>&1 || true

echo "==> Logging in to nvcr.io (if needed)..."
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin

echo "==> Pulling NIM image..."
if ! docker pull nvcr.io/nim/nvidia/parakeet-1-1b-rnnt-multilingual:latest; then
  echo "" >&2
  echo "Pull failed. Your Legacy/Personal key may be fine — this is usually missing" >&2
  echo "catalog entitlement for the NIM image. Do all of the following:" >&2
  echo "  1. Open and click Get API Key / accept terms on the container page:" >&2
  echo "     https://catalog.ngc.nvidia.com/orgs/nim/nvidia/containers/parakeet-1-1b-rnnt-multilingual" >&2
  echo "  2. Also open the Build page and enable deploy access:" >&2
  echo "     https://build.nvidia.com/nvidia/parakeet-1_1b-rnnt-multilingual-asr" >&2
  echo "  3. If prompted, start NVIDIA AI Enterprise trial (required for some NIM orgs)" >&2
  echo "  4. Re-run: ./scripts/parakeet_nim_up.sh" >&2
  exit 1
fi

migrate_and_prune_legacy_cache
ensure_cache_volume_writable "$STR_CACHE_VOLUME"
ensure_cache_volume_writable "$OFL_CACHE_VOLUME"

STR_SELECTOR="${PARAKEET_NIM_TAGS_SELECTOR:-diarizer=sortformer,mode=str,type=default,vad=silero}"
OFL_SELECTOR="${PARAKEET_NIM_OFL_TAGS_SELECTOR:-diarizer=sortformer,mode=ofl,type=default,vad=silero}"

echo "==> Starting Parakeet NIM containers (first run per profile may take 15–30 min)..."
echo "    parakeet-nim-str: $STR_SELECTOR"
echo "    parakeet-nim-ofl: $OFL_SELECTOR"
$COMPOSE up -d parakeet-nim-str parakeet-nim-ofl

STR_HTTP_PORT="${PARAKEET_NIM_HTTP_PORT:-9000}"
OFL_HTTP_PORT="${PARAKEET_NIM_OFL_HTTP_PORT:-9001}"
STR_GRPC_PORT="${PARAKEET_NIM_GRPC_PORT:-50051}"
OFL_GRPC_PORT="${PARAKEET_NIM_OFL_GRPC_PORT:-50052}"

STR_OK=0
OFL_OK=0
wait_for_ready "parakeet-nim-str" "http://localhost:${STR_HTTP_PORT}/v1/health/ready" 120 && STR_OK=1 || true
wait_for_ready "parakeet-nim-ofl" "http://localhost:${OFL_HTTP_PORT}/v1/health/ready" 120 && OFL_OK=1 || true

echo ""
if [[ "$STR_OK" -eq 1 && "$OFL_OK" -eq 1 ]]; then
  echo "Both Parakeet NIM containers are ready."
elif [[ "$STR_OK" -eq 1 ]]; then
  echo "Streaming container is ready; offline container still warming up."
elif [[ "$OFL_OK" -eq 1 ]]; then
  echo "Offline container is ready; streaming container still warming up."
else
  echo "Timed out waiting for readiness. Check logs:"
  echo "  $COMPOSE logs -f parakeet-nim-str parakeet-nim-ofl"
  exit 1
fi

echo ""
echo "  Streaming (mode=str):  gRPC localhost:${STR_GRPC_PORT}  HTTP localhost:${STR_HTTP_PORT}"
echo "  Offline  (mode=ofl):   gRPC localhost:${OFL_GRPC_PORT}  HTTP localhost:${OFL_HTTP_PORT}"
echo ""
echo "Streaming test:"
echo "  python clients/transcribe_grpc.py data/codeswitch/en_only.wav --server localhost:${STR_GRPC_PORT}"
echo ""
echo "Offline batch test (HTTP):"
echo "  curl -sS http://localhost:${OFL_HTTP_PORT}/v1/audio/transcriptions -F file=@data/codeswitch/en_only.wav -F model=parakeet"
