#!/usr/bin/env bash
# The verification gate, defined once.
#
# Called by .claude/hooks/after_edit.py, .claude/hooks/stop_gate.py, the /verify
# command, and .github/workflows/gate.yml. One definition means CI and local can
# never disagree about what "green" means.
#
# Usage:
#   scripts/gate.sh                 backend, plus frontend iff apps/frontend is dirty
#   scripts/gate.sh --backend-only  pytest only
#   scripts/gate.sh --frontend-only tsc + oxlint only
#   scripts/gate.sh --all           everything, regardless of dirty state (CI uses this)
#
# Exits 0 only if every check it ran passed. Prints the real tail of any failure:
# a summary is not evidence.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

MODE="auto"
case "${1:-}" in
  --backend-only)  MODE="backend" ;;
  --frontend-only) MODE="frontend" ;;
  --all)           MODE="all" ;;
  "")              MODE="auto" ;;
  *) echo "gate.sh: unknown argument '$1'" >&2; exit 64 ;;
esac

run_backend=0
run_frontend=0
case "$MODE" in
  backend)  run_backend=1 ;;
  frontend) run_frontend=1 ;;
  all)      run_backend=1; run_frontend=1 ;;
  auto)
    run_backend=1
    # Frontend checks cost 2.8s, so only pay them when frontend files moved.
    if git status --porcelain -- apps/frontend 2>/dev/null | grep -q .; then
      run_frontend=1
    fi
    ;;
esac

FAILED=0
TAIL_LINES="${GATE_TAIL_LINES:-25}"

section() { printf '\n=== %s ===\n' "$1"; }

# Runs a command, prints a one-line verdict, and on failure prints the tail of
# the real output. Never claims a pass it did not observe.
check() {
  local label="$1"; shift
  local out status
  out="$("$@" 2>&1)"; status=$?
  if [ "$status" -eq 0 ]; then
    printf 'PASS  %-22s %s\n' "$label" "$(printf '%s' "$out" | tail -1)"
  else
    FAILED=1
    printf 'FAIL  %-22s (exit %s)\n' "$label" "$status"
    printf '%s\n' "$out" | tail -n "$TAIL_LINES" | sed 's/^/      /'
  fi
  return 0
}

if [ "$run_backend" -eq 1 ]; then
  section "backend"
  check "pytest" uv run pytest -q -p no:warnings
fi

if [ "$run_frontend" -eq 1 ]; then
  section "frontend"
  if [ -d apps/frontend/node_modules ]; then
    ( cd apps/frontend && exec npx tsc -b ) >/tmp/.gate_tsc 2>&1 && \
      printf 'PASS  %-22s no type errors\n' "tsc -b" || {
        FAILED=1
        printf 'FAIL  %-22s\n' "tsc -b"
        tail -n "$TAIL_LINES" /tmp/.gate_tsc | sed 's/^/      /'
      }
    # oxlint currently reports warnings only, so its exit status gates on errors
    # and the warning count is reported rather than enforced.
    ( cd apps/frontend && exec npx oxlint ) >/tmp/.gate_oxlint 2>&1
    ox_status=$?
    # oxlint prints one line per diagnostic and no summary line, so count them.
    ox_warn=$(grep -c ': warning ' /tmp/.gate_oxlint || true)
    ox_err=$(grep -c ': error ' /tmp/.gate_oxlint || true)
    ox_summary="${ox_warn} warning(s), ${ox_err} error(s)"
    if [ "$ox_status" -eq 0 ]; then
      printf 'PASS  %-22s %s\n' "oxlint" "$ox_summary"
    else
      FAILED=1
      printf 'FAIL  %-22s %s\n' "oxlint" "$ox_summary"
      tail -n "$TAIL_LINES" /tmp/.gate_oxlint | sed 's/^/      /'
    fi
    rm -f /tmp/.gate_tsc /tmp/.gate_oxlint
  else
    printf 'SKIP  %-22s apps/frontend/node_modules missing (run npm install)\n' "tsc + oxlint"
  fi
fi

section "gate"
if [ "$FAILED" -eq 0 ]; then
  echo "green"
  exit 0
fi
echo "red"
exit 1
