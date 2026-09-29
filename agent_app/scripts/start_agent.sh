#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"
PROJECT_ROOT="$(dirname "$APP_ROOT")"
REPO_ROOT="$PROJECT_ROOT"
if [[ ! -x "$REPO_ROOT/.venv/bin/python" && -x "$(dirname "$PROJECT_ROOT")/.venv/bin/python" ]]; then
  REPO_ROOT="$(dirname "$PROJECT_ROOT")"
fi

# Load optional deployment configuration. Explicit variables set by the caller take precedence.
if [[ -f "$APP_ROOT/.env" ]]; then
  set -a
  source "$APP_ROOT/.env"
  set +a
fi
PYTHON_BIN="${MENET_PYTHON:-$REPO_ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python executable: $PYTHON_BIN" >&2
  exit 1
fi

cd "$APP_ROOT"
export MENET_CORE_ROOT="${MENET_CORE_ROOT:-$REPO_ROOT/MENET}"
export PYTHONPATH="$APP_ROOT:$MENET_CORE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" -m uvicorn agent.api:app \
  --host "${MENET_AGENT_HOST:-127.0.0.1}" \
  --port "${MENET_AGENT_PORT:-8010}"
