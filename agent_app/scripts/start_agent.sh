#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"
PROJECT_ROOT="$(dirname "$APP_ROOT")"
VENV="$PROJECT_ROOT/.venv"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Missing $VENV. Create it with: python3 -m venv $VENV" >&2
  exit 1
fi

cd "$APP_ROOT"
export MENET_CORE_ROOT="${MENET_CORE_ROOT:-$PROJECT_ROOT/MENET}"
export PYTHONPATH="$APP_ROOT:$MENET_CORE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$VENV/bin/python" -m uvicorn agent.api:app \
  --host "${MENET_AGENT_HOST:-127.0.0.1}" \
  --port "${MENET_AGENT_PORT:-8010}"
