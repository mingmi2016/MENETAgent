#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"
PROJECT_ROOT="$(dirname "$APP_ROOT")"
VENV="$PROJECT_ROOT/.venv"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Missing $VENV. Install agent_app/requirements.txt first." >&2
  exit 1
fi

cd "$APP_ROOT"
export MENET_CORE_ROOT="${MENET_CORE_ROOT:-$PROJECT_ROOT/MENET}"
export PYTHONPATH="$APP_ROOT:$MENET_CORE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$VENV/bin/python" -m celery -A agent.celery_app.celery worker \
  --loglevel "${MENET_WORKER_LOGLEVEL:-INFO}" \
  --pool "${MENET_WORKER_POOL:-solo}"
