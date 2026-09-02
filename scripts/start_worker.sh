#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(dirname "$(realpath "$0")")")"
if [[ ! -x .venv/bin/celery ]]; then
  echo "Missing Celery in .venv. Install requirements.txt first." >&2
  exit 1
fi

exec .venv/bin/celery -A agent.celery_app.celery worker \
  --loglevel "${MENET_WORKER_LOGLEVEL:-INFO}" \
  --pool "${MENET_WORKER_POOL:-solo}"
