#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"
PROJECT_ROOT="$(dirname "$APP_ROOT")"
VENV="$PROJECT_ROOT/.venv"

cd "$APP_ROOT"
export MENET_CORE_ROOT="${MENET_CORE_ROOT:-$PROJECT_ROOT/MENET}"
export PYTHONPATH="$APP_ROOT:$MENET_CORE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$VENV/bin/python" -m agent.mcp_server
