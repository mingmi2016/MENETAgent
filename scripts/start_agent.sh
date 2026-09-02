#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(dirname "$(realpath "$0")")")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv. Create it with: python3 -m venv .venv" >&2
  exit 1
fi

exec .venv/bin/uvicorn agent.api:app --host "${MENET_AGENT_HOST:-127.0.0.1}" --port "${MENET_AGENT_PORT:-8010}"
