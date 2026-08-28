#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"

cd "$SCRIPT_DIR"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: the WSL Python virtual environment is missing or broken." >&2
    echo "Recreate it from ${SCRIPT_DIR} with: uv sync --frozen" >&2
    exit 1
fi

echo "Starting Agent Wiki Frontend and Backend Server..."
exec "$PYTHON" run.py "$@"
