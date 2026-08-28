#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${DICE_PYTHON:-python3}"
exec "$PYTHON_BIN" "$ROOT_DIR/backend/componentctl.py" stop-selected tts --game "${DICE_GAME:-dice}"
