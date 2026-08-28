#!/usr/bin/env bash
set -euo pipefail
PLUGIN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${DICE_PYTHON:-python3}" "$PLUGIN_DIR/launcher.py" stop "$@"
