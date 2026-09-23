#!/usr/bin/env bash
# Startup for the test2 K3 deployment (10.0.91.132).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
systemctl --user start dice-wangjie-mediamtx.service
export START_HEALTH_TIMEOUT="${START_HEALTH_TIMEOUT:-90}"
exec bash "$ROOT_DIR/scripts/start_web.sh"
