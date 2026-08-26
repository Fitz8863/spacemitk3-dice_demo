#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="${PID_FILE:-${ROOT_DIR}/web/.dice-arena-web.pid}"

if [[ ! -f "$PID_FILE" ]]; then
    echo "Dice Arena web is not running (no pid file)"
    exit 0
fi

pid="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
    rm -f "$PID_FILE"
    echo "Removed invalid pid file"
    exit 0
fi

if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    for _ in {1..20}; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.1
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "Web process $pid did not stop cleanly" >&2
        exit 1
    fi
fi
rm -f "$PID_FILE"
echo "Dice Arena web stopped"
