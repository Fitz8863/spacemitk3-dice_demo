#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
PID_FILE="${PID_FILE:-${ROOT_DIR}/web/.dice-arena-web.pid}"
LOG_FILE="${LOG_FILE:-${ROOT_DIR}/web/dice-arena-web.log}"

mkdir -p "$(dirname "$PID_FILE")"

if [[ -f "$PID_FILE" ]]; then
    old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
        echo "Dice Arena web is already running: pid=$old_pid port=$PORT"
        exit 0
    fi
    rm -f "$PID_FILE"
fi

cd "$ROOT_DIR"
nohup /usr/bin/python3 backend/server.py --host "$HOST" --port "$PORT" \
    >>"$LOG_FILE" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$PID_FILE"

sleep 0.3
if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "Failed to start Dice Arena web. See $LOG_FILE" >&2
    exit 1
fi

echo "Dice Arena K3 backend started"
echo "  root: $ROOT_DIR"
echo "  url:  http://127.0.0.1:$PORT"
echo "  pid:  $pid"
echo "  log:  $LOG_FILE"
echo "Stop it with: scripts/stop_web.sh"
