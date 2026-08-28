#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
RUNTIME_DIR="${DICE_RUNTIME_DIR:-${ROOT_DIR}/.runtime}"
PID_FILE="${PID_FILE:-${RUNTIME_DIR}/web-${PORT}.pid}"
PYTHON_BIN="${DICE_PYTHON:-python3}"
LOG_FILE="${LOG_FILE:-${RUNTIME_DIR}/web-${PORT}.log}"
SELECTED_TTS_PROVIDER="${DICE_TTS_PROVIDER:-}"
if [[ -z "$SELECTED_TTS_PROVIDER" ]]; then
    SELECTED_TTS_PROVIDER="$("$PYTHON_BIN" "$ROOT_DIR/backend/componentctl.py" selected tts --game dice)"
fi
TTS_AUTOSTART_ENABLED="${TTS_AUTOSTART:-1}"

mkdir -p "$(dirname "$PID_FILE")"

is_expected_web() {
    local pid="$1"
    local cwd exe script arg i
    local -a argv=()
    local port_matches=0
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    cwd="$(readlink -f "/proc/${pid}/cwd" 2>/dev/null || true)"
    exe="$(readlink -f "/proc/${pid}/exe" 2>/dev/null || true)"
    [[ "$cwd" == "$ROOT_DIR" ]] || return 1
    [[ "$(basename "$exe")" == python* ]] || return 1
    mapfile -d '' -t argv < "/proc/${pid}/cmdline" 2>/dev/null || return 1
    [[ "${#argv[@]}" -ge 2 ]] || return 1
    script="${argv[1]}"
    if [[ "$script" != /* ]]; then
        script="$(readlink -f "$cwd/$script" 2>/dev/null || true)"
    else
        script="$(readlink -f "$script" 2>/dev/null || true)"
    fi
    [[ "$script" == "$ROOT_DIR/backend/server.py" ]] || return 1
    for ((i = 2; i < ${#argv[@]}; i++)); do
        arg="${argv[i]}"
        if [[ "$arg" == "--port=$PORT" ]]; then
            port_matches=1
            break
        fi
        if [[ "$arg" == "--port" && $((i + 1)) -lt ${#argv[@]} && "${argv[i + 1]}" == "$PORT" ]]; then
            port_matches=1
            break
        fi
    done
    [[ "$port_matches" == "1" ]]
}

find_expected_pid() {
    local proc pid
    for proc in /proc/[0-9]*; do
        [[ -d "$proc" ]] || continue
        pid="${proc##*/}"
        if is_expected_web "$pid"; then
            printf '%s\n' "$pid"
            return 0
        fi
    done
    return 1
}

start_selected_tts() {
    [[ "$TTS_AUTOSTART_ENABLED" != "0" ]] || return 0
    if ! "$PYTHON_BIN" "$ROOT_DIR/backend/componentctl.py" start "$SELECTED_TTS_PROVIDER"; then
        if [[ "${TTS_REQUIRED:-0}" == "1" ]]; then
            echo "TTS provider $SELECTED_TTS_PROVIDER is required but could not be started" >&2
            exit 1
        fi
        echo "Warning: TTS provider $SELECTED_TTS_PROVIDER is unavailable" >&2
    fi
}

pid=""
if [[ -f "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if ! is_expected_web "$pid"; then
        rm -f "$PID_FILE"
        pid=""
    fi
fi
if [[ -z "$pid" ]]; then
    pid="$(find_expected_pid || true)"
    if [[ -n "$pid" ]]; then
        printf '%s\n' "$pid" > "$PID_FILE"
    fi
fi
if [[ -n "$pid" ]]; then
    running_tts_provider="$(
        curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/api/health" 2>/dev/null \
        | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin).get("tts_provider", ""))' \
        2>/dev/null || true
    )"
    running_tts_provider="${running_tts_provider:-unknown}"
    if [[ "$running_tts_provider" != "$SELECTED_TTS_PROVIDER" ]]; then
        echo "Dice Arena web is already running with TTS provider $running_tts_provider." >&2
        echo "Run scripts/stop_web.sh, then start again with DICE_TTS_PROVIDER=$SELECTED_TTS_PROVIDER." >&2
        exit 1
    fi
    start_selected_tts
    echo "Dice Arena web is already running: pid=$pid port=$PORT tts_provider=$running_tts_provider"
    exit 0
fi

# Refuse to hide an unrelated process already listening on the requested port.
if command -v ss >/dev/null 2>&1 && ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN; then
    echo "Port ${PORT} is already in use by a different process" >&2
    exit 1
fi

cd "$ROOT_DIR"
start_selected_tts
nohup "$PYTHON_BIN" backend/server.py --host "$HOST" --port "$PORT" \
    >>"$LOG_FILE" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$PID_FILE"

sleep 0.3
if ! is_expected_web "$pid"; then
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
