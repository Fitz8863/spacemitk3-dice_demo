#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8080}"
RUNTIME_DIR="${DICE_RUNTIME_DIR:-${ROOT_DIR}/.runtime}"
PID_FILE="${PID_FILE:-${RUNTIME_DIR}/web-${PORT}.pid}"

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

pid=""
if [[ -f "$PID_FILE" ]]; then
    candidate="$(cat "$PID_FILE" 2>/dev/null || true)"
    if is_expected_web "$candidate"; then
        pid="$candidate"
    else
        rm -f "$PID_FILE"
    fi
fi
if [[ -z "$pid" ]]; then
    pid="$(find_expected_pid || true)"
fi
if [[ -z "$pid" ]]; then
    echo "Dice Arena web is not running"
    exit 0
fi

kill "$pid"
for _ in {1..30}; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.1
done
if kill -0 "$pid" 2>/dev/null; then
    echo "Web process $pid did not stop cleanly" >&2
    exit 1
fi
rm -f "$PID_FILE"
echo "Dice Arena web stopped: pid=$pid"
