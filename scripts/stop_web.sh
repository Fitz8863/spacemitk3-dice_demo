#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8080}"
RUNTIME_DIR="${DICE_RUNTIME_DIR:-${ROOT_DIR}/.runtime}"
PID_FILE="${PID_FILE:-${RUNTIME_DIR}/web-${PORT}.pid}"
TTS_PROVIDER_FILE="${TTS_PROVIDER_FILE:-${RUNTIME_DIR}/web-${PORT}.tts-provider}"
PYTHON_BIN="${DICE_PYTHON:-python3}"

resolve_tts_providers() {
    # Prefer the manifest-derived list (local + remote slots + per-line
    # overrides); fall back to the recorded/running provider ids.
    local providers
    providers="$("$PYTHON_BIN" "$ROOT_DIR/backend/componentctl.py" referenced tts --game "${DICE_GAME:-dice}" 2>/dev/null || true)"
    if [[ -n "$providers" ]]; then
        printf '%s\n' "$providers"
        return 0
    fi
    local provider=""
    if [[ -f "$TTS_PROVIDER_FILE" ]]; then
        provider="$(head -n 1 "$TTS_PROVIDER_FILE" 2>/dev/null || true)"
    fi
    if [[ -z "$provider" ]]; then
        provider="$(curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/api/health" 2>/dev/null \
            | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin).get("tts_provider", ""))' 2>/dev/null || true)"
    fi
    printf '%s\n' "$provider"
}

stop_selected_tts() {
    local providers="${1:-}"
    if [[ -z "$providers" ]]; then
        providers="$(resolve_tts_providers)"
    fi
    local id
    local -a stopped=()
    while IFS= read -r id; do
        [[ -n "$id" ]] || continue
        for stopped_id in "${stopped[@]:-}"; do
            [[ "$stopped_id" == "$id" ]] && id="" && break
        done
        [[ -n "$id" ]] || continue
        stopped+=("$id")
        "$PYTHON_BIN" "$ROOT_DIR/backend/componentctl.py" stop "$id" || {
            echo "Warning: failed to stop TTS provider $id" >&2
        }
    done <<< "$providers"
    if [[ "${#stopped[@]}" -eq 0 ]]; then
        echo "Unable to resolve selected TTS provider; skipping TTS stop" >&2
    fi
}

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
    stop_selected_tts
    rm -f "$TTS_PROVIDER_FILE"
    exit 0
fi

running_tts_provider="$(curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/api/health" 2>/dev/null \
    | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin).get("tts_provider", ""))' 2>/dev/null || true)"
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
# Stop both the provider reported by the running backend and everything the
# current manifest references, so a manifest edit between start and stop
# cannot leak a warmed local runtime.
stop_selected_tts "$(printf '%s\n%s\n' "$running_tts_provider" "$(resolve_tts_providers)")"
rm -f "$TTS_PROVIDER_FILE"
