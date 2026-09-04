#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
RUNTIME_DIR="${DICE_RUNTIME_DIR:-${ROOT_DIR}/.runtime}"
PID_FILE="${PID_FILE:-${RUNTIME_DIR}/web-${PORT}.pid}"
TTS_PROVIDER_FILE="${TTS_PROVIDER_FILE:-${RUNTIME_DIR}/web-${PORT}.tts-provider}"
PYTHON_BIN="${DICE_PYTHON:-python3}"
LOG_FILE="${LOG_FILE:-${RUNTIME_DIR}/web-${PORT}.log}"
# How long to wait for /api/health after spawning the server.  The port only
# binds after registry + config + engine prewarming (several seconds), so a
# "process is alive" check is not a usable service.
START_HEALTH_TIMEOUT="${START_HEALTH_TIMEOUT:-30}"

# Resolve referenced providers tolerantly: a broken config/manifest must not
# kill the script before it can say what is wrong.  The running server keeps
# the last good config via hot-reload, so a JSON broken mid-edit only bites
# at restart time — exactly here.
REFERENCED_TTS_PROVIDERS=""
if ! REFERENCED_TTS_PROVIDERS="$("$PYTHON_BIN" "$ROOT_DIR/backend/componentctl.py" referenced tts --game dice 2>&1)"; then
    echo "Warning: cannot resolve referenced TTS providers — a config/manifest JSON may be broken:" >&2
    printf '%s\n' "$REFERENCED_TTS_PROVIDERS" >&2
    REFERENCED_TTS_PROVIDERS=""
fi
# The first referenced id is the arena's primary (local-slot) voice.
SELECTED_TTS_PROVIDER="$(printf '%s\n' "$REFERENCED_TTS_PROVIDERS" | head -n1)"
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

wait_for_exit() {
    # wait_for_exit <pid> <seconds>: 0 = process gone, 1 = still alive.
    local pid="$1" seconds="$2" waited=0
    while (( waited < seconds * 10 )); do
        if ! kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        sleep 0.1
        waited=$((waited + 1))
    done
    if kill -0 "$pid" 2>/dev/null; then
        return 1
    fi
    return 0
}

wait_until_healthy() {
    # wait_until_healthy <pid> <seconds>: 0 = /api/health answering.
    local pid="$1" timeout_seconds="$2" waited=0
    while (( waited < timeout_seconds * 2 )); do
        if ! is_expected_web "$pid"; then
            return 1
        fi
        if curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.5
        waited=$((waited + 1))
    done
    return 1
}

runtime_children() {
    # Resident engine children of this deployment (yolov8_camera,
    # stream_asr; arecord exits by itself once stream_asr's pipe closes).
    # Matched by executable path under this project so a sibling checkout or
    # unrelated process is never touched.
    local pid exe
    for pid in $(pgrep -x yolov8_camera 2>/dev/null; pgrep -x stream_asr 2>/dev/null); do
        exe="$(readlink -f "/proc/${pid}/exe" 2>/dev/null || true)"
        case "$exe" in
            "$ROOT_DIR"/*) printf '%s\n' "$pid" ;;
        esac
    done
}

kill_runtime_children() {
    # Used only after killing a hung previous instance: its resident children
    # are orphaned and still hold the camera/microphone, which would make the
    # fresh server's engine prewarming fail on the devices.
    local waited=0 pid
    while (( waited < 50 )); do
        if [[ -z "$(runtime_children)" ]]; then
            return 0
        fi
        sleep 0.1
        waited=$((waited + 1))
    done
    for pid in $(runtime_children); do
        echo "Warning: killing leftover runtime child pid=$pid (still holding camera/mic)" >&2
        kill -KILL "$pid" 2>/dev/null || true
    done
}

start_selected_tts() {
    [[ "$TTS_AUTOSTART_ENABLED" != "0" ]] || return 0
    local id
    # Every referenced local provider gets started so a manifest may mix
    # board-local and remote engines per speech line; remote-only providers
    # have no lifecycle and their start is a no-op.
    while IFS= read -r id; do
        [[ -n "$id" ]] || continue
        if ! "$PYTHON_BIN" "$ROOT_DIR/backend/componentctl.py" start "$id"; then
            if [[ "${TTS_REQUIRED:-0}" == "1" ]]; then
                echo "TTS provider $id is required but could not be started" >&2
                exit 1
            fi
            echo "Warning: TTS provider $id is unavailable" >&2
        fi
    done <<< "$REFERENCED_TTS_PROVIDERS"
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
    if [[ -n "$running_tts_provider" ]]; then
        if [[ -n "$SELECTED_TTS_PROVIDER" && "$running_tts_provider" != "$SELECTED_TTS_PROVIDER" ]]; then
            echo "Dice Arena web is already running with TTS provider $running_tts_provider." >&2
            echo "To switch providers, edit backend/config.json providers.tts_local / providers.tts_remote (or the game manifest override), then run scripts/stop_web.sh and scripts/start_web.sh." >&2
            exit 1
        fi
        start_selected_tts
        printf '%s\n' "$SELECTED_TTS_PROVIDER" > "$TTS_PROVIDER_FILE"
        echo "Dice Arena web is already running: pid=$pid port=$PORT tts_provider=$running_tts_provider"
        exit 0
    fi
    # The instance is alive but not answering — usually a previous shutdown
    # still in flight (stop_web.sh raced the serial teardown).  Wait for it
    # to disappear, escalating once, and then start fresh instead of failing
    # the restart.
    echo "Previous web pid=$pid is not answering (still shutting down?); waiting for it to exit…" >&2
    if ! wait_for_exit "$pid" 15; then
        echo "Previous web pid=$pid is hung; sending SIGKILL" >&2
        kill -KILL "$pid" 2>/dev/null || true
        if ! wait_for_exit "$pid" 5; then
            echo "Previous web pid=$pid survived SIGKILL; resolve manually with: kill -9 $pid" >&2
            exit 1
        fi
        kill_runtime_children
    fi
    echo "Previous instance exited; starting a fresh one." >&2
    rm -f "$PID_FILE"
    pid=""
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
printf '%s\n' "$SELECTED_TTS_PROVIDER" > "$TTS_PROVIDER_FILE"

# Success means the service answers, not merely that the process was alive:
# prewarming happens before the port binds, and a boot-time crash would
# otherwise be reported as a successful start.
if ! wait_until_healthy "$pid" "$START_HEALTH_TIMEOUT"; then
    rm -f "$PID_FILE"
    rm -f "$TTS_PROVIDER_FILE"
    echo "Failed to start Dice Arena web (no healthy response within ${START_HEALTH_TIMEOUT}s). Last log lines:" >&2
    tail -n 25 "$LOG_FILE" >&2 || true
    exit 1
fi

echo "Dice Arena K3 backend started"
echo "  root: $ROOT_DIR"
echo "  url:  http://127.0.0.1:$PORT"
echo "  pid:  $pid"
echo "  log:  $LOG_FILE"
echo "Stop it with: scripts/stop_web.sh"
