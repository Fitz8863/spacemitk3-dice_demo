#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_RUNTIME="$ROOT/runtime"
[[ -x "$DEFAULT_RUNTIME/bin/llama-server" ]] || DEFAULT_RUNTIME="$ROOT/llama.cpp-optimized/build-k3"
RUNTIME="${QWEN3_TTS_RUNTIME:-$DEFAULT_RUNTIME}"
PORT="${QWEN3_TTS_PORT:-18080}"
PID_FILE="$ROOT/llama-server.pid"
TERM_TIMEOUT="${QWEN3_TTS_STOP_TIMEOUT:-20}"

SERVER_BIN="$RUNTIME/bin/llama-server"

# Read /proc/<pid>/cmdline as an argument array. This avoids matching an
# unrelated process merely because its command line contains the same text.
read_cmdline_args() {
    local pid="$1"
    local proc_cmdline="/proc/$pid/cmdline"
    [[ -r "$proc_cmdline" ]] || return 1

    CMDLINE_ARGS=()
    # /proc cmdline is NUL-separated and has no trailing newline. readarray
    # returns a non-zero status for an empty/non-standard proc entry, which is
    # harmless here because the array is checked below.
    readarray -d '' -t CMDLINE_ARGS < "$proc_cmdline" 2>/dev/null || true
    ((${#CMDLINE_ARGS[@]} > 0))
}

# Only accept the Qwen3-TTS server started by this project:
#   exact llama-server path + --media-backend smt + configured HTTP port.
# The PID file is not trusted on its own because it can be stale after a
# crash, a manual kill, or a previous start attempt that raced with an old
# server already listening on the port.
is_target_server() {
    local pid="$1"
    local i
    local has_backend=0
    local has_port=0

    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    read_cmdline_args "$pid" || return 1
    [[ "${CMDLINE_ARGS[0]:-}" == "$SERVER_BIN" ]] || return 1

    for ((i = 0; i < ${#CMDLINE_ARGS[@]}; i++)); do
        case "${CMDLINE_ARGS[i]}" in
            --media-backend)
                [[ "${CMDLINE_ARGS[i + 1]:-}" == "smt" ]] && has_backend=1
                ;;
            --port)
                [[ "${CMDLINE_ARGS[i + 1]:-}" == "$PORT" ]] && has_port=1
                ;;
        esac
    done

    ((has_backend == 1 && has_port == 1))
}

append_unique_pid() {
    local candidate="$1"
    local existing
    [[ "$candidate" =~ ^[0-9]+$ ]] || return 0
    for existing in "${TARGET_PIDS[@]}"; do
        [[ "$existing" == "$candidate" ]] && return 0
    done
    TARGET_PIDS+=("$candidate")
}

# Discover a live server if the PID file is missing or stale. This is what
# handles the case where start_server.sh wrote a PID for a failed second
# instance while an older healthy instance kept the port occupied.
discover_target_servers() {
    local proc pid
    for proc in /proc/[0-9]*; do
        [[ -d "$proc" ]] || continue
        pid="${proc##*/}"
        if is_target_server "$pid"; then
            append_unique_pid "$pid"
        fi
    done
}

pid_from_file=""
if [[ -f "$PID_FILE" ]]; then
    pid_from_file="$(tr -d '[:space:]' < "$PID_FILE" 2>/dev/null || true)"
    if [[ ! "$pid_from_file" =~ ^[0-9]+$ ]]; then
        echo "警告：PID 文件无效，将扫描实际服务进程: $PID_FILE" >&2
    elif ! is_target_server "$pid_from_file"; then
        echo "警告：PID 文件中的进程不存在或不是当前 Qwen3-TTS 服务，将扫描实际服务进程: pid=$pid_from_file" >&2
    fi
fi

TARGET_PIDS=()
if [[ "$pid_from_file" =~ ^[0-9]+$ ]] && is_target_server "$pid_from_file"; then
    append_unique_pid "$pid_from_file"
fi
discover_target_servers

if ((${#TARGET_PIDS[@]} == 0)); then
    rm -f "$PID_FILE"
    echo "没有找到正在运行的 Qwen3-TTS 服务"
    exit 0
fi

echo "准备停止 Qwen3-TTS: pid=${TARGET_PIDS[*]}"

# start_server.sh uses setsid, so the llama-server normally owns its own
# process group (PGID == PID). Terminating that group also cleans up any
# worker children inherited from the server, while the check prevents us
# from signalling an unrelated process group.
TARGET_PGIDS=()
append_unique_pgid() {
    local candidate="$1"
    local existing
    [[ "$candidate" =~ ^[0-9]+$ ]] || return 0
    for existing in "${TARGET_PGIDS[@]}"; do
        [[ "$existing" == "$candidate" ]] && return 0
    done
    TARGET_PGIDS+=("$candidate")
}

for pid in "${TARGET_PIDS[@]}"; do
    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]' || true)"
    if [[ "$pgid" =~ ^[0-9]+$ && "$pgid" == "$pid" ]]; then
        append_unique_pgid "$pgid"
        echo "发送 SIGTERM 到进程组: -$pgid"
        kill -TERM -- "-$pgid" 2>/dev/null || true
    else
        echo "发送 SIGTERM 到进程: $pid"
        kill -TERM "$pid" 2>/dev/null || true
    fi
done

remaining_targets() {
    local pid
    for pid in "${TARGET_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null && [[ -d "/proc/$pid" ]]; then
            return 0
        fi
    done
    return 1
}

for _ in $(seq 1 "$((TERM_TIMEOUT * 4))"); do
    remaining_targets || break
    sleep 0.25
done

# If graceful shutdown did not finish, use SIGKILL on the same validated
# process groups/PIDs. Do not use killall or an unqualified name match.
if remaining_targets; then
    echo "警告：服务未在 ${TERM_TIMEOUT}s 内退出，发送 SIGKILL" >&2
    for pgid in "${TARGET_PGIDS[@]}"; do
        kill -KILL -- "-$pgid" 2>/dev/null || true
    done
    for pid in "${TARGET_PIDS[@]}"; do
        kill -KILL "$pid" 2>/dev/null || true
    done

    for _ in $(seq 1 20); do
        remaining_targets || break
        sleep 0.25
    done
fi

if remaining_targets; then
    echo "错误：仍有 Qwen3-TTS 进程未退出，保留 PID 文件以便重试" >&2
    exit 1
fi

rm -f "$PID_FILE"
echo "Qwen3-TTS 已停止: pid=${TARGET_PIDS[*]}"
