#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TTS_DIR="${DICE_TTS_ROOT:-${ROOT_DIR}/tts/qwen3-tts}"
HOST="${QWEN3_TTS_HOST:-127.0.0.1}"
PORT="${QWEN3_TTS_PORT:-18080}"
SERVER_BIN="$(readlink -f "${TTS_DIR}/runtime/bin/llama-server" 2>/dev/null || true)"
MODEL_DIR="${TTS_DIR}/qwen3-tts-0.6b"

if [[ -z "$SERVER_BIN" || ! -x "$SERVER_BIN" ]]; then
    echo "Missing migrated Qwen3-TTS runtime: ${TTS_DIR}/runtime/bin/llama-server" >&2
    exit 1
fi
if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
    echo "Missing migrated Qwen3-TTS model config: ${MODEL_DIR}/config.json" >&2
    exit 1
fi

is_expected_server() {
    local pid="$1"
    local exe cmdline
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    exe="$(readlink -f "/proc/${pid}/exe" 2>/dev/null || true)"
    cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
    [[ "$exe" == "$SERVER_BIN" ]] || return 1
    [[ "$cmdline" == *"--media-backend smt"* ]] || return 1
    [[ "$cmdline" == *"--smt-config-dir ${MODEL_DIR}"* ]] || return 1
    [[ "$cmdline" == *"--port ${PORT}"* ]] || return 1
}

find_expected_pid() {
    local proc pid
    for proc in /proc/[0-9]*; do
        [[ -d "$proc" ]] || continue
        pid="${proc##*/}"
        if is_expected_server "$pid"; then
            printf '%s\n' "$pid"
            return 0
        fi
    done
    return 1
}

if curl -fsS --max-time 2 "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    if pid="$(find_expected_pid)"; then
        echo "Qwen3-TTS already available from migrated project: pid=${pid} http://${HOST}:${PORT}"
        exit 0
    fi
    echo "TTS port ${HOST}:${PORT} is occupied by another runtime; refusing to reuse or stop it." >&2
    echo "Stop the old Qwen3-TTS service explicitly, then run this script again:" >&2
    echo "  cd ${TTS_DIR} && ./stop_server.sh" >&2
    exit 1
fi

if [[ ! -x "${TTS_DIR}/start_server.sh" ]]; then
    echo "Missing migrated Qwen3-TTS launcher: ${TTS_DIR}/start_server.sh" >&2
    exit 1
fi

exec "${TTS_DIR}/start_server.sh"
