#!/usr/bin/env bash
set -euo pipefail

PLUGIN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="$(cd -- "${PLUGIN_DIR}/../../.." && pwd)"
PORT="${DICE_MOSS_TTS_PORT:-18082}"
RUNTIME_DIR="${DICE_RUNTIME_DIR:-${ROOT_DIR}/.runtime}"
PID_FILE="${DICE_MOSS_TTS_PID_FILE:-${RUNTIME_DIR}/moss-tts-${PORT}.pid}"

is_expected_bridge() {
    local pid="$1"
    local cwd exe cmdline
    [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
    kill -0 "${pid}" 2>/dev/null || return 1
    cwd="$(readlink -f "/proc/${pid}/cwd" 2>/dev/null || true)"
    exe="$(readlink -f "/proc/${pid}/exe" 2>/dev/null || true)"
    cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
    [[ "${cwd}" == "${ROOT_DIR}" ]] || return 1
    [[ "$(basename "${exe}")" == python* ]] || return 1
    [[ "${cmdline}" == *"backend/components/tts_moss_nano/daemon.py"* ]] || return 1
    [[ "${cmdline}" == *"--port ${PORT}"* ]] || return 1
}

pid=""
if [[ -f "${PID_FILE}" ]]; then
    candidate="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if is_expected_bridge "${candidate}"; then
        pid="${candidate}"
    else
        rm -f "${PID_FILE}"
    fi
fi
if [[ -z "${pid}" ]]; then
    echo "MOSS-TTS bridge is not running"
    exit 0
fi

kill "${pid}"
for _ in {1..100}; do
    kill -0 "${pid}" 2>/dev/null || break
    sleep 0.1
done
if kill -0 "${pid}" 2>/dev/null; then
    echo "MOSS-TTS bridge ${pid} did not stop cleanly" >&2
    exit 1
fi
rm -f "${PID_FILE}"
echo "MOSS-TTS bridge stopped: pid=${pid}"
