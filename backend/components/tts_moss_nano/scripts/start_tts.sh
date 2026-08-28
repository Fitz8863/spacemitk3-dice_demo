#!/usr/bin/env bash
set -euo pipefail

PLUGIN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="$(cd -- "${PLUGIN_DIR}/../../.." && pwd)"
HOST="${DICE_MOSS_TTS_HOST:-127.0.0.1}"
PORT="${DICE_MOSS_TTS_PORT:-18082}"
MOSS_ROOT="${DICE_MOSS_TTS_ROOT:-${ROOT_DIR}/tts/moss-tts-nano}"
MODEL_DIR="${DICE_MOSS_TTS_MODEL_DIR:-}"
VOICE="${DICE_MOSS_TTS_VOICE:-Junhao}"
REFERENCE_AUDIO="${DICE_MOSS_TTS_REFERENCE_AUDIO:-}"
PID_FILE="${DICE_MOSS_TTS_PID_FILE:-/tmp/dice-arena-moss-tts-$(id -u)-${PORT}.pid}"
LOG_FILE="${DICE_MOSS_TTS_LOG_FILE:-/tmp/dice-arena-moss-tts-$(id -u)-${PORT}.log}"
START_TIMEOUT="${DICE_MOSS_TTS_START_TIMEOUT_SECONDS:-300}"

# The MOSS delivery bundles its Python modules and SpaceMIT EP dependencies.
# Export these before launching Python: the dynamic loader reads the library
# search path at process startup, before daemon.py can adjust os.environ.
export PYTHONPATH="${MOSS_ROOT}/python:${MOSS_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${MOSS_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [[ ! -d "${MOSS_ROOT}" ]]; then
    echo "MOSS-TTS root does not exist: ${MOSS_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${MOSS_ROOT}/src/onnx_tts_runtime.py" ]]; then
    echo "MOSS OnnxTtsRuntime is missing: ${MOSS_ROOT}/src/onnx_tts_runtime.py" >&2
    exit 1
fi
if [[ -n "${MODEL_DIR}" && ! -f "${MODEL_DIR}/browser_poc_manifest.json" ]]; then
    echo "MOSS model manifest is missing: ${MODEL_DIR}/browser_poc_manifest.json" >&2
    exit 1
fi

is_expected_bridge() {
    local pid="$1"
    local cwd exe cmdline
    [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
    kill -0 "${pid}" 2>/dev/null || return 1
    cwd="$(readlink -f "/proc/${pid}/cwd" 2>/dev/null || true)"
    exe="$(readlink -f "/proc/${pid}/exe" 2>/dev/null || true)"
    cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
    [[ "${cwd}" == "${ROOT_DIR}" ]] || return 1
    [[ "${exe}" == /usr/bin/python3* || "${exe}" == */python3* ]] || return 1
    [[ "${cmdline}" == *"backend/components/tts_moss_nano/daemon.py"* ]] || return 1
    [[ "${cmdline}" == *"--port ${PORT}"* ]] || return 1
}

find_expected_pid() {
    local proc pid
    for proc in /proc/[0-9]*; do
        [[ -d "${proc}" ]] || continue
        pid="${proc##*/}"
        if is_expected_bridge "${pid}"; then
            printf '%s\n' "${pid}"
            return 0
        fi
    done
    return 1
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
    pid="$(find_expected_pid || true)"
fi
if [[ -n "${pid}" ]]; then
    if curl -fsS --max-time 2 "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
        echo "MOSS-TTS bridge already running: pid=${pid} http://${HOST}:${PORT}"
        exit 0
    fi
    echo "MOSS-TTS bridge process ${pid} is not healthy; refusing to reuse it" >&2
    exit 1
fi

if command -v ss >/dev/null 2>&1 && ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN; then
    echo "Port ${PORT} is already in use by a different process" >&2
    exit 1
fi

cd "${ROOT_DIR}"
command=(
    /usr/bin/python3 "${PLUGIN_DIR}/daemon.py"
    --root "${MOSS_ROOT}"
    --host "${HOST}"
    --port "${PORT}"
    --voice "${VOICE}"
)
if [[ -n "${MODEL_DIR}" ]]; then
    command+=(--model-dir "${MODEL_DIR}")
fi
if [[ -n "${REFERENCE_AUDIO}" ]]; then
    command+=(--reference-audio "${REFERENCE_AUDIO}")
fi
nohup "${command[@]}" >>"${LOG_FILE}" 2>&1 &
pid=$!
printf '%s\n' "${pid}" > "${PID_FILE}"

for ((elapsed=0; elapsed<START_TIMEOUT; elapsed++)); do
    if ! kill -0 "${pid}" 2>/dev/null; then
        echo "MOSS-TTS bridge exited during startup; see ${LOG_FILE}" >&2
        rm -f "${PID_FILE}"
        exit 1
    fi
    health="$(curl -fsS --max-time 2 "http://${HOST}:${PORT}/health" 2>/dev/null || true)"
    if [[ -n "${health}" ]] && /usr/bin/python3 -c 'import json,sys; raise SystemExit(0 if json.load(sys.stdin).get("ready") else 1)' <<<"${health}"; then
        echo "MOSS-TTS bridge started: pid=${pid} http://${HOST}:${PORT}"
        exit 0
    fi
    sleep 1
done

echo "MOSS-TTS bridge did not become ready within ${START_TIMEOUT}s; see ${LOG_FILE}" >&2
exit 1
