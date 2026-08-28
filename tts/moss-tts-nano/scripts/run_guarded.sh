#!/usr/bin/env bash

set -euo pipefail

if (( $# == 0 )); then
    echo "usage: $0 <demo command> [args ...]" >&2
    exit 2
fi

for required_command in spacemit-tcm-smi timeout flock; do
    if ! command -v "${required_command}" >/dev/null 2>&1; then
        echo "missing required command: ${required_command}" >&2
        exit 2
    fi
done

benchmark_timeout_seconds="${SPACEMIT_BENCH_TIMEOUT_SECONDS:-300}"
if [[ ! "${benchmark_timeout_seconds}" =~ ^[1-9][0-9]*$ ]]; then
    echo "SPACEMIT_BENCH_TIMEOUT_SECONDS must be a positive integer" >&2
    exit 2
fi

verbose="${MOSS_TTS_VERBOSE:-0}"
if [[ "${verbose}" != "0" && "${verbose}" != "1" ]]; then
    echo "MOSS_TTS_VERBOSE must be 0 or 1" >&2
    exit 2
fi

interactive="${MOSS_TTS_INTERACTIVE:-0}"
if [[ "${interactive}" != "0" && "${interactive}" != "1" ]]; then
    echo "MOSS_TTS_INTERACTIVE must be 0 or 1" >&2
    exit 2
fi

lock_root="${TMPDIR:-/tmp}"
exec 9>"${lock_root%/}/moss_tts_nano_spacemit_demo.lock"
if ! flock -n 9; then
    echo "another guarded MOSS-TTS demo is already running" >&2
    exit 20
fi

list_competing_ai_processes() {
    ps -eo pid=,comm=,args= | awk '
        $2 ~ /^python([0-9.]*)?$/ && $0 ~ /(moss_spacemit_demo|benchmark_onnx|infer_onnx|MOSS-TTS|onnxruntime)/ { print; next }
        $2 ~ /^(onnxruntime_perf_test|llama-server)$/ { print; next }
        $2 ~ /^muggle_/ { print }
    '
}

all_tcm_blocks_free() {
    local status_text="$1"
    local availability free_blocks total_blocks

    availability="$(sed -n 's/.*available_blocks=\([0-9][0-9]*\/[0-9][0-9]*\).*/\1/p' <<<"${status_text}" | head -n 1)"
    [[ -n "${availability}" ]] || return 1
    free_blocks="${availability%%/*}"
    total_blocks="${availability##*/}"
    [[ "${free_blocks}" == "${total_blocks}" ]]
}

tcm_availability() {
    sed -n 's/.*available_blocks=\([0-9][0-9]*\/[0-9][0-9]*\).*/\1/p' <<<"$1" | head -n 1
}

competing_processes="$(list_competing_ai_processes)"
if [[ -n "${competing_processes}" ]]; then
    echo "refusing TCM cleanup while another AI process is running:" >&2
    echo "${competing_processes}" >&2
    exit 21
fi

preflight_status="$(spacemit-tcm-smi)"
if [[ "${verbose}" == "1" ]]; then
    echo "TCM preflight before guarded cleanup:" >&2
    echo "${preflight_status}" >&2
fi
if ! cleanup_output="$(spacemit-tcm-smi -c 2>&1)"; then
    echo "TCM cleanup failed:" >&2
    echo "${cleanup_output}" >&2
    exit 22
fi

clean_status="$(spacemit-tcm-smi)"
if ! all_tcm_blocks_free "${clean_status}"; then
    echo "TCM cleanup did not return every block to free; demo aborted" >&2
    echo "${clean_status}" >&2
    exit 22
fi
if [[ "${verbose}" == "1" ]]; then
    echo "TCM after guarded cleanup:" >&2
    echo "${clean_status}" >&2
else
    echo "TCM: ready ($(tcm_availability "${clean_status}") free)" >&2
fi

postflight() {
    local demo_rc=$?
    local final_status final_competing_processes

    trap - EXIT INT TERM
    set +e
    final_status="$(spacemit-tcm-smi 2>&1)"
    if ! all_tcm_blocks_free "${final_status}"; then
        echo "TCM postflight is not fully free:" >&2
        echo "${final_status}" >&2
        final_competing_processes="$(list_competing_ai_processes)"
        if [[ -z "${final_competing_processes}" ]]; then
            echo "stale TCM ownership detected; performing guarded recovery" >&2
            spacemit-tcm-smi -c >&2
            spacemit-tcm-smi >&2
        else
            echo "TCM remains occupied by another process; ownership was not stolen" >&2
            echo "${final_competing_processes}" >&2
            demo_rc=23
        fi
    elif [[ "${verbose}" == "1" ]]; then
        echo "TCM postflight:" >&2
        echo "${final_status}" >&2
    else
        echo "TCM: released ($(tcm_availability "${final_status}") free)" >&2
    fi
    exit "${demo_rc}"
}
trap postflight EXIT INT TERM

if [[ "${interactive}" == "1" ]]; then
    "$@"
else
    timeout --signal=TERM --kill-after=10s "${benchmark_timeout_seconds}s" "$@"
fi
