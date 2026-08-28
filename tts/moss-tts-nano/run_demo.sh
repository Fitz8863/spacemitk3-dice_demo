#!/usr/bin/env bash

set -euo pipefail

demo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${demo_root}"

if [[ "$(uname -m)" != "riscv64" ]]; then
    echo "this demo requires a riscv64 K3 target" >&2
    exit 2
fi

export PYTHONPATH="${demo_root}/python:${demo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${demo_root}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export SPACEMIT_EP_INTRA_THREAD_NUM=4
export SPACEMIT_EP_INTER_THREAD_NUM=1
export SPACEMIT_EP_INTRA_THREAD_AFFINITY="8;9;10;11"
export SPACEMIT_DEMO_EXPECT_EP_DIR="${demo_root}/lib"
export SPACEMIT_BENCH_TIMEOUT_SECONDS="${SPACEMIT_BENCH_TIMEOUT_SECONDS:-300}"
unset SPACEMIT_EP_DISABLE_OP_TYPE_FILTER
unset SPACEMIT_EP_ENABLE_PAIR_TCM

for argument in "$@"; do
    if [[ "${argument}" == "--verbose" ]]; then
        export MOSS_TTS_VERBOSE=1
        export MOSS_TTS_ORT_LOG_SEVERITY=2
    fi
done

ep_library="${demo_root}/lib/libspacemit_ep.so.2.0.6"
if [[ ! -f "${ep_library}" ]]; then
    echo "bundled EP library is missing: ${ep_library}" >&2
    exit 2
fi
if ldd "${ep_library}" | grep -q "not found"; then
    ldd "${ep_library}" >&2
    echo "bundled EP dependency closure is incomplete" >&2
    exit 2
fi

exec "${demo_root}/scripts/run_guarded.sh" \
    python3 "${demo_root}/src/moss_spacemit_demo.py" \
    --model-dir "${demo_root}/models/MOSS-TTS-Nano-100M-ONNX-xslim-dynq" \
    "$@"
