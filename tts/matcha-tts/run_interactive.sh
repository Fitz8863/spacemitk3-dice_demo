#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The binary may have been built on a different checkout path. Resolve the
# Sherpa-ONNX libraries from this checkout instead of relying on its RUNPATH.
SHERPA_LIB_DIR="$ROOT/runtime/sherpa_onnx/lib"
if [[ ! -f "$SHERPA_LIB_DIR/libsherpa-onnx-c-api.so" ]]; then
    echo "error: Sherpa-ONNX C API library not found: $SHERPA_LIB_DIR/libsherpa-onnx-c-api.so" >&2
    exit 1
fi
export LD_LIBRARY_PATH="$SHERPA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

MODEL_DIR="$ROOT/matcha-model"
BINARY="$ROOT/build-cpp/matcha_tts_interactive"
if [[ ! -x "$BINARY" ]]; then
    echo "error: binary not found: $BINARY" >&2
    echo "build it with: cmake --build '$ROOT/build-cpp' -j2" >&2
    exit 1
fi

# All TTS controls are command-line options handled by the binary. The model
# directory is passed explicitly so the script itself has no TTS env interface.
exec "$BINARY" --model-dir "$MODEL_DIR" "$@"
