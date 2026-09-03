#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The binary may have been built on a different checkout path. Resolve the
# Sherpa-ONNX libraries from this checkout instead of relying on its RUNPATH.
SHERPA_LIB_DIR="${MATCHA_SHERPA_LIB_DIR:-$ROOT/runtime/sherpa_onnx/lib}"
if [[ ! -f "$SHERPA_LIB_DIR/libsherpa-onnx-c-api.so" ]]; then
  echo "error: Sherpa-ONNX C API library not found: $SHERPA_LIB_DIR/libsherpa-onnx-c-api.so" >&2
  echo "set MATCHA_SHERPA_LIB_DIR to the directory containing the Sherpa-ONNX .so files" >&2
  exit 1
fi
export LD_LIBRARY_PATH="$SHERPA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
MODEL_DIR="$ROOT/matcha-model"
BINARY="$ROOT/build-cpp/matcha_tts_capi"

if [[ ! -x "$BINARY" ]]; then
  echo "error: C++ binary not found: $BINARY" >&2
  echo "build it with: cmake --build '$ROOT/build-cpp' -j2" >&2
  exit 1
fi

if [[ ! -f "$MODEL_DIR/model-steps-3.q.onnx" || ! -f "$MODEL_DIR/vocos-16khz-univ.q.onnx" ]]; then
  echo "error: Matcha model files are missing under $MODEL_DIR" >&2
  exit 1
fi

OUTPUT="${1:-$ROOT/output.wav}"
TEXT="${2:-你好，这是开发板上的 C++ 文本转语音测试。}"
SPEED="${3:-1.0}"

exec "$BINARY" "$MODEL_DIR" "$OUTPUT" "$TEXT" "$SPEED"
