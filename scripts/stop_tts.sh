#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TTS_DIR="${DICE_TTS_ROOT:-${ROOT_DIR}/tts/qwen3-tts}"

if [[ ! -x "${TTS_DIR}/stop_server.sh" ]]; then
    echo "Missing migrated Qwen3-TTS stopper: ${TTS_DIR}/stop_server.sh" >&2
    exit 1
fi

exec "${TTS_DIR}/stop_server.sh"
