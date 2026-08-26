#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if ! curl -fsS --max-time 2 "http://${QWEN3_TTS_HOST:-127.0.0.1}:${QWEN3_TTS_PORT:-18080}/health" >/dev/null 2>&1; then
    "$ROOT/start_server.sh"
fi
exec python3 "$ROOT/qwen3_tts_interactive.py" "$@"
