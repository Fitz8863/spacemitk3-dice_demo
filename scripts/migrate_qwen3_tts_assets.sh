#!/usr/bin/env bash
set -euo pipefail

# Copy the board-local Qwen3-TTS deployment into this Dice Arena checkout.
# Model weights and speaker embeddings remain ignored by Git; this script is
# intentionally explicit so private reference audio is never copied by a
# blanket directory sync.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_ROOT="${DICE_TTS_ROOT:-${ROOT_DIR}/tts/qwen3-tts}"
PYTHON_BIN="${DICE_PYTHON:-python3}"
# The source is external input and must be explicit. The destination is always
# derived from this checkout unless the caller deliberately overrides it.
SOURCE_ROOT="${QWEN3_TTS_SOURCE:-}"

display_path() {
    local path="$1"
    if [[ "$path" == "$ROOT_DIR"/* ]]; then
        printf '%s' "${path#"$ROOT_DIR"/}"
    else
        printf '%s' "$path"
    fi
}

usage() {
    cat <<USAGE
Usage: $(basename "$0") [--source PATH] [--dest PATH]

Defaults:
  source:  (required; use --source PATH or QWEN3_TTS_SOURCE)
  dest:   $(display_path "$DEST_ROOT")

Copies the Qwen3-TTS launcher, Python client, docs, riscv64 runtime, model
metadata/weights, and the speaker file referenced by qwen3-tts-0.6b/config.json.
It does not copy voice_presets/source_audio or arbitrary speaker embeddings.
USAGE
}

while (($#)); do
    case "$1" in
        --source)
            [[ $# -ge 2 ]] || { echo "--source requires a path" >&2; exit 2; }
            SOURCE_ROOT="$2"; shift 2 ;;
        --dest)
            [[ $# -ge 2 ]] || { echo "--dest requires a path" >&2; exit 2; }
            DEST_ROOT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$SOURCE_ROOT" ]]; then
    echo "Qwen3-TTS source is required; use --source PATH or QWEN3_TTS_SOURCE" >&2
    usage >&2
    exit 2
fi
SOURCE_ROOT="$(cd -- "$SOURCE_ROOT" 2>/dev/null && pwd)" || { echo "Source does not exist: $SOURCE_ROOT" >&2; exit 1; }
[[ -f "$SOURCE_ROOT/qwen3-tts-0.6b/config.json" ]] || {
    echo "Missing source model config: $SOURCE_ROOT/qwen3-tts-0.6b/config.json" >&2
    exit 1
}

mkdir -p "$DEST_ROOT"
copy_path() {
    local relative="$1"
    local source="$SOURCE_ROOT/$relative"
    local dest="$DEST_ROOT/$relative"
    [[ -e "$source" || -L "$source" ]] || return 0
    mkdir -p "$(dirname "$dest")"
    if [[ -d "$source" && ! -L "$source" ]]; then
        mkdir -p "$dest"
        cp -a "$source"/. "$dest"/
    else
        cp -a "$source" "$dest"
    fi
    echo "copied: $relative"
}

for path in \
    AGENTS.md README.md .gitignore start_server.sh stop_server.sh run_interactive.sh \
    qwen3_tts_interactive.py runtime docs/build-llama-realtime-k3.md \
    patches/llama.cpp-realtime.patch voice_embeddings voice_presets/README.md \
    voice_presets/manifest.json; do
    copy_path "$path"
done

# Preserve the model directory but never assume every .bin is safe to copy.
for path in qwen3-tts-0.6b/LICENSE qwen3-tts-0.6b/SHA256SUMS \
    qwen3-tts-0.6b/manifest.json qwen3-tts-0.6b/config.json; do
    copy_path "$path"
done
for path in "$SOURCE_ROOT"/qwen3-tts-0.6b/*.onnx "$SOURCE_ROOT"/qwen3-tts-0.6b/*.gguf; do
    [[ -e "$path" ]] || continue
    copy_path "qwen3-tts-0.6b/$(basename "$path")"
done

speaker_file="$("$PYTHON_BIN" - "$SOURCE_ROOT/qwen3-tts-0.6b/config.json" <<'PY'
import json
import sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
name = config.get("tts_model", {}).get("speaker_file", "")
if not name or Path(name).name != name:
    raise SystemExit("speaker_file must be a plain filename")
print(name)
PY
)"
copy_path "qwen3-tts-0.6b/${speaker_file}"

cat <<SUMMARY
Migration complete.
  source: $SOURCE_ROOT
  dest:   $(display_path "$DEST_ROOT")

Private source audio and unconfigured speaker embeddings were not copied.
Verify with:
  $ROOT_DIR/scripts/start_tts.sh
  curl -fsS http://127.0.0.1:18080/health
SUMMARY
