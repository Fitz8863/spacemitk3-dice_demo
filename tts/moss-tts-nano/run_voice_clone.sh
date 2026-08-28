#!/usr/bin/env bash

set -euo pipefail

demo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec "${demo_root}/run_demo.sh" \
    --reference-audio "${demo_root}/assets/warm-female-reference.wav" \
    --text "你好，这是 MOSS TTS Nano 在 K3 上的声音克隆演示。" \
    --output "${demo_root}/outputs/voice-clone.wav" \
    "$@"
