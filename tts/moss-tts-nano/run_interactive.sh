#!/usr/bin/env bash

set -euo pipefail

demo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MOSS_TTS_INTERACTIVE=1

exec "${demo_root}/run_demo.sh" --interactive "$@"
