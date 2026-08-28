# Repository Guidelines

## Project Structure & Module Organization

This packaged MOSS-TTS-Nano SpaceMIT EP demo targets K3 `riscv64`. Python is in `src/`, models/tokenizers in `models/`, bundled packages in `python/`, libraries in `lib/`, headers in `include/`, reference audio in `assets/` and `voice/`, and generated audio in `outputs/`. Supported entry points are `run_*.sh`; `scripts/run_guarded.sh` handles TCM cleanup and process safety.

## Upstream Reference vs. Board Delivery

The [official OpenMOSS repository](https://github.com/OpenMOSS/MOSS-TTS-Nano) is the behavior/model-layout reference, with PyTorch, ONNX CPU, web-demo, export, and finetuning code. This checkout is a K3 delivery: it bundles dependencies, uses SpaceMIT EP, pins CPU affinity, supports text-chunk PCM playback, and guards TCM lifecycle. It is not frame-level codec streaming. Port upstream fixes selectively and revalidate on K3.

## Build, Test, and Development Commands

There is no Makefile, CMake project, or install step. Run from the repository root on K3:

```bash
sha256sum -c SHA256SUMS
python3 -m py_compile src/*.py src/moss_tts_nano/*.py
python3 src/tts_robust_normalizer_single_script.py
./run_demo.sh --text "你好，这是一次测试。" --output outputs/test.wav
./run_interactive.sh --output-dir outputs/interactive-test
```

Use `./run_voice_clone.sh` for the bundled voice. Add `--verbose` or `--report-json path.json` for diagnostics. Requirements include Python 3.14, vendor ONNX Runtime (`+spacemit`), `libsndfile`, `aplay`, `spacemit-tcm-smi`, `flock`, and `timeout`; do not use the upstream CPU/PyTorch setup here.

## Coding Style & Naming Conventions

Use Python 3.14 syntax, four-space indentation, type annotations, `snake_case`, `PascalCase` classes, and uppercase constants. Keep Bash on `set -euo pipefail`, quote paths, and preserve root-relative setup. Keep model names, CLI flags, and environment variables stable. Do not casually replace bundled artifacts.

## Testing Guidelines

No external framework or coverage threshold is configured. Python changes should pass compilation and the normalizer self-test. Runtime changes should run a short K3 synthesis, verify PCM playback (and a WAV only when requested), and confirm TCM release. Test touched interactive/voice-clone paths; use a separate `outputs/` subdirectory.

## Commit & Pull Request Guidelines

Local Git branches keep the decode-cache A/B variants: `main` (dynamic), `kv-fixed-320`, `kv-fixed-512`, and `kv-fixed-1024`. Use an imperative subject such as `Add PCM chunk playback`; run `git status` before switching branches. Include summary, validation commands, target runtime, and report paths. Never commit credentials, private audio, or regenerated artifacts without approval.

## Security & Configuration Tips

Verify `SHA256SUMS` before testing. Keep custom recordings and generated audio local unless distribution is intended. Do not substitute ordinary PyPI ONNX Runtime.
