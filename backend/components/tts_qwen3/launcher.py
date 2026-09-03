"""Lifecycle adapter for the Qwen3 runtime.

The provider package owns configuration and the runtime overlay.  The model
delivery under ``tts/qwen3-tts`` remains a dumb executable runtime and is not
aware of Dice Arena component configuration.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
from components.tts_qwen3.settings import load_settings


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _prepare_model_overlay(model_dir: Path, speaker_file: str) -> Path:
    """Create an ignored model overlay when a custom speaker is configured."""
    speaker_path = Path(speaker_file).expanduser()
    if not speaker_path.is_absolute():
        speaker_path = model_dir / speaker_path
    speaker_path = speaker_path.resolve()
    if not speaker_path.is_file():
        raise SystemExit(f"Qwen3-TTS speaker file is missing: {speaker_path}")

    runtime_dir = Path(os.environ.get("DICE_RUNTIME_DIR", PROJECT_ROOT / ".runtime"))
    overlay = runtime_dir / "qwen3-tts-config"
    if overlay.exists() or overlay.is_symlink():
        if overlay.is_symlink() or not overlay.is_dir():
            overlay.unlink()
        else:
            shutil.rmtree(overlay)
    overlay.mkdir(parents=True, exist_ok=True)

    for item in model_dir.iterdir():
        if item.name == "config.json":
            continue
        (overlay / item.name).symlink_to(item)

    # The runtime config stores only a basename; link an external embedding
    # into the overlay without modifying the checked-in model directory.
    speaker_name = speaker_path.name
    target = overlay / speaker_name
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(speaker_path)
    source_config = model_dir / "config.json"
    if not source_config.is_file():
        raise SystemExit(f"Qwen3-TTS model config is missing: {source_config}")
    try:
        config = json.loads(source_config.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"unable to read Qwen3-TTS model config: {exc}") from exc
    config.setdefault("tts_model", {})["speaker_file"] = speaker_name
    (overlay / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return overlay


def main() -> int:
    stop = bool(sys.argv[1:]) and sys.argv[1] == "--stop"
    forwarded = sys.argv[2:] if stop else sys.argv[1:]
    settings = load_settings()
    runtime_root = settings.root
    script = runtime_root / ("stop_server.sh" if stop else "start_server.sh")
    if not script.is_file():
        raise SystemExit(f"Qwen3-TTS runtime script is missing: {script}")

    env = os.environ.copy()
    env.update({
        "QWEN3_TTS_HOST": settings.host,
        "QWEN3_TTS_PORT": str(settings.port),
        "DICE_QWEN3_TTS_ROOT": str(runtime_root),
    })
    if not stop:
        env["QWEN3_TTS_MODEL"] = str(
            _prepare_model_overlay(settings.model_dir, settings.speaker_file)
        )
    completed = subprocess.run([str(script), *forwarded], cwd=PROJECT_ROOT, env=env)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
