"""Single source of truth for the Qwen3 Dice Arena package settings."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.env import load_board_env
from core.tts_config import config_value, load_component_config, resolve_config_path

load_board_env()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
COMPONENT_DIR = Path(__file__).resolve().parent


def _value(config: dict[str, Any], env_name: str, *keys: str, default: Any) -> Any:
    env_value = os.environ.get(env_name)
    if env_value not in (None, ""):
        return env_value
    return config_value(config, *keys, default=default)


@dataclass(frozen=True)
class QwenSettings:
    root: Path
    model_dir: Path
    url: str
    host: str
    port: int
    speaker_file: str
    default_voice: str
    default_speed: float
    timeout_seconds: float
    chunk_chars: int


def load_settings(config: dict[str, Any] | None = None) -> QwenSettings:
    config = config if config is not None else load_component_config(COMPONENT_DIR)
    default_root = PROJECT_ROOT / "tts" / "qwen3-tts"
    root_value = _value(config, "DICE_QWEN3_TTS_ROOT", "runtime", "root", default=str(default_root))
    root = resolve_config_path(root_value, base_dir=PROJECT_ROOT) or default_root.resolve()
    model_value = _value(config, "DICE_QWEN3_TTS_MODEL_DIR", "runtime", "model_dir", default="qwen3-tts-0.6b")
    model_dir = resolve_config_path(model_value, base_dir=root) or root
    host = str(_value(config, "DICE_QWEN3_TTS_HOST", "runtime", "host", default="127.0.0.1"))
    port = int(_value(config, "DICE_QWEN3_TTS_PORT", "runtime", "port", default=18080))
    url = str(_value(
        config,
        "DICE_QWEN3_TTS_URL",
        "runtime",
        "base_url",
        default=os.environ.get("DICE_TTS_URL", f"http://{host}:{port}"),
    )).rstrip("/")
    speaker_file = str(_value(config, "DICE_QWEN3_TTS_SPEAKER", "voice", "speaker_file", default="anke.spk.bin"))
    default_voice = str(_value(config, "DICE_QWEN3_TTS_DEFAULT_VOICE", "voice", "default", default="default"))
    default_speed = float(_value(config, "DICE_QWEN3_TTS_SPEED", "generation", "speed", default=1.0))
    timeout_seconds = float(_value(config, "DICE_QWEN3_TTS_TIMEOUT_SECONDS", "generation", "timeout_seconds", default=120))
    chunk_chars = max(8, int(_value(config, "DICE_QWEN3_TTS_CHUNK_CHARS", "generation", "chunk_chars", default=24)))
    if not (0.25 <= default_speed <= 4.0):
        raise ValueError("Qwen3 generation.speed must be between 0.25 and 4.0")
    if port <= 0 or timeout_seconds <= 0:
        raise ValueError("Qwen3 runtime port and timeout must be positive")
    return QwenSettings(
        root=root.resolve(),
        model_dir=model_dir.resolve(),
        url=url,
        host=host,
        port=port,
        speaker_file=speaker_file,
        default_voice=default_voice,
        default_speed=default_speed,
        timeout_seconds=timeout_seconds,
        chunk_chars=chunk_chars,
    )
