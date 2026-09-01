"""Single source of truth for the Qwen3 Dice Arena package settings."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.tts_config import config_value, load_component_config, resolve_config_path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
COMPONENT_DIR = Path(__file__).resolve().parent


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
    root_value = config_value(config, "runtime", "root", default=str(default_root))
    root = resolve_config_path(root_value, base_dir=PROJECT_ROOT) or default_root.resolve()
    model_value = config_value(config, "runtime", "model_dir", default="qwen3-tts-0.6b")
    model_dir = resolve_config_path(model_value, base_dir=root) or root
    host = str(config_value(config, "runtime", "host", default="127.0.0.1"))
    port = int(config_value(config, "runtime", "port", default=18080))
    url = str(config_value(
        config,
        "runtime",
        "base_url",
        default=f"http://{host}:{port}",
    )).rstrip("/")
    speaker_file = str(config_value(config, "voice", "speaker_file", default="anke.spk.bin"))
    default_voice = str(config_value(config, "voice", "default", default="default"))
    default_speed = float(config_value(config, "generation", "speed", default=1.0))
    timeout_seconds = float(config_value(config, "generation", "timeout_seconds", default=120))
    chunk_chars = max(8, int(config_value(config, "generation", "chunk_chars", default=24)))
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
