"""Single source of truth for the MOSS-TTS-Nano Dice Arena package settings."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.tts_config import config_value, load_component_config, resolve_config_path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
COMPONENT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class MossSettings:
    root: Path
    model_dir: Path
    base_url: str
    host: str
    port: int
    voice_mode: str
    voice: str
    reference_audio: Path | None
    max_new_frames: int
    voice_clone_max_text_tokens: int
    first_chunk_text_tokens: int
    warmup_text: str
    seed: int
    start_timeout_seconds: int
    request_timeout_seconds: float
    ep_intra_thread_num: int
    ep_inter_thread_num: int
    ep_intra_thread_affinity: str
    ep_disable_op_type_filter: str


def resolve_model_dir(root: Path, configured: str | Path | None = None) -> Path:
    """Find the first packaged MOSS model layout when config leaves it blank."""
    if configured not in (None, ""):
        model_dir = Path(str(configured)).expanduser()
        if not model_dir.is_absolute():
            model_dir = root / model_dir
        return model_dir.resolve()
    candidates = (
        root / "models" / "MOSS-TTS-Nano-100M-ONNX-xslim-dynq",
        root / "models" / "MOSS-TTS-Nano-100M-ONNX",
    )
    for candidate in candidates:
        if (candidate / "browser_poc_manifest.json").is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def load_settings(config: dict[str, Any] | None = None) -> MossSettings:
    config = config if config is not None else load_component_config(COMPONENT_DIR)
    default_root = PROJECT_ROOT / "tts" / "moss-tts-nano"
    root = resolve_config_path(
        config_value(config, "runtime", "root", default=str(default_root)),
        base_dir=PROJECT_ROOT,
    ) or default_root.resolve()
    model_dir = resolve_model_dir(root, config_value(config, "runtime", "model_dir", default=""))
    host = str(config_value(config, "runtime", "host", default="127.0.0.1"))
    port = int(config_value(config, "runtime", "port", default=18082))
    base_url = str(config_value(
        config,
        "runtime",
        "base_url",
        default=f"http://{host}:{port}",
    )).rstrip("/")
    voice_mode = str(config_value(config, "voice", "mode", default="builtin")).strip().lower()
    voice = str(config_value(config, "voice", "name", default="Junhao")).strip() or "Junhao"
    reference_value = (
        config_value(config, "voice", "reference_audio", default="")
        if voice_mode == "clone"
        else None
    )
    reference_audio = (
        resolve_config_path(reference_value, base_dir=root)
        if voice_mode == "clone"
        else None
    )
    max_new_frames = int(config_value(config, "generation", "max_new_frames", default=120))
    clone_tokens = int(config_value(config, "generation", "voice_clone_max_text_tokens", default=24))
    first_chunk = int(config_value(config, "generation", "first_chunk_text_tokens", default=16))
    warmup_text = str(config_value(config, "startup", "warmup_text", default="你好，这是 MOSS TTS Nano 在 K3 上的演示。"))
    seed = int(config_value(config, "generation", "seed", default=1234))
    start_timeout = int(config_value(config, "startup", "start_timeout_seconds", default=300))
    request_timeout = float(config_value(config, "limits", "request_timeout_seconds", default=120))
    ep_intra = int(config_value(config, "execution_provider", "intra_thread_num", default=4))
    ep_inter = int(config_value(config, "execution_provider", "inter_thread_num", default=1))
    ep_affinity = str(config_value(config, "execution_provider", "intra_thread_affinity", default="8;9;10;11"))
    ep_filter = str(config_value(config, "execution_provider", "disable_op_type_filter", default=""))
    if voice_mode not in {"builtin", "clone"}:
        raise ValueError("MOSS voice.mode must be builtin or clone")
    if voice_mode == "clone" and reference_audio is None:
        raise ValueError("MOSS clone mode requires voice.reference_audio")
    if port <= 0 or max_new_frames <= 0 or clone_tokens <= 0 or first_chunk < 0:
        raise ValueError("MOSS runtime and generation limits are invalid")
    if start_timeout <= 0 or request_timeout <= 0 or ep_intra <= 0 or ep_inter <= 0:
        raise ValueError("MOSS startup, timeout, and EP thread settings must be positive")
    return MossSettings(
        root=root.resolve(),
        model_dir=(model_dir or root).resolve(),
        base_url=base_url,
        host=host,
        port=port,
        voice_mode=voice_mode,
        voice=voice,
        reference_audio=reference_audio,
        max_new_frames=max_new_frames,
        voice_clone_max_text_tokens=clone_tokens,
        first_chunk_text_tokens=first_chunk,
        warmup_text=warmup_text,
        seed=seed,
        start_timeout_seconds=start_timeout,
        request_timeout_seconds=request_timeout,
        ep_intra_thread_num=ep_intra,
        ep_inter_thread_num=ep_inter,
        ep_intra_thread_affinity=ep_affinity,
        ep_disable_op_type_filter=ep_filter,
    )
