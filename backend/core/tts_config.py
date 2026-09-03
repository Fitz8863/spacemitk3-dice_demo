"""Shared loading and path resolution for component-local TTS configuration.

Every TTS adapter owns a small ``config.json`` next to its ``provider.py``.
Values in the process environment or explicit CLI arguments still take
precedence, which keeps existing board deployments backwards compatible.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


class TtsConfigError(ValueError):
    """Raised when a component TTS config is missing or malformed."""


_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _is_loopback_http_url(base_url: str) -> bool:
    parsed = urlsplit(base_url)
    return parsed.scheme == "http" and parsed.hostname in _LOOPBACK_HOSTS


def load_component_config(component_dir: Path) -> dict[str, Any]:
    """Load ``config.json`` from one component package.

    A missing config is an error rather than silently falling back to a
    different component's settings. This makes adding a new TTS package
    predictable and catches deployment mistakes at startup.
    """
    path = component_dir / "config.json"
    if not path.is_file():
        raise TtsConfigError(f"TTS component config is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TtsConfigError(f"unable to read TTS component config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TtsConfigError(f"TTS component config must be a JSON object: {path}")
    validate_tts_component_config(payload)
    return payload


def config_value(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Read a nested config value, returning *default* for absent sections."""
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def resolve_config_path(value: str | Path | None, *, base_dir: Path) -> Path | None:
    """Resolve a config path relative to its documented component base.

    Absolute paths remain supported for board-local assets, but are never
    written into the checked-in defaults. ``~`` is expanded for compatibility
    with existing local deployments.
    """
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def config_relative_path(value: str | Path | None, *, base_dir: Path) -> str | None:
    """Return a stable display path relative to *base_dir* when possible."""
    resolved = resolve_config_path(value, base_dir=base_dir)
    if resolved is None:
        return None
    try:
        return str(resolved.relative_to(base_dir.resolve()))
    except ValueError:
        return str(resolved)


def validate_tts_component_config(config: dict[str, Any]) -> None:
    """Validate the small common section shared by local and cloud providers."""
    if not isinstance(config, dict):
        raise TtsConfigError("TTS component config must be an object")
    runtime = config.get("runtime", {})
    if not isinstance(runtime, dict):
        raise TtsConfigError("TTS config runtime must be an object")
    kind = runtime.get("kind", "local")
    if kind not in {"local", "cloud", "external"}:
        raise TtsConfigError("TTS runtime.kind must be local, cloud, or external")

    base_url = runtime.get("base_url", runtime.get("url"))
    if base_url is not None:
        if not isinstance(base_url, str):
            raise TtsConfigError("TTS runtime.base_url must be a string")
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise TtsConfigError("TTS runtime.base_url must be an absolute HTTP(S) URL")
    elif kind in {"cloud", "external"}:
        raise TtsConfigError("cloud/external TTS runtimes require runtime.base_url")
    if kind == "local":
        # Local runtimes are board-internal bridges; restricting them to a
        # loopback origin keeps a misconfigured URL from turning the backend
        # into a request proxy.
        if base_url is not None and not _is_loopback_http_url(base_url):
            raise TtsConfigError("local TTS runtime.base_url must be an http:// loopback address")
        host = runtime.get("host")
        if host is not None and str(host) not in _LOOPBACK_HOSTS:
            raise TtsConfigError("local TTS runtime.host must be a loopback address")

    voice = config.get("voice", {})
    if voice is not None and not isinstance(voice, dict):
        raise TtsConfigError("TTS config voice must be an object")
    generation = config.get("generation", {})
    if generation is not None and not isinstance(generation, dict):
        raise TtsConfigError("TTS config generation must be an object")
