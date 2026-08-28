"""Game manifest registry and pipeline dispatch."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any, Callable

from core.errors import GameDisabledError, GameNotFoundError

ROOT = Path(__file__).resolve().parents[2]
GAMES_ROOT = ROOT / "backend" / "games"

_PROVIDER_SLOT_ALIASES = {
    # Migration alias for manifests written before visual roles were explicit.
    "vision_adjudicator": ("vision",),
}
_PROVIDER_ENV_ALIASES = {
    "vision_adjudicator": ("DICE_VISION_PROVIDER",),
}


class GameRegistry:
    def __init__(self) -> None:
        self._games: dict[str, dict[str, Any]] = {}

    def register(self, manifest: dict[str, Any]) -> None:
        self._games[manifest["id"]] = manifest

    def get(self, game_id: str) -> dict[str, Any]:
        manifest = self._games.get(game_id)
        if manifest is None:
            raise GameNotFoundError(game_id)
        return manifest

    def all(self) -> list[dict[str, Any]]:
        return list(self._games.values())


def load_games() -> GameRegistry:
    registry = GameRegistry()
    if not GAMES_ROOT.is_dir():
        return registry
    for manifest_path in sorted(GAMES_ROOT.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            game_id = manifest.get("id")
            if not isinstance(game_id, str) or not game_id.isidentifier():
                raise ValueError("invalid id")
            if not isinstance(manifest.get("name"), str):
                raise ValueError("missing or invalid name")
            if not isinstance(manifest.get("enabled"), bool):
                raise ValueError("missing or invalid enabled")
            texts = manifest.get("texts", {})
            if not isinstance(texts, dict):
                raise ValueError("texts must be an object")
            manifest["texts"] = texts

            if "components" in manifest:
                legacy_components = manifest["components"]
                if not isinstance(legacy_components, list) or not all(
                    isinstance(item, str) and item for item in legacy_components
                ):
                    raise ValueError("components must be a list of ids")

            providers = manifest.get("providers", {})
            if not isinstance(providers, dict) or not all(
                isinstance(key, str) and isinstance(value, str) and value
                for key, value in providers.items()
            ):
                raise ValueError("providers must map semantic slots to provider ids")
            # Keep the old flat tts_provider spelling as a migration alias.
            if "tts" not in providers and isinstance(manifest.get("tts_provider"), str):
                providers["tts"] = manifest["tts_provider"]
            manifest["providers"] = providers
            registry.register(manifest)
        except (OSError, ValueError, TypeError) as exc:
            print(f"[games] skip {manifest_path}: {exc}", flush=True)
    return registry


def require_game(registry: GameRegistry, game_id: str) -> dict[str, Any]:
    manifest = registry.get(game_id)
    if not manifest.get("enabled", False):
        raise GameDisabledError(game_id)
    return manifest


def resolve_provider_id(
    manifest: dict[str, Any],
    provider_slot: str,
    fallback: str = "",
) -> str:
    """Resolve one semantic provider slot from env and game configuration.

    Slots describe responsibility (for example ``vision_adjudicator``), not
    implementation technology (for example YOLO). Canonical configuration is
    checked first, followed by narrowly scoped migration aliases.
    """
    env_names = (
        f"DICE_{provider_slot.upper()}_PROVIDER",
        *_PROVIDER_ENV_ALIASES.get(provider_slot, ()),
    )
    for env_name in env_names:
        override = os.environ.get(env_name, "").strip()
        if override:
            return override

    providers = manifest.get("providers", {})
    if isinstance(providers, dict):
        for slot in (provider_slot, *_PROVIDER_SLOT_ALIASES.get(provider_slot, ())):
            configured = providers.get(slot)
            if isinstance(configured, str) and configured.strip():
                return configured.strip()
    return fallback


def run_game(
    registry: GameRegistry,
    game_id: str,
    on_log: Callable[[str], None],
    is_cancelled: Callable[[], bool],
    on_event: Callable[[dict[str, Any]], None],
    timeout_seconds: float,
    components: Any,
) -> dict[str, Any]:
    """Run a game's pipeline with the shared provider registry injected."""
    manifest = require_game(registry, game_id)
    module = importlib.import_module(f"games.{game_id}.pipeline")
    return module.run(
        on_log,
        is_cancelled,
        timeout_seconds,
        components=components,
        manifest=manifest,
        on_event=on_event,
    )
