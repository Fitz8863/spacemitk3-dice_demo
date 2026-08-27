"""Game registry: scan ``backend/games/*/manifest.json`` into a lookup.

Each game declares its id, display metadata, TTS texts, and the components it
orchestrates. The backend exposes the list (``GET /api/games``) and routes
``/api/analyze`` to a game pipeline. This module stays thin: game logic lives
in each game's ``pipeline.py``, and components stay game-agnostic.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Callable

from core.errors import GameConfigError, GameDisabledError, GameNotFoundError

ROOT = Path(__file__).resolve().parents[2]  # repo root (main/)
GAMES_ROOT = ROOT / "backend" / "games"


class GameRegistry:
    def __init__(self) -> None:
        self._games: dict[str, dict[str, Any]] = {}

    def register(self, manifest: dict[str, Any]) -> None:
        self._games[manifest["id"]] = manifest

    def get(self, game_id: str) -> dict[str, Any]:
        """Get a game manifest by ID, raises GameNotFoundError if not found."""
        manifest = self._games.get(game_id)
        if manifest is None:
            raise GameNotFoundError(game_id)
        return manifest

    def all(self) -> list[dict[str, Any]]:
        return list(self._games.values())


def load_games() -> GameRegistry:
    """Scan the games directory for manifests and build the registry."""
    registry = GameRegistry()
    if not GAMES_ROOT.is_dir():
        return registry
    for manifest_path in sorted(GAMES_ROOT.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"[games] skip {manifest_path}: {exc}", flush=True)
            continue
        game_id = manifest.get("id")
        # ``game_id`` later feeds importlib, so keep it a plain identifier.
        if not isinstance(game_id, str) or not game_id.isidentifier():
            print(f"[games] skip {manifest_path}: invalid id", flush=True)
            continue

        # Basic validation
        if "name" not in manifest or not isinstance(manifest["name"], str):
            print(f"[games] skip {manifest_path}: missing or invalid 'name'", flush=True)
            continue
        if "enabled" not in manifest or not isinstance(manifest["enabled"], bool):
            print(f"[games] skip {manifest_path}: missing or invalid 'enabled'", flush=True)
            continue

        manifest["texts"] = manifest.get("texts", {})
        manifest["components"] = manifest.get("components", [])
        registry.register(manifest)
    return registry


def require_game(registry: GameRegistry, game_id: str) -> dict[str, Any]:
    """Return an enabled game's manifest or raise GameNotFoundError/GameDisabledError."""
    manifest = registry.get(game_id)  # raises GameNotFoundError if not found
    if not manifest.get("enabled", False):
        raise GameDisabledError(game_id)
    return manifest


def run_game(
    registry: GameRegistry,
    game_id: str,
    on_log: Callable[[str], None],
    is_cancelled: Callable[[], bool],
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run ``games/<id>/pipeline.py:run`` after validating the game is enabled."""
    require_game(registry, game_id)
    module = importlib.import_module(f"games.{game_id}.pipeline")
    return module.run(on_log, is_cancelled, timeout_seconds)
