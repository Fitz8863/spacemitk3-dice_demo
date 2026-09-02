#!/usr/bin/env python3
"""Inspect and manage provider runtimes declared by component packages."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from core.components import build_registry
from core.errors import DiceArenaError
from core.games import load_games, require_game, resolve_provider_id
from core.state_schema import iter_speech_actions


ROOT = Path(__file__).resolve().parents[1]


def _command_for(manifest: dict[str, Any], action: str) -> list[str] | None:
    lifecycle = manifest.get("lifecycle", {})
    if not isinstance(lifecycle, dict):
        raise ValueError("manifest lifecycle must be an object")
    command = lifecycle.get(action)
    if command is None:
        return None
    if not isinstance(command, list) or not command or not all(
        isinstance(item, str) and item for item in command
    ):
        raise ValueError(f"lifecycle.{action} must be a non-empty string array")
    resolved = list(command)
    executable = Path(resolved[0])
    if not executable.is_absolute() and "/" in resolved[0]:
        candidate = (ROOT / executable).resolve()
        try:
            candidate.relative_to(ROOT)
        except ValueError as exc:
            raise ValueError(f"lifecycle.{action} executable escapes the project root") from exc
        resolved[0] = str(candidate)
    return resolved


def _health(provider: Any) -> dict[str, Any]:
    try:
        health = provider.health()
    except Exception as exc:
        return {"id": provider.id, "type": provider.type, "ok": False, "error": str(exc)}
    if not isinstance(health, dict):
        return {
            "id": provider.id,
            "type": provider.type,
            "ok": False,
            "error": "health() must return an object",
        }
    normalized = dict(health)
    normalized.setdefault("id", provider.id)
    normalized.setdefault("type", provider.type)
    normalized.setdefault("role", provider.role)
    return normalized


def _selected_provider_id(provider_slot: str, game_id: str) -> str:
    manifest = require_game(load_games(), game_id)
    fallbacks = {"tts_local": "tts_qwen3", "vision_adjudicator": "vision_yolov8_adjudicator"}
    return resolve_provider_id(
        manifest,
        provider_slot,
        fallbacks.get(provider_slot, ""),
    )


def _referenced_tts_providers(manifest: dict[str, Any], *, local_fallback: str) -> list[str]:
    """Collect every TTS provider id this manifest can synthesize through.

    Covers both semantic slots (local/remote) plus any per-action ``provider``
    override on a state-machine speech action (including ``select_by`` cases);
    the default (local slot) provider comes first so start scripts keep a
    stable notion of the game's primary voice.
    """
    ids: list[str] = []

    def add(provider_id: str) -> None:
        provider_id = provider_id.strip()
        if provider_id and provider_id not in ids:
            ids.append(provider_id)

    add(resolve_provider_id(manifest, "tts_local", local_fallback))
    add(resolve_provider_id(manifest, "tts_remote", ""))
    machine = manifest.get("state_machine")
    if isinstance(machine, dict):
        for action in iter_speech_actions(machine):
            provider = action.get("provider")
            if isinstance(provider, str):
                add(provider)
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Dice Arena provider runtime control")
    parser.add_argument(
        "action",
        choices=(
            "list", "selected", "referenced", "health", "start", "stop",
            "start-selected", "stop-selected",
        ),
    )
    parser.add_argument("provider", nargs="?")
    parser.add_argument("--game", default="dice")
    args = parser.parse_args()

    if args.action in {"selected", "referenced", "start-selected", "stop-selected"}:
        if not args.provider:
            parser.error("provider slot is required for selected actions")
        try:
            if args.action == "referenced":
                for referenced_id in _referenced_tts_providers(
                    require_game(load_games(), args.game), local_fallback="tts_qwen3"
                ):
                    print(referenced_id)
                return 0
            selected_id = _selected_provider_id(args.provider, args.game)
        except DiceArenaError as exc:
            print(exc.message, file=sys.stderr)
            return 2
        if not selected_id:
            print(f"No {args.provider} provider is configured for game {args.game}", file=sys.stderr)
            return 2
        if args.action == "selected":
            print(selected_id)
            return 0
        args.action = args.action.removesuffix("-selected")
        args.provider = selected_id

    registry = build_registry(log=False)
    if args.action == "list":
        print(json.dumps({"components": registry.all(include_health=True)}, ensure_ascii=False))
        return 0
    if not args.provider:
        parser.error("provider id is required for health/start/stop")

    try:
        provider = registry.get(args.provider)
        manifest = registry.get_manifest(args.provider)
    except DiceArenaError as exc:
        print(exc.message, file=sys.stderr)
        return 2

    if args.action == "health":
        health = _health(provider)
        print(json.dumps(health, ensure_ascii=False))
        return 0 if health.get("ok") is not False else 1

    if args.action == "start":
        health = _health(provider)
        if health.get("ok") is True:
            print(f"Provider already ready: {provider.id}")
            return 0

    try:
        command = _command_for(manifest, args.action)
    except ValueError as exc:
        print(f"Provider {provider.id} lifecycle is invalid: {exc}", file=sys.stderr)
        return 2
    if command is None:
        print(
            f"Provider {provider.id} has no lifecycle.{args.action} command; "
            "runtime is managed externally",
            file=sys.stderr if args.action == "start" else sys.stdout,
        )
        # Cloud and externally managed providers are valid functional packages.
        # Their health is checked by the provider itself; stop is always a no-op.
        return 0

    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        return completed.returncode
    if args.action == "start":
        health = _health(provider)
        if health.get("ok") is not True:
            print(
                f"Provider {provider.id} start command returned but health is not ready: "
                f"{json.dumps(health, ensure_ascii=False)}",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
