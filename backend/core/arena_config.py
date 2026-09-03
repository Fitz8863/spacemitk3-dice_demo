"""Project-wide (arena) configuration: deployment defaults under game manifests.

``backend/config.json`` describes what kind of machine this deployment is:
which engine fills each provider slot, the default TTS voice/speed, and the
global ASR breaker.  Game manifests override any of it per game, and speech
entries can pin a provider per line.  The local TTS selection is additionally
frozen at process start — switching engines requires a restart.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.participants import normalize_participants

ROOT = Path(__file__).resolve().parents[2]
ARENA_CONFIG_PATH = ROOT / "backend" / "config.json"


class ArenaConfigError(ValueError):
    """Raised when backend/config.json is malformed or self-contradictory."""


def validate_arena_config(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ArenaConfigError("arena config must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ArenaConfigError("arena config schema_version must be 1")
    providers = payload.get("providers", {})
    if not isinstance(providers, dict) or not all(
        isinstance(key, str) and key and isinstance(value, str) and value.strip()
        for key, value in providers.items()
    ):
        raise ArenaConfigError("providers must map slots to non-empty provider ids")
    participants = payload.get("participants")
    if participants is not None:
        try:
            normalize_participants(participants)
        except ValueError as exc:
            raise ArenaConfigError(f"participants is invalid: {exc}") from exc
    voice = payload.get("voice")
    if voice is not None and (not isinstance(voice, str) or not voice.strip()):
        raise ArenaConfigError("voice must be a non-empty string")
    speed = payload.get("speed")
    if speed is not None and (
        not isinstance(speed, (int, float)) or isinstance(speed, bool) or speed <= 0
    ):
        raise ArenaConfigError("speed must be a positive number")
    asr_enabled = payload.get("asr_enabled", True)
    if not isinstance(asr_enabled, bool):
        raise ArenaConfigError("asr_enabled must be boolean")
    standby = payload.get("standby", {})
    if not isinstance(standby, dict):
        raise ArenaConfigError("standby must be an object")
    if standby.get("enabled", True) is True and "wake_phrases" in standby:
        phrases = standby["wake_phrases"]
        if (
            not isinstance(phrases, list)
            or not phrases
            or not all(isinstance(word, str) and word.strip() for word in phrases)
        ):
            raise ArenaConfigError("standby.wake_phrases must be a non-empty list of words")
    idle_seconds = standby.get("idle_seconds", 120)
    if (
        not isinstance(idle_seconds, (int, float))
        or isinstance(idle_seconds, bool)
        or not (5 <= idle_seconds <= 3600)
    ):
        raise ArenaConfigError("standby.idle_seconds must be between 5 and 3600 seconds")
    for field in ("enabled", "boot_standby"):
        if field in standby and not isinstance(standby[field], bool):
            raise ArenaConfigError(f"standby.{field} must be boolean")
    return payload


def arena_standby(arena: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalized standby screen settings (browser-safe defaults included)."""
    standby = (arena or {}).get("standby")
    if not isinstance(standby, Mapping):
        standby = {}
    enabled = standby.get("enabled", True)
    boot = standby.get("boot_standby", False)
    return {
        "enabled": enabled if isinstance(enabled, bool) else True,
        "idle_seconds": standby.get("idle_seconds", 120),
        "boot_standby": boot if isinstance(boot, bool) else False,
        "wake_phrases": list(standby.get("wake_phrases") or []),
    }


def load_arena_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load the arena config; a missing file means "no global defaults".

    A present-but-broken file raises so a typo cannot silently disable the
    deployment defaults (the caller keeps the last good config).
    """
    config_path = Path(path) if path is not None else ARENA_CONFIG_PATH
    if not config_path.is_file():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArenaConfigError(f"cannot read arena config {config_path}: {exc}") from exc
    return validate_arena_config(payload)


def arena_slot_value(arena: Mapping[str, Any] | None, slot: str) -> str:
    providers = (arena or {}).get("providers")
    if isinstance(providers, Mapping):
        value = providers.get(slot)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def arena_asr_enabled(arena: Mapping[str, Any] | None) -> bool:
    value = (arena or {}).get("asr_enabled", True)
    return value if isinstance(value, bool) else True


def with_global_defaults(
    manifest: Mapping[str, Any], arena: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Underlay arena defaults beneath one game manifest.

    Per field the game manifest wins; the arena fills only what it leaves
    out.  The ASR section is the exception: the arena breaker ANDs with the
    game's own switch.  Returns a fresh dict — the source manifest and the
    games registry are never mutated.
    """
    merged = dict(manifest)
    arena = arena or {}
    global_providers = arena.get("providers")
    if isinstance(global_providers, Mapping) and global_providers:
        game_providers = merged.get("providers")
        providers = dict(global_providers)
        if isinstance(game_providers, Mapping):
            providers.update(game_providers)
        merged["providers"] = providers
    if "participants" not in merged and isinstance(arena.get("participants"), Mapping):
        # The player/Agent physical-side mapping is a property of the table
        # setup, not of any game; games may still override it.
        merged["participants"] = dict(arena["participants"])
    if "voice" not in merged and isinstance(arena.get("voice"), str) and arena["voice"]:
        merged["voice"] = arena["voice"]
    if "speed" not in merged:
        speed = arena.get("speed")
        if isinstance(speed, (int, float)) and not isinstance(speed, bool):
            merged["speed"] = speed
    asr = merged.get("asr")
    if isinstance(asr, dict):
        asr = dict(asr)
        if not arena_asr_enabled(arena):
            asr["enabled"] = False
        merged["asr"] = asr
    return merged


def resolve_local_tts_pin(
    arena: Mapping[str, Any] | None,
    game_manifests: Iterable[Mapping[str, Any]],
) -> str | None:
    """Resolve the single local TTS engine this process may run.

    Candidates are the arena ``tts_local`` slot plus every *enabled* game's
    own override.  Exactly one distinct id is allowed: the local engines are
    heavy resident processes, the demo cannot honor a mid-run switch, so a
    conflicting configuration must fail loudly at startup.
    """
    candidates: list[str] = []
    arena_value = arena_slot_value(arena, "tts_local")
    if arena_value:
        candidates.append(arena_value)
    for manifest in game_manifests:
        if not isinstance(manifest, Mapping) or not manifest.get("enabled", False):
            continue
        providers = manifest.get("providers")
        value = providers.get("tts_local") if isinstance(providers, Mapping) else None
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    distinct = sorted(set(candidates))
    if len(distinct) > 1:
        raise ArenaConfigError(
            "multiple local TTS engines configured: "
            f"{', '.join(distinct)} — keep exactly one (arena providers.tts_local "
            "plus optional per-game overrides) and restart"
        )
    return distinct[0] if distinct else None
