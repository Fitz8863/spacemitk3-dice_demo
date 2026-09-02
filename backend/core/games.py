"""Game manifest registry and pipeline dispatch."""
from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from typing import Any, Callable

from core.errors import GameDisabledError, GameNotFoundError
from core.participants import normalize_participants
from core.state_schema import validate_state_machine

ROOT = Path(__file__).resolve().parents[2]
GAMES_ROOT = ROOT / "backend" / "games"

_PROVIDER_SLOT_ALIASES = {
    # Migration alias for manifests written before visual roles were explicit.
    "vision_adjudicator": ("vision",),
}


def resolve_game_audio_path(game_id: str, audio: str, root: Path | None = None) -> Path:
    """Resolve a manifest audio path without allowing it to escape its game.

    ``root`` overrides the module-level games root so a hot-reloaded or
    injected registry can serve audio from its own directory.
    """
    game_root = ((root or GAMES_ROOT) / game_id).resolve()
    relative = Path(audio)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".wav":
        raise ValueError("audio path must be a relative .wav path")
    candidate = (game_root / relative).resolve()
    try:
        candidate.relative_to(game_root)
    except ValueError as exc:
        raise ValueError("audio path escapes the game directory") from exc
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def render_speech_text(template: str, values: Any) -> str:
    """Render scalar result values into a manifest-owned TTS template."""
    if not isinstance(values, dict):
        raise ValueError("speech values must be an object")
    normalized: dict[str, str] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not isinstance(value, (str, int, float)):
            raise ValueError("speech values must contain only scalar fields")
        normalized[key] = str(value)
    return re.sub(
        r"\{([a-zA-Z0-9_]+)\}",
        lambda match: normalized.get(match.group(1), match.group(0)),
        template,
    )


def validate_asr_section(asr: Any, machine: dict[str, Any]) -> dict[str, Any]:
    """Validate the optional ``asr`` voice-input section of a game manifest.

    ``phrases`` maps intent names to trigger words.  Intents must be declared
    in the state machine's ``on_intent`` tables so a typo cannot silently
    disable voice control; the built-in ``speech_done`` intent is not
    remappable because it belongs to the speech acknowledgement protocol.
    """
    if not isinstance(asr, dict):
        raise ValueError("asr must be an object")
    enabled = asr.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("asr.enabled must be boolean")
    phrases = asr.get("phrases")
    if not isinstance(phrases, dict) or not phrases:
        raise ValueError("asr.phrases must map intents to non-empty trigger-word lists")
    known_intents: set[str] = set()
    for state in (machine.get("states") or {}).values():
        if isinstance(state, dict):
            known_intents.update((state.get("on_intent") or {}).keys())
    for intent, words in phrases.items():
        if intent not in known_intents:
            raise ValueError(
                f"asr.phrases key {intent!r} is not an intent of this game's state machine"
            )
        if (
            not isinstance(words, list)
            or not words
            or not all(isinstance(word, str) and word.strip() for word in words)
        ):
            raise ValueError(f"asr.phrases.{intent} must be a non-empty list of trigger words")
        if len(words) != len(set(words)):
            raise ValueError(f"asr.phrases.{intent} must not contain duplicate trigger words")
    return {"enabled": enabled, "phrases": phrases}


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

    def public_all(self) -> list[dict[str, Any]]:
        """Return browser-safe manifests without model, prompt, or device data."""
        return [public_game_manifest(manifest) for manifest in self._games.values()]


def _public_video(video: Any) -> dict[str, Any]:
    if not isinstance(video, dict):
        return {"enabled": False, "path": ""}
    result = {"enabled": bool(video.get("enabled", True)), "path": video.get("path", "")}
    if not isinstance(result["path"], str):
        result["path"] = ""
    return result


def public_game_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Project a game manifest to fields safe for an untrusted browser client."""
    public: dict[str, Any] = {
        key: manifest[key]
        for key in (
            "id", "name", "icon", "description", "enabled", "participants",
            "voice", "speed", "providers",
        )
        if key in manifest
    }
    profile = manifest.get("vision_profile")
    if isinstance(profile, dict):
        safe_profile: dict[str, Any] = {"game_id": profile.get("game_id", manifest.get("id"))}
        safe_profile["video"] = _public_video(profile.get("video"))
        multi = profile.get("multi_view")
        if isinstance(multi, dict):
            safe_multi: dict[str, Any] = {
                "enabled": bool(multi.get("enabled", False)),
                "min_views": int(multi.get("min_views", 1)),
                "views": [],
            }
            for view in multi.get("views", []):
                if not isinstance(view, dict) or not isinstance(view.get("id"), str):
                    continue
                safe_multi["views"].append({
                    "id": view["id"],
                    "video": _public_video(view["video"]),
                })
            safe_profile["multi_view"] = safe_multi
        public["vision_profile"] = safe_profile
    asr = manifest.get("asr")
    if isinstance(asr, dict):
        # Trigger words stay server-side; the browser only learns whether the
        # voice input channel exists for this game.
        public["asr"] = {"enabled": bool(asr.get("enabled", True))}
    return public


def load_games(root: Path | None = None) -> GameRegistry:
    registry = GameRegistry()
    games_root = root if root is not None else GAMES_ROOT
    if not games_root.is_dir():
        return registry
    for manifest_path in sorted(games_root.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            game_id = manifest.get("id")
            if not isinstance(game_id, str) or not game_id.isidentifier():
                raise ValueError("invalid id")
            if not isinstance(manifest.get("name"), str):
                raise ValueError("missing or invalid name")
            if not isinstance(manifest.get("enabled"), bool):
                raise ValueError("missing or invalid enabled")
            manifest["participants"] = normalize_participants(manifest.get("participants"))
            if "texts" in manifest:
                # The per-key speech table was replaced by inline speech
                # actions on state-machine states; keeping both would let a
                # manifest silently drift between two schemas.
                raise ValueError("legacy 'texts' section removed; inline speech in state_machine")
            machine = manifest.get("state_machine")
            if not isinstance(machine, dict):
                raise ValueError("state_machine must be an object")
            manifest["state_machine"] = validate_state_machine(machine, game_id)
            if "asr" in manifest:
                manifest["asr"] = validate_asr_section(manifest["asr"], manifest["state_machine"])

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
            if "tts_local" not in providers and isinstance(manifest.get("tts_provider"), str):
                providers["tts_local"] = manifest["tts_provider"]
            manifest["providers"] = providers
            profile = manifest.get("vision_profile")
            profile_path = manifest_path.parent / "vision_profile.json"
            if profile is not None:
                from components.vision_yolov8_adjudicator.profile import validate_profile

                profile = validate_profile(profile)
            elif profile_path.is_file():
                from components.vision_yolov8_adjudicator.profile import load_profile

                profile = load_profile(profile_path)
            else:
                profile = None
            if profile is not None:
                if profile.get("game_id") != game_id:
                    raise ValueError("vision profile game_id does not match manifest id")
                manifest["vision_profile"] = profile
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
    """Resolve one semantic provider slot from the game manifest.

    Slots describe responsibility (for example ``vision_adjudicator``), not
    implementation technology (for example YOLO). The game manifest is the
    single configuration source; canonical slots are checked first, followed
    by narrowly scoped migration aliases.
    """
    providers = manifest.get("providers", {})
    if isinstance(providers, dict):
        for slot in (provider_slot, *_PROVIDER_SLOT_ALIASES.get(provider_slot, ())):
            configured = providers.get(slot)
            if isinstance(configured, str) and configured.strip():
                return configured.strip()
    return fallback


def resolve_adjudication_timeout(manifest: dict[str, Any], default: float) -> float:
    """Resolve a game's total visual adjudication budget.

    The game owns this deadline because detection time is part of its rules;
    the process-wide value remains an operational fallback/override.
    """
    profile = manifest.get("vision_profile")
    timeouts = profile.get("timeouts") if isinstance(profile, dict) else None
    configured = timeouts.get("adjudication_seconds") if isinstance(timeouts, dict) else None
    if isinstance(configured, (int, float)) and not isinstance(configured, bool) and configured > 0:
        return float(configured)
    return float(default)


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
