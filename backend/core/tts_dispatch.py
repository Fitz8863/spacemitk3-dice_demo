"""Provider selection and invocation for the TTS application seam."""
from __future__ import annotations

from typing import Any, Callable

from core.errors import DiceArenaError
from core.games import GameRegistry, require_game, resolve_provider_id


class TtsDispatcher:
    """Keep HTTP handlers independent from concrete TTS implementations."""

    def __init__(
        self,
        components: Any,
        games: GameRegistry,
        *,
        default_provider: str = "tts_qwen3",
    ) -> None:
        self.components = components
        self.games = games
        self.default_provider = default_provider

    @staticmethod
    def game_id(payload: dict[str, Any] | None, fallback: str = "dice") -> str:
        value = (payload or {}).get("game") or fallback
        return str(value)

    def provider_id(self, game_id: str = "dice") -> str:
        manifest = require_game(self.games, game_id)
        return resolve_provider_id(manifest, "tts", self.default_provider)

    def provider(self, payload: dict[str, Any] | None = None, game_id: str | None = None):
        selected_game = game_id or self.game_id(payload)
        provider_id = self.provider_id(selected_game)
        return self.components.require(provider_id, expected_type="tts")

    def synthesize(
        self,
        payload: dict[str, Any],
        *,
        game_id: str | None = None,
    ) -> tuple[bytes, dict[str, str]]:
        provider = self.provider(payload, game_id)
        synthesize = getattr(provider, "synthesize", None)
        if not callable(synthesize):
            raise DiceArenaError(
                f"TTS provider {provider.id} does not implement synthesize()",
                "TTS_PROVIDER_ERROR",
                502,
            )
        audio, headers = synthesize(payload)
        return audio, headers

    def stream(
        self,
        payload: dict[str, Any],
        write_frame: Callable[[bytes], None],
        *,
        game_id: str | None = None,
    ) -> Any:
        provider = self.provider(payload, game_id)
        stream = getattr(provider, "stream", None)
        if not callable(stream):
            raise DiceArenaError(
                f"TTS provider {provider.id} does not implement stream()",
                "TTS_PROVIDER_ERROR",
                502,
            )
        stream(payload, write_frame)
        return provider

    def health(self, game_id: str = "dice") -> dict[str, Any]:
        provider = self.provider(game_id=game_id)
        try:
            health = provider.health()
        except Exception as exc:
            return {
                "id": provider.id,
                "type": provider.type,
                "role": provider.role,
                "ok": False,
                "error": str(exc),
            }
        if not isinstance(health, dict):
            return {
                "id": provider.id,
                "type": provider.type,
                "role": provider.role,
                "ok": False,
                "error": "health() must return an object",
            }
        result = dict(health)
        result.setdefault("id", provider.id)
        result.setdefault("type", provider.type)
        result.setdefault("role", provider.role)
        return result
