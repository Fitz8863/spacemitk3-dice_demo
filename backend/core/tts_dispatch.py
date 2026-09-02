"""Provider selection and invocation for the TTS application seam."""
from __future__ import annotations

from typing import Any, Callable

from core.errors import DiceArenaError
from core.games import GameRegistry, require_game, resolve_provider_id


# Speech-entry mode -> manifest provider slot.  Exactly three modes exist:
# ``audio`` (never reaches the dispatcher), ``tts_local``, ``tts_remote``.
SPEECH_MODE_SLOTS = {
    "tts_local": "tts_local",
    "tts_remote": "tts_remote",
}


class TtsDispatcher:
    """Keep HTTP handlers independent from concrete TTS implementations."""

    def __init__(
        self,
        components: Any,
        games: GameRegistry,
        *,
        default_provider: str = "tts_qwen3",
        slot_fallbacks: Callable[[str], str] | None = None,
        pinned_local_tts: str | None = None,
    ) -> None:
        self.components = components
        self.games = games
        self.default_provider = default_provider
        # Arena (backend/config.json) slot defaults, filled in only when the
        # game manifest leaves a slot unconfigured.
        self._slot_fallbacks = slot_fallbacks
        # The local TTS engine is chosen once at process start; while set it
        # overrides even later manifest edits (switching requires a restart).
        self._pinned_local_tts = pinned_local_tts

    @staticmethod
    def game_id(payload: dict[str, Any] | None, fallback: str = "dice") -> str:
        value = (payload or {}).get("game") or fallback
        return str(value)

    def _slot_value(self, slot: str, game_id: str) -> str:
        """Resolve one provider slot: pin > game manifest > arena > builtin."""
        if slot == "tts_local" and self._pinned_local_tts:
            return self._pinned_local_tts
        provider_id = resolve_provider_id(require_game(self.games, game_id), slot)
        if not provider_id and self._slot_fallbacks is not None:
            provider_id = self._slot_fallbacks(slot)
        if not provider_id and slot == "tts_local":
            provider_id = self.default_provider
        return provider_id

    def provider_id(self, game_id: str = "dice") -> str:
        return self._slot_value("tts_local", game_id)

    def provider_id_for_speech_entry(self, entry: dict[str, Any], game_id: str) -> str:
        """Resolve the provider id for one manifest speech entry.

        A per-entry ``provider`` field pins an explicit id; otherwise the
        entry's mode selects the slot (``tts_local`` -> local slot,
        ``tts_remote`` -> remote slot).
        """
        explicit = entry.get("provider")
        if explicit:
            return str(explicit)
        mode = str(entry.get("mode", "tts_local"))
        slot = SPEECH_MODE_SLOTS.get(mode)
        if slot is None:
            raise DiceArenaError(
                f"unknown speech mode {mode!r}; use tts_local or tts_remote",
                "TTS_SLOT_NOT_CONFIGURED",
                500,
            )
        provider_id = self._slot_value(slot, game_id)
        if not provider_id:
            raise DiceArenaError(
                f"speech mode {mode!r} needs providers.{slot} "
                f"in game {game_id}'s manifest",
                "TTS_SLOT_NOT_CONFIGURED",
                500,
            )
        return provider_id

    def provider(
        self,
        payload: dict[str, Any] | None = None,
        game_id: str | None = None,
        *,
        provider_id: str | None = None,
    ):
        selected_game = game_id or self.game_id(payload)
        resolved = provider_id or self.provider_id(selected_game)
        return self.components.require(resolved, expected_type="tts")

    def synthesize(
        self,
        payload: dict[str, Any],
        *,
        game_id: str | None = None,
        provider_id: str | None = None,
    ) -> tuple[bytes, dict[str, str]]:
        provider = self.provider(payload, game_id, provider_id=provider_id)
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
        provider_id: str | None = None,
    ) -> Any:
        provider = self.provider(payload, game_id, provider_id=provider_id)
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
