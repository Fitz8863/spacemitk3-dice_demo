"""Shared TTS provider interface.

Concrete adapters only need to implement :meth:`synthesize`.  The default
``stream`` method turns that single WAV result into one Dice Arena frame, so a
new non-streaming model can be added without changing the HTTP bridge or the
browser. Providers with lower-latency segmented generation may override
``stream``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from core.components import Component
from core.errors import TtsValidationError


class TtsProvider(Component, ABC):
    """Small stable interface for all speech-synthesis adapters."""

    type = "tts"
    max_text_chars = 4000

    def validate(self, payload: dict[str, Any]) -> tuple[str, str, float]:
        text = str(payload.get("text", "")).strip()
        if not text:
            raise TtsValidationError("text is required")
        if len(text) > self.max_text_chars:
            raise TtsValidationError(
                f"text is too long; limit is {self.max_text_chars} characters"
            )
        try:
            speed = float(payload.get("speed", 1.0))
        except (TypeError, ValueError) as exc:
            raise TtsValidationError("speed must be a number") from exc
        voice = str(payload.get("voice", "default"))
        return text, voice, max(0.25, min(4.0, speed))

    @abstractmethod
    def synthesize(self, payload: dict[str, Any]) -> tuple[bytes, dict[str, str]]:
        """Generate one WAV and optional response headers."""
        raise NotImplementedError

    def stream(self, payload: dict[str, Any], write_frame: Callable[[bytes], None]) -> None:
        """Default stream adapter for providers that return one complete WAV."""
        audio, _ = self.synthesize(payload)
        write_frame(audio)

    def invoke(self, payload: dict[str, Any]) -> bytes:
        audio, _ = self.synthesize(payload)
        return audio
