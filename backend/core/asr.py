"""Provider interface for streaming speech recognition.

The recognition algorithm (zipformer on the board, or a future cloud model)
does not define the provider interface; the capability does: a streaming ASR
provider owns one microphone listening session and reports finalized
sentences to a callback.  Intent mapping and speech gating are decided by the
round engine, not by the provider.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable

from core.components import Component


class AsrSessionError(RuntimeError):
    """Raised when a listening session is requested in an invalid state."""


class AsrProvider(Component):
    """Streaming ASR adapter: owns mic capture, emits finalized sentences."""

    type = "asr"

    @abstractmethod
    def start_session(
        self,
        on_sentence: Callable[[str], None],
        *,
        on_log: Callable[[str], None] | None = None,
    ) -> Any:
        """Start one listening session and return an opaque session handle.

        ``on_sentence`` is invoked from a reader thread with each finalized
        (VAD-segmented) recognition text and must not block.  ``on_log``
        receives diagnostics (process lifecycle, stderr) as plain lines.
        """
        raise NotImplementedError

    @abstractmethod
    def stop_session(self, handle: Any) -> None:
        """Stop the session and release the microphone.  Idempotent."""
        raise NotImplementedError
