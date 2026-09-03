"""Provider interface for streaming speech recognition.

The recognition algorithm (zipformer on the board, or a future cloud model)
does not define the provider interface; the capability does: a streaming ASR
provider owns one microphone listening session and reports finalized
sentences to a callback.  Intent mapping and speech gating are decided by the
round engine, not by the provider.

Session semantics
-----------------
``start_session``/``stop_session`` describe *routing*, not necessarily
processes.  A provider may run its engine resident: the heavy model loads
once (via the optional ``prewarm()`` hook, which the backend calls blocking
at startup and whose failure refuses startup), and each session is an
instant callback swap on that engine — switching between round intent
listening and standby wake-word listening never re-pays the model load.
``stop_session`` then detaches the routing while the engine stays warm, and
the optional ``shutdown()`` hook (invoked by the backend's shutdown seam)
releases the engine and the microphone.  A provider without ``prewarm``
keeps the classic per-session spawn/teardown behaviour; both satisfy this
interface.  ``start_session`` must never block on model loading.
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
        Attaching a session while another is active replaces it.
        """
        raise NotImplementedError

    @abstractmethod
    def stop_session(self, handle: Any) -> None:
        """End the session's routing.  Idempotent; ignores foreign handles.

        On a resident provider this detaches the callback and leaves the
        engine (and microphone) warm; releasing them is ``shutdown()``'s job.
        """
        raise NotImplementedError
