"""Voice input channel: map recognized sentences onto round intents.

The bridge owns one ASR listening session for the active round.  Every
finalized sentence is matched against the game manifest's ``asr.phrases``
map; a hit submits the intent exactly like a button press.  Sentences are
dropped while a speech directive may still be playing (the speech gate) so
the round's own announcements cannot trigger themselves — skipping a
playing announcement stays a button-only action.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable

from core.asr import AsrSessionError
from core.errors import DiceArenaError
from core.games import resolve_provider_id

_TERMINAL_STATUSES = {"exited", "cancelled", "error"}
_WATCHER_POLL_SECONDS = 2.0


def normalize_speech_text(text: str) -> str:
    """Lowercase and strip all whitespace so matching tolerates model spacing."""
    return "".join(text.split()).lower()


def match_phrase_intent(phrases: dict[str, list[str]], text: str) -> str | None:
    """Return the first intent whose trigger word occurs in ``text``."""
    normalized = normalize_speech_text(text)
    if not normalized:
        return None
    for intent, words in phrases.items():
        for word in words:
            if normalize_speech_text(word) in normalized:
                return intent
    return None


class AsrIntentBridge:
    """One ASR session bound to the currently active round.

    Sessions are round-scoped: creating a round starts listening (when the
    game manifest enables ASR), the round ending stops it.  Nothing here
    blocks round creation — a broken ASR provider must never take down the
    button-driven flow.
    """

    def __init__(
        self,
        *,
        components: Any,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self._components = components
        self._log = log or (lambda line: print(f"[asr] {line}", flush=True))
        self._lock = threading.Lock()
        self._provider: Any = None
        self._session: Any = None
        self._watcher: threading.Thread | None = None

    def start_for_round(self, round_: Any) -> bool:
        """Start listening for ``round_``; True when a session is active."""
        asr = (round_.manifest or {}).get("asr") or {}
        if not asr.get("enabled", False):
            return False
        provider_id = resolve_provider_id(round_.manifest, "asr")
        if not provider_id:
            self._log("asr.enabled is true but providers.asr is not configured; voice input off")
            return False
        try:
            provider = self._components.require(provider_id, expected_type="asr")
        except DiceArenaError as exc:
            self._log(f"ASR provider {provider_id} unavailable: {exc.message}")
            return False
        self.stop()
        phrases = dict(asr.get("phrases") or {})
        round_id = str(round_.id)[:8]
        try:
            session = provider.start_session(
                lambda text: self._on_sentence(round_, phrases, text),
                on_log=self._log,
            )
        except AsrSessionError as exc:
            self._log(f"ASR session failed to start for round {round_id}: {exc}")
            return False
        except Exception as exc:  # provider bug must not break round creation
            self._log(f"ASR session raised for round {round_id}: {exc!r}")
            return False
        with self._lock:
            self._provider = provider
            self._session = session
            self._watcher = threading.Thread(
                target=self._watch_round,
                args=(round_, session),
                daemon=True,
                name="asr-bridge",
            )
            self._watcher.start()
        self._log(f"listening for round {round_id} (provider {provider_id})")
        return True

    def _on_sentence(self, round_: Any, phrases: dict[str, list[str]], text: str) -> None:
        """Match one finalized sentence and surface the outcome to the UI.

        Every recognition result is relayed as an ``asr`` observation event so
        the browser can acknowledge what was heard (submitted / suppressed by
        the speech gate / rejected by the current state / unmatched).  A
        sentence with no visible feedback would leave the player wondering
        whether voice input works at all.
        """
        intent = match_phrase_intent(phrases, text)
        if intent is None:
            self._log(f"heard {text!r}: no trigger word matched")
            round_.emit_observation({"event": "asr", "status": "unmatched", "text": text})
            return
        if round_.speech_active:
            self._log(f"heard {text!r} matched {intent!r} while speech is playing; ignored")
            round_.emit_observation({
                "event": "asr", "status": "suppressed", "text": text, "matched": intent,
            })
            return
        try:
            round_.submit_intent(intent, {"source": "asr", "text": text})
        except DiceArenaError as exc:
            # Spoken words that the current state does not accept, or a round
            # that just ended, are normal conversation — not an error path.
            self._log(f"heard {text!r} matched {intent!r}: {exc.message}")
            round_.emit_observation({
                "event": "asr", "status": "rejected", "text": text, "matched": intent,
            })
            return
        self._log(f"heard {text!r} -> intent {intent!r} submitted")
        round_.emit_observation({
            "event": "asr", "status": "submitted", "text": text, "matched": intent,
        })

    def _watch_round(self, round_: Any, session: Any) -> None:
        """Stop the session once its round reaches a terminal status."""
        while True:
            time.sleep(_WATCHER_POLL_SECONDS)
            with self._lock:
                if self._session is not session:
                    return
            if round_.status in _TERMINAL_STATUSES:
                self.stop()
                return

    def stop(self) -> None:
        with self._lock:
            session = self._session
            provider = self._provider
            watcher = self._watcher
            self._session = None
            self._provider = None
            self._watcher = None
        if session is not None and provider is not None:
            try:
                provider.stop_session(session)
            except Exception as exc:
                self._log(f"session stop failed: {exc}")
        if watcher is not None and watcher is not threading.current_thread():
            watcher.join(timeout=_WATCHER_POLL_SECONDS + 1.0)
