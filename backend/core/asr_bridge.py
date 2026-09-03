"""Voice input channel: map recognized sentences onto round intents.

The bridge owns the microphone *routing* on the ASR engine: every finalized
sentence flows to whichever callback is currently attached — the active
round's intent matcher, or the standby wake-word matcher.  With a resident
engine (see ``asr_zipformer``) attaching and detaching are instant callback
swaps: the model load is paid once at process startup, and switching between
round and standby listening never re-spawns the engine.  Sentences are
dropped while a speech directive may still be playing (the speech gate) so
the round's own announcements cannot trigger themselves — skipping a playing
announcement stays a button-only action.
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
    intents = match_phrase_intents(phrases, text)
    return intents[0] if intents else None


def match_phrase_intents(phrases: dict[str, list[str]], text: str) -> list[str]:
    """Return every intent whose trigger word occurs in ``text``, in order.

    One word may legitimately back several intents (``确定`` both confirms
    the rules and starts the shake); which one applies is decided by the
    round's current state, so the caller tries the candidates in order and
    keeps the first the state machine accepts.
    """
    normalized = normalize_speech_text(text)
    if not normalized:
        return []
    intents: list[str] = []
    for intent, words in phrases.items():
        for word in words:
            if normalize_speech_text(word) in normalized:
                if intent not in intents:
                    intents.append(intent)
                break
    return intents


class AsrIntentBridge:
    """Routes the ASR engine's sentences to the active consumer.

    Attachments are exclusive and instant: starting a round (when the game
    manifest enables ASR) swaps the routing to the round's intent matcher,
    and the round ending detaches it — the resident engine itself stays
    warm for the next consumer.  Nothing here blocks round creation — a
    broken ASR provider must never take down the button-driven flow.
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
        candidates = match_phrase_intents(phrases, text)
        if not candidates:
            self._log(f"heard {text!r}: no trigger word matched")
            round_.emit_observation({"event": "asr", "status": "unmatched", "text": text})
            return
        if round_.speech_active:
            self._log(
                f"heard {text!r} matched {candidates!r} while speech is playing; ignored"
            )
            round_.emit_observation({
                "event": "asr", "status": "suppressed", "text": text,
                "matched": candidates[0],
            })
            return
        # One word can back several intents (确定 confirms the rules and
        # starts the shake); try the candidates in order and keep the first
        # the current state accepts, so the same word follows the game.
        rejected: list[str] = []
        for intent in candidates:
            try:
                round_.submit_intent(intent, {"source": "asr", "text": text})
            except DiceArenaError as exc:
                # Spoken words that the current state does not accept, or a
                # round that just ended, are normal conversation.
                self._log(f"heard {text!r} matched {intent!r}: {exc.message}")
                rejected.append(intent)
                continue
            self._log(f"heard {text!r} -> intent {intent!r} submitted")
            round_.emit_observation({
                "event": "asr", "status": "submitted", "text": text, "matched": intent,
            })
            return
        round_.emit_observation({
            "event": "asr", "status": "rejected", "text": text,
            "matched": candidates[0],
        })

    def _watch_round(self, round_: Any, session: Any) -> None:
        """Detach the round's routing once its round reaches a terminal
        status (the resident engine stays warm for the next consumer)."""
        while True:
            time.sleep(_WATCHER_POLL_SECONDS)
            with self._lock:
                if self._session is not session:
                    return
            if round_.status in _TERMINAL_STATUSES:
                self.stop()
                return

    # ---- 选路会话（无回合屏幕：待机页与游戏列表共用）----

    def start_select_session(
        self,
        *,
        phrases: dict[str, list[str]],
        asr_enabled: bool,
        provider_id: str,
        on_select: Callable[[str, str], None],
        on_heard: Callable[[str], None] | None = None,
    ) -> bool:
        """Route sentences to whichever key's trigger words they contain.

        Used by the screens outside a round: the caller builds the phrase
        table (``{key: [trigger words]}`` — game ids and/or the ``"wake"``
        key) and owns the dispatch.  A hit calls ``on_select(key, text)``
        with the *first* matching key in table order, so the caller controls
        priority — the standby screen declares game keys before ``"wake"``
        so "我想玩摇骰子游戏" selects the dice game even though the wake
        word "游戏" also occurs in it.  Mutually exclusive with round
        sessions and with previous select sessions (last start wins).
        """
        table = {
            str(key): [str(word) for word in words if str(word).strip()]
            for key, words in (phrases or {}).items()
            if words
        }
        if not asr_enabled or not table:
            return False
        try:
            provider = self._components.require(provider_id, expected_type="asr")
        except DiceArenaError as exc:
            self._log(f"ASR provider {provider_id} unavailable for select: {exc.message}")
            return False
        self.stop()
        try:
            session = provider.start_session(
                lambda text: self._on_select_sentence(table, on_select, on_heard, text),
                on_log=self._log,
            )
        except AsrSessionError as exc:
            self._log(f"select ASR session failed to start: {exc}")
            return False
        except Exception as exc:  # provider bug must not break the caller
            self._log(f"select ASR session raised: {exc!r}")
            return False
        with self._lock:
            self._provider = provider
            self._session = session
        self._log(f"select listening for {sorted(table)}")
        return True

    def _on_select_sentence(
        self,
        phrases: dict[str, list[str]],
        on_select: Callable[[str, str], None],
        on_heard: Callable[[str], None] | None,
        text: str,
    ) -> None:
        key = match_phrase_intent(phrases, text)
        if key:
            self._log(f"select heard {text!r} -> {key}")
            try:
                on_select(key, text)
            except Exception as exc:
                self._log(f"select callback error: {exc}")
            return
        self._log(f"select heard {text!r}: no match")
        if on_heard is not None:
            try:
                on_heard(text)
            except Exception:
                pass

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
