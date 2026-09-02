"""Backend-authoritative game state machine engine.

One :class:`GameRound` drives one game session through the state graph
declared in the game manifest's ``state_machine`` section.  The round is the
single source of truth: frontends submit *intents* (button presses, speech
playback acknowledgements) and render the emitted event stream; they never
advance state themselves.

Execution model
---------------
* Entering a state runs its ``on_enter`` action list on a dedicated worker
  thread.  ``speech`` actions emit a directive event; ``await`` speech blocks
  the sequence until the frontend acknowledges with the built-in
  ``speech_done`` intent (or the fallback timeout fires so a dead client
  cannot stall the round).
* ``duration`` starts once the on_enter sequence has been issued (matching
  the stage rhythm where the reveal hold begins when the 停 clip finishes).
  The worker emits ``tick`` events every ``tick_seconds`` and applies
  ``on_expire`` when the budget elapses.
* Every transition bumps a generation counter.  Timers and long actions
  check the generation before acting, so an expiring timer racing an early
  intent resolves to whichever transitioned first.
* ``adjudicate`` actions call an injectable adjudication callable whose
  provider events are relayed verbatim; its returned result routes through
  the state's ``on_event`` table (``adjudication.result`` /
  ``adjudication.diagnosis``) and becomes the template context for later
  speech placeholders.
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable, Mapping

from core.errors import DiceArenaError
from core.games import render_speech_text
from core.state_schema import validate_state_machine

# A frontend that never acknowledges an awaited speech directive must not
# stall the round forever; after this many seconds the sequence continues.
AWAIT_FALLBACK_SECONDS = 30.0

_RELAYED_PROVIDER_EVENTS = {"phase", "progress", "video", "result", "diagnosis"}
_TERMINAL_STATUSES = {"exited", "cancelled", "error"}


class IntentRejectedError(DiceArenaError):
    """The intent is not accepted in the round's current state."""

    def __init__(self, round_id: str, state: str, intent: str) -> None:
        super().__init__(
            f"intent {intent!r} is not accepted in state {state!r}",
            "ROUND_INTENT_REJECTED",
            409,
        )


class RoundClosedError(DiceArenaError):
    """The round already reached a terminal status."""

    def __init__(self, round_id: str) -> None:
        super().__init__("round is no longer running", "ROUND_CLOSED", 409)


AdjudicateFn = Callable[
    [Mapping[str, Any], Callable[[dict[str, Any]], None], Callable[[], bool], Callable[[str], None]],
    dict[str, Any],
]


def now_ms() -> int:
    return int(time.time() * 1000)


class GameRound:
    """One authoritative game session driven by a manifest state machine."""

    def __init__(
        self,
        *,
        game_id: str,
        manifest: Mapping[str, Any],
        adjudicate_fn: AdjudicateFn | None = None,
        await_fallback_seconds: float = AWAIT_FALLBACK_SECONDS,
        round_id: str | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.id = round_id or uuid.uuid4().hex
        self.game_id = game_id
        # Snapshot the manifest at round creation: hot-reloaded manifests
        # only affect rounds created afterwards.
        self.manifest = dict(manifest)
        self.machine = validate_state_machine(
            dict(self.manifest.get("state_machine") or {}), game_id
        )
        self._adjudicate_fn = adjudicate_fn or self._default_adjudicate_fn
        self._await_fallback_seconds = await_fallback_seconds
        self._log = log or (lambda line: print(f"[round:{self.id[:8]}] {line}", flush=True))

        self.status = "running"
        self.state = ""
        self.result: dict[str, Any] | None = None
        self.error = ""
        self.events: list[dict[str, Any]] = []
        self.event_sequence = 0
        self.revision = 0
        self.created_at = now_ms()
        self.finished_at: int | None = None

        self.condition = threading.Condition()
        self._cancelled = False
        # Bumped on every transition; workers capture their generation and
        # abandon work when it no longer matches.
        self._generation = 0
        self._awaiting_directive: str | None = None
        self._worker: threading.Thread | None = None

    # ---- lifecycle -----------------------------------------------------

    def start(self) -> None:
        if self.state:
            raise RuntimeError("round already started")
        self._enter_state(str(self.machine["initial"]))

    def cancel(self) -> None:
        with self.condition:
            if self.status in _TERMINAL_STATUSES:
                return
            self._cancelled = True
            self._finish_locked("cancelled")

    def is_cancelled(self) -> bool:
        with self.condition:
            return self._cancelled

    # ---- intents -------------------------------------------------------

    def submit_intent(self, name: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Apply one frontend intent; returns the round snapshot."""
        payload = payload or {}
        with self.condition:
            if self.status in _TERMINAL_STATUSES:
                raise RoundClosedError(self.id)
            if name == "speech_done":
                self._ack_speech_locked(str(payload.get("directive_id") or ""))
                return self._snapshot_locked()
            state = self._state_config_locked()
            transition = (state.get("on_intent") or {}).get(name)
            if transition is None:
                raise IntentRejectedError(self.id, self.state, name)
            # The transition only applies if no other transition wins the
            # race between reading it here and applying it below.
            expected_generation = self._generation
        self._apply_transition(
            transition,
            trigger=f"intent:{name}",
            expected_generation=expected_generation,
        )
        return self.snapshot()

    # ---- snapshots / SSE support ----------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self.condition:
            return self._snapshot_locked()

    def wait_for_update(self, after_revision: int, timeout: float) -> dict[str, Any]:
        with self.condition:
            self.condition.wait_for(
                lambda: self.revision > after_revision or self.status in _TERMINAL_STATUSES,
                timeout=max(0.0, timeout),
            )
            return self._snapshot_locked()

    def find_directive(self, directive_id: str) -> dict[str, Any] | None:
        """Return one emitted speech directive by id (for the frame endpoint).

        The events list is a bounded window, so very old directives may no
        longer resolve; repeated reads of a live directive are idempotent.
        """
        with self.condition:
            for event in self.events:
                if event.get("event") == "speech" and event.get("directive_id") == directive_id:
                    return dict(event)
        return None

    def _snapshot_locked(self) -> dict[str, Any]:
        return {
            "round_id": self.id,
            "game_id": self.game_id,
            "status": self.status,
            "state": self.state,
            "error": self.error,
            "result": self.result,
            "events": list(self.events),
            "event_sequence": self.event_sequence,
            "revision": self.revision,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }

    # ---- internal helpers (all called with self.condition held) --------

    def _state_config_locked(self) -> dict[str, Any]:
        return self.machine["states"][self.state]

    def _emit_locked(self, event: dict[str, Any]) -> None:
        event = dict(event)
        event.setdefault("timestamp_ms", now_ms())
        self.event_sequence += 1
        event.setdefault("sequence", self.event_sequence)
        self.events.append(event)
        self.events = self.events[-60:]
        self.revision += 1
        self.condition.notify_all()

    def _emit(self, event: dict[str, Any]) -> None:
        with self.condition:
            self._emit_locked(event)

    def _finish_locked(self, status: str, error: str = "") -> None:
        if self.status in _TERMINAL_STATUSES:
            return
        self.status = status
        self.error = error
        self.finished_at = now_ms()
        self._generation += 1
        self._awaiting_directive = None
        self._emit_locked({"event": "round_complete", "status": status, "state": self.state})

    def _ack_speech_locked(self, directive_id: str) -> None:
        """Wake the worker waiting on an awaited speech directive."""
        if self._awaiting_directive == directive_id:
            self._awaiting_directive = None
            self.condition.notify_all()
        # A late or duplicate acknowledgement for an already-released
        # directive is idempotent and ignored.

    # ---- transitions ----------------------------------------------------

    def _enter_state(self, name: str) -> None:
        with self.condition:
            if self.status in _TERMINAL_STATUSES:
                return
            self.state = name
            self._generation += 1
            generation = self._generation
            state = self.machine["states"][name]
            self._emit_locked({
                "event": "state_changed",
                "state": name,
                "ui": dict(state.get("ui") or {}),
            })
        worker = threading.Thread(
            target=self._run_state,
            args=(generation, name),
            name=f"round-{self.id[:8]}-{name}",
            daemon=True,
        )
        with self.condition:
            if self.status in _TERMINAL_STATUSES or self._generation != generation:
                return
            self._worker = worker
        worker.start()

    def _apply_transition(
        self,
        transition: Mapping[str, Any] | None,
        *,
        trigger: str,
        expected_generation: int | None = None,
    ) -> None:
        if not isinstance(transition, dict):
            return
        if expected_generation is not None:
            with self.condition:
                superseded = (
                    self.status in _TERMINAL_STATUSES
                    or self._generation != expected_generation
                )
            if superseded:
                # Another transition (or cancellation) won the race; this
                # one is silently dropped and the snapshot stays coherent.
                return
        if transition.get("exit") is True:
            with self.condition:
                self._finish_locked("exited")
            return
        if "to" in transition:
            self._enter_state(str(transition["to"]))
            return
        actions = transition.get("actions") or []
        # Intent-attached actions run on the caller's thread.  They are
        # short replay announcements; awaiting here would block the HTTP
        # request, so await semantics are honoured only in on_enter.
        for action in actions:
            if action.get("action") == "speech":
                self._emit_speech(action)
        _ = trigger

    # ---- state worker ---------------------------------------------------

    def _run_state(self, generation: int, name: str) -> None:
        try:
            state = self.machine["states"][name]
            for action in state.get("on_enter") or []:
                if not self._worker_alive(generation):
                    return
                if action.get("action") == "speech":
                    directive = self._emit_speech(action)
                    if action.get("await"):
                        if not self._wait_speech_done(generation, directive["directive_id"]):
                            return
                elif action.get("action") == "adjudicate":
                    if not self._run_adjudication(generation):
                        return
            duration = state.get("duration")
            if isinstance(duration, (int, float)) and duration > 0:
                if not self._run_timer(generation, name, float(duration), state):
                    return
                if not self._worker_alive(generation):
                    return
                self._apply_transition(
                    state.get("on_expire"),
                    trigger="expire",
                    expected_generation=generation,
                )
        except Exception as exc:  # a worker crash must end the round, not hang it
            with self.condition:
                self._finish_locked("error", error=str(exc))
            self._log(f"state {name} worker failed: {exc}")

    def _worker_alive(self, generation: int) -> bool:
        with self.condition:
            return (
                self.status not in _TERMINAL_STATUSES
                and self._generation == generation
            )

    def _wait_speech_done(self, generation: int, directive_id: str) -> bool:
        """Block until the frontend acknowledges or the fallback fires."""
        with self.condition:
            self._awaiting_directive = directive_id
            self.condition.notify_all()
            deadline = time.monotonic() + self._await_fallback_seconds
            while True:
                if (
                    self.status in _TERMINAL_STATUSES
                    or self._generation != generation
                    or self._awaiting_directive is None
                ):
                    self._awaiting_directive = None
                    return self._generation == generation and self.status not in _TERMINAL_STATUSES
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # Fallback: continue the sequence without the client.
                    self._awaiting_directive = None
                    self._emit_locked({
                        "event": "speech_timeout",
                        "directive_id": directive_id,
                    })
                    return self._generation == generation
                self.condition.wait(timeout=min(remaining, 0.5))

    def _run_timer(
        self, generation: int, name: str, duration: float, state: Mapping[str, Any]
    ) -> bool:
        tick = float(state.get("tick_seconds") or 1.0)
        end = time.monotonic() + duration
        while True:
            with self.condition:
                if (
                    self.status in _TERMINAL_STATUSES
                    or self._generation != generation
                ):
                    return False
                remaining = end - time.monotonic()
                if remaining <= 0:
                    return True
                self._emit_locked({
                    "event": "tick",
                    "state": name,
                    "remaining_ms": int(remaining * 1000),
                })
            time.sleep(min(tick, max(remaining, 0.01)))

    # ---- actions --------------------------------------------------------

    def _emit_speech(self, action: Mapping[str, Any]) -> dict[str, Any]:
        """Resolve one speech action against the current context and emit it."""
        with self.condition:
            context = {
                key: value
                for key, value in (self.result or {}).items()
                # Template placeholders are scalars only; lists like
                # player_values must not break rendering.
                if isinstance(value, (str, int, float)) and not isinstance(value, bool)
            }
            voice_default = self.manifest.get("voice", "default")
            speed_default = self.manifest.get("speed", 1.0)
        if "select_by" in action:
            outcome = str(context.get("winner_role") or "")
            case = (action.get("cases") or {}).get(outcome)
            if not isinstance(case, dict):
                raise RuntimeError(
                    f"speech select_by=winner_role has no case for {outcome!r}"
                )
            entry = dict(case)
        else:
            entry = dict(action)
        text = render_speech_text(str(entry.get("text") or ""), context)
        directive: dict[str, Any] = {
            "event": "speech",
            "directive_id": uuid.uuid4().hex[:12],
            "mode": entry.get("mode", "tts_local"),
            "text": text,
            "await": bool(action.get("await")),
        }
        if entry.get("mode", "tts_local") == "audio":
            directive["audio"] = entry.get("audio", "")
        if entry.get("provider"):
            directive["provider"] = entry["provider"]
        directive["voice"] = entry.get("voice") or voice_default
        directive["speed"] = entry.get("speed") or speed_default
        self._emit(directive)
        return directive

    def _run_adjudication(self, generation: int) -> bool:
        def on_event(event: dict[str, Any]) -> None:
            # Provider lifecycle events are relayed verbatim so the browser
            # keeps its existing progress/video/diagnosis rendering.
            if isinstance(event, dict) and event.get("event") in _RELAYED_PROVIDER_EVENTS:
                self._emit(event)

        def is_cancelled() -> bool:
            with self.condition:
                return (
                    self._cancelled
                    or self.status in _TERMINAL_STATUSES
                    or self._generation != generation
                )

        try:
            outcome = self._adjudicate_fn(self.manifest, on_event, is_cancelled, self._log)
        except Exception as exc:
            with self.condition:
                self._finish_locked("error", error=str(exc))
            self._log(f"adjudication failed: {exc}")
            return False
        if not isinstance(outcome, dict) or not self._worker_alive(generation):
            return False
        with self.condition:
            self.result = outcome
            state = self._state_config_locked()
        if outcome.get("diagnosed"):
            route = "adjudication.diagnosis"
        else:
            route = "adjudication.result"
        transition = (state.get("on_event") or {}).get(route)
        if not isinstance(transition, dict):
            with self.condition:
                self._finish_locked(
                    "error",
                    error=f"state {self.state!r} has no on_event route for {route}",
                )
            return False
        self._apply_transition(transition, trigger=route, expected_generation=generation)
        return True

    @staticmethod
    def _default_adjudicate_fn(
        manifest: Mapping[str, Any],
        on_event: Callable[[dict[str, Any]], None],
        is_cancelled: Callable[[], bool],
        on_log: Callable[[str], None],
    ) -> dict[str, Any]:
        raise RuntimeError("no adjudicate_fn configured for this round")
