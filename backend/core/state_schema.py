"""Validation for the game state-machine section of a game manifest.

A state machine is an explicitly named directed graph: ``states`` keys are
nodes, the ``to`` targets under ``on_intent`` / ``on_expire`` / ``on_event``
are edges, and ``initial`` is the entry node.  The JSON object's writing
order carries no execution semantics, and any state may point at any other
(forward, backward, or skipping) — removing a state means deleting its node
and re-pointing the edges that referenced it; the validator turns a missed
re-point into a hard load error instead of silent misbehaviour.

Speech lines are inlined on ``speech`` actions.  Action types are validated
against a registry-style whitelist so future capabilities (for example a
``robot`` command) join by extending the whitelist and adding an executor,
not by loosening the schema.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

# Kept in sync with TtsProvider.max_text_chars; imported lazily-free as a
# plain constant to keep this module free of the component import graph.
MAX_SPEECH_TEXT_CHARS = 4000

ACTION_TYPES = {"speech", "adjudicate"}

# The single context selector supported today: the adjudicated winner maps a
# result-state announcement to PLAYER/AGENT/TIE speech cases.
SELECT_BY_KEYS = {"winner_role"}
SELECT_BY_CASES = {"PLAYER", "AGENT", "TIE"}

_STATE_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")
_INTENT_NAME_RE = re.compile(r"[a-z][a-z0-9_]*")
_EVENT_NAME_RE = re.compile(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*")

SPEECH_MODES = {"tts_local", "tts_remote", "audio"}


class StateMachineError(ValueError):
    """Raised when a game's ``state_machine`` section is invalid."""


def _error(field: str, message: str) -> StateMachineError:
    return StateMachineError(f"{field}: {message}")


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(field, "must be a non-empty string")
    return value.strip()


def _require_number(
    value: Any, field: str, *, low: float, high: float | None = None, low_inclusive: bool = False
) -> float:
    import math

    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise _error(field, "must be a finite number")
    number = float(value)
    if low_inclusive:
        if number < low:
            raise _error(field, f"must be >= {low}")
    elif number <= low:
        raise _error(field, f"must be > {low}")
    if high is not None and number > high:
        raise _error(field, f"must be <= {high}")
    return number


def _validate_audio_path(value: Any, field: str) -> str:
    """Validate a game-relative wav reference without accepting traversal."""
    audio = _require_str(value, field)
    segments = audio.replace("\\", "/").split("/")
    if audio.startswith(("/", "\\")) or ":" in segments[0]:
        raise _error(field, "must be a relative .wav path")
    if any(part in {"", ".", ".."} for part in segments):
        raise _error(field, "must not contain empty, '.', or '..' segments")
    if not audio.lower().endswith(".wav"):
        raise _error(field, "must reference a .wav file")
    return audio


def _validate_speech_content(action: Mapping[str, Any], field: str) -> None:
    """Validate one speech payload (mode/text/audio/provider/voice/speed)."""
    mode = action.get("mode", "tts_local")
    if mode not in SPEECH_MODES:
        raise _error(f"{field}.mode", "must be tts_local, tts_remote, or audio")
    if "provider" in action:
        if mode == "audio":
            raise _error(f"{field}.provider", "audio speech never synthesizes; drop the provider field")
        _require_str(action["provider"], f"{field}.provider")
    if mode == "audio":
        if "audio" not in action:
            raise _error(f"{field}.audio", "audio speech requires a wav reference")
        _validate_audio_path(action["audio"], f"{field}.audio")
        if "text" in action and not isinstance(action["text"], str):
            raise _error(f"{field}.text", "audio speech text must be a string")
        return
    text = action.get("text")
    if not isinstance(text, str) or not text.strip():
        raise _error(f"{field}.text", "tts speech requires non-empty text")
    if len(text) > MAX_SPEECH_TEXT_CHARS:
        raise _error(f"{field}.text", f"must be at most {MAX_SPEECH_TEXT_CHARS} characters")
    if "voice" in action:
        _require_str(action["voice"], f"{field}.voice")
    if "speed" in action:
        _require_number(
            action["speed"], f"{field}.speed", low=0.25, high=4.0, low_inclusive=True
        )


def _validate_action(action: Any, field: str) -> None:
    if not isinstance(action, dict):
        raise _error(field, "must be an object")
    action_type = action.get("action")
    if action_type not in ACTION_TYPES:
        raise _error(
            f"{field}.action", f"must be one of {sorted(ACTION_TYPES)} (got {action_type!r})"
        )
    if action_type == "adjudicate":
        return
    if "select_by" in action:
        if action.get("select_by") not in SELECT_BY_KEYS:
            raise _error(f"{field}.select_by", f"must be one of {sorted(SELECT_BY_KEYS)}")
        if "await" in action:
            raise _error(f"{field}.await", "select_by speech cannot await")
        cases = action.get("cases")
        if not isinstance(cases, dict):
            raise _error(f"{field}.cases", "must be an object")
        missing = SELECT_BY_CASES - set(cases)
        if missing:
            raise _error(f"{field}.cases", f"missing outcomes: {sorted(missing)}")
        for name, case in cases.items():
            if name not in SELECT_BY_CASES:
                raise _error(f"{field}.cases.{name}", f"must be one of {sorted(SELECT_BY_CASES)}")
            _validate_speech_content(case, f"{field}.cases.{name}")
        return
    _validate_speech_content(action, field)
    if "await" in action and not isinstance(action["await"], bool):
        raise _error(f"{field}.await", "must be a boolean")


def _validate_action_list(actions: Any, field: str) -> None:
    if not isinstance(actions, list) or not actions:
        raise _error(field, "must be a non-empty action array")
    for index, action in enumerate(actions):
        _validate_action(action, f"{field}[{index}]")


def _validate_transition(transition: Any, field: str) -> dict[str, Any]:
    """Validate one trigger outcome: switch state, run actions, or exit."""
    if not isinstance(transition, dict):
        raise _error(field, "must be an object")
    keys = {"to", "exit", "actions"} & set(transition)
    if len(keys) != 1:
        raise _error(field, "must contain exactly one of 'to', 'exit', or 'actions'")
    if "to" in transition:
        _require_str(transition["to"], f"{field}.to")
    elif "exit" in transition:
        if transition["exit"] is not True:
            raise _error(f"{field}.exit", "must be true")
    else:
        _validate_action_list(transition["actions"], f"{field}.actions")
    return transition


def _validate_state(name: str, state: Any, field: str) -> None:
    if not isinstance(state, dict):
        raise _error(field, "must be an object")
    if "ui" in state:
        ui = state["ui"]
        if not isinstance(ui, dict):
            raise _error(f"{field}.ui", "must be an object")
        for key, value in ui.items():
            if key not in {"title", "copy", "view"}:
                raise _error(f"{field}.ui.{key}", "must be 'title', 'copy', or 'view'")
            if not isinstance(value, str):
                raise _error(f"{field}.ui.{key}", "must be a string")
        if "view" in ui and not re.fullmatch(r"[a-z][a-z0-9_]*", ui["view"]):
            raise _error(f"{field}.ui.view", "must be a lowercase view identifier")
    has_duration = False
    if "duration" in state:
        _require_number(state["duration"], f"{field}.duration", low=0, high=300)
        has_duration = True
    if "tick_seconds" in state:
        _require_number(state["tick_seconds"], f"{field}.tick_seconds", low=0, high=60)
    if "on_enter" in state:
        _validate_action_list(state["on_enter"], f"{field}.on_enter")
    if "on_expire" in state:
        if not has_duration:
            raise _error(f"{field}.on_expire", "requires a 'duration' on the same state")
        _validate_transition(state["on_expire"], f"{field}.on_expire")
    if "on_intent" in state:
        intents = state["on_intent"]
        if not isinstance(intents, dict) or not intents:
            raise _error(f"{field}.on_intent", "must be a non-empty object")
        for intent_name, transition in intents.items():
            if not isinstance(intent_name, str) or not _INTENT_NAME_RE.fullmatch(intent_name):
                raise _error(f"{field}.on_intent", f"invalid intent name {intent_name!r}")
            _validate_transition(transition, f"{field}.on_intent.{intent_name}")
    if "on_event" in state:
        events = state["on_event"]
        if not isinstance(events, dict) or not events:
            raise _error(f"{field}.on_event", "must be a non-empty object")
        for event_name, transition in events.items():
            if not isinstance(event_name, str) or not _EVENT_NAME_RE.fullmatch(event_name):
                raise _error(f"{field}.on_event", f"invalid event name {event_name!r}")
            _validate_transition(transition, f"{field}.on_event.{event_name}")


def _collect_transition_targets(machine: Mapping[str, Any]):
    """Yield every (field, target) edge for reference checking."""
    for name, state in machine["states"].items():
        if not isinstance(state, dict):
            continue
        for trigger in ("on_expire",):
            if isinstance(state.get(trigger), dict) and "to" in state[trigger]:
                yield f"states.{name}.{trigger}.to", state[trigger]["to"]
        for trigger in ("on_intent", "on_event"):
            table = state.get(trigger)
            if isinstance(table, dict):
                for key, transition in table.items():
                    if isinstance(transition, dict) and "to" in transition:
                        yield f"states.{name}.{trigger}.{key}.to", transition["to"]


def _reachable_states(machine: Mapping[str, Any]) -> set[str]:
    """BFS over the transition graph starting at ``initial``."""
    seen = {machine["initial"]}
    frontier = [machine["initial"]]
    while frontier:
        name = frontier.pop()
        state = machine["states"].get(name)
        if not isinstance(state, dict):
            continue
        targets: list[str] = []
        for trigger in ("on_expire",):
            transition = state.get(trigger)
            if isinstance(transition, dict) and isinstance(transition.get("to"), str):
                targets.append(transition["to"])
        for trigger in ("on_intent", "on_event"):
            table = state.get(trigger)
            if isinstance(table, dict):
                for transition in table.values():
                    if isinstance(transition, dict) and isinstance(transition.get("to"), str):
                        targets.append(transition["to"])
        for target in targets:
            if target not in seen:
                seen.add(target)
                frontier.append(target)
    return seen


def validate_state_machine(machine: Any, game_id: str) -> dict[str, Any]:
    """Validate and normalize one game's ``state_machine`` section in place."""
    if not isinstance(machine, dict):
        raise StateMachineError("state_machine must be an object")
    if machine.get("schema_version") != 1:
        raise StateMachineError("state_machine.schema_version must be 1")
    states = machine.get("states")
    if not isinstance(states, dict) or not states:
        raise StateMachineError("state_machine.states must be a non-empty object")
    for name, state in states.items():
        if not isinstance(name, str) or not _STATE_NAME_RE.fullmatch(name):
            raise StateMachineError(f"state name {name!r} must match [A-Za-z0-9_-]+")
        _validate_state(name, state, f"state_machine.states.{name}")
    initial = machine.get("initial")
    if not isinstance(initial, str) or initial not in states:
        raise StateMachineError("state_machine.initial must name a declared state")
    for field, target in _collect_transition_targets(machine):
        if target not in states:
            raise _error(field, f"references unknown state {target!r}")
    reachable = _reachable_states(machine)
    unreachable = sorted(set(states) - reachable)
    if unreachable:
        # A state no path reaches is usually leftover code after deleting an
        # upstream node; warn loudly but keep loading so a deliberate reserve
        # state stays possible.
        print(
            f"[state-machine] game {game_id!r} has unreachable states: {unreachable}",
            flush=True,
        )
    return machine


def iter_speech_actions(machine: Mapping[str, Any]):
    """Yield every speech action object declared anywhere in the machine.

    Used both by provider-reference collection (componentctl) and by tests;
    covers ``on_enter`` lists, ``on_intent``/``on_expire`` action lists, and
    ``select_by`` case payloads.
    """
    for state in machine.get("states", {}).values():
        if not isinstance(state, dict):
            continue
        candidates: list[Any] = list(state.get("on_enter") or [])
        for trigger in ("on_intent", "on_expire"):
            table = state.get(trigger)
            if isinstance(table, dict):
                for transition in table.values():
                    if isinstance(transition, dict) and isinstance(transition.get("actions"), list):
                        candidates.extend(transition["actions"])
        for action in candidates:
            if not isinstance(action, dict) or action.get("action") != "speech":
                continue
            yield action
            cases = action.get("cases")
            if isinstance(cases, dict):
                for case in cases.values():
                    payload = dict(case)
                    payload.setdefault("action", "speech")
                    yield payload
