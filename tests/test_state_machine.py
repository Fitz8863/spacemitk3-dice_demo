"""Tests for the state-machine schema validator and round engine."""
from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.errors import DiceArenaError  # noqa: E402
from core.state_machine import (  # noqa: E402
    GameRound,
    IntentRejectedError,
    RoundClosedError,
)
from core.state_schema import (  # noqa: E402
    StateMachineError,
    iter_speech_actions,
    validate_state_machine,
)


def machine(**overrides):
    """A small but complete state graph covering the dice flow shapes."""
    states = {
        "rules": {
            "on_enter": [{"action": "speech", "mode": "tts_local", "text": "规则"}],
            "on_intent": {
                "confirm": {"to": "ready"},
                "repeat": {"actions": [
                    {"action": "speech", "mode": "tts_local", "text": "规则"},
                ]},
                "back": {"exit": True},
            },
        },
        "ready": {
            "on_enter": [{"action": "speech", "mode": "tts_local", "text": "准备"}],
            "on_intent": {
                "start_shake": {"to": "shake_countdown"},
                "back": {"exit": True},
            },
        },
        "shake_countdown": {
            "duration": 3,
            "tick_seconds": 0.9,
            "on_enter": [
                {"action": "speech", "mode": "audio", "audio": "audio/warm.wav"},
            ],
            "on_expire": {"to": "shaking"},
        },
        "shaking": {
            "duration": 10,
            "on_intent": {"stop_shake": {"to": "open_reveal"}},
            "on_expire": {"to": "open_reveal"},
        },
        "open_reveal": {
            "on_enter": [
                {"action": "speech", "mode": "audio", "audio": "audio/stop.wav", "await": True},
                {"action": "speech", "mode": "tts_local", "text": "开盖"},
            ],
            "duration": 4,
            "on_expire": {"to": "analysis"},
        },
        "analysis": {
            "on_enter": [
                {"action": "speech", "mode": "tts_local", "text": "识别中"},
                {"action": "adjudicate"},
            ],
            "on_event": {
                "adjudication.result": {"to": "result"},
                "adjudication.diagnosis": {"to": "analysis_failed"},
            },
        },
        "analysis_failed": {
            "on_enter": [
                {"action": "speech", "mode": "tts_local", "text": "请重试"},
            ],
            "on_intent": {
                "retry": {"to": "analysis"},
                "new_round": {"to": "ready"},
                "back": {"exit": True},
            },
        },
        "result": {
            "on_enter": [{
                "action": "speech",
                "select_by": "winner_role",
                "cases": {
                    "PLAYER": {"mode": "tts_local", "text": "玩家赢 {player_score}"},
                    "AGENT": {"mode": "tts_local", "text": "Agent 赢 {agent_score}"},
                    "TIE": {"mode": "tts_local", "text": "平局"},
                },
            }],
            "on_intent": {"new_round": {"to": "ready"}, "back": {"exit": True}},
        },
    }
    payload = {"schema_version": 1, "initial": "rules", "states": states}
    payload.update(overrides)
    return payload


def manifest_with(machine_payload=None, **extra):
    return {
        "id": "dice",
        "name": "Dice",
        "enabled": True,
        "voice": "default",
        "speed": 1.0,
        "state_machine": machine_payload if machine_payload is not None else machine(),
        **extra,
    }


def wait_for(predicate, timeout=20.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def reached(round_, state_name):
    """True once the round has passed through (or sits in) a state.

    Intermediate states can slide between two polling ticks, so tests must
    assert on the state-change event instead of the current state.
    """
    if round_.snapshot()["state"] == state_name:
        return True
    return any(
        e.get("event") == "state_changed" and e.get("state") == state_name
        for e in round_.snapshot()["events"]
    )


class SchemaValidationTests(unittest.TestCase):
    def test_accepts_full_graph_and_warns_on_unreachable_state(self):
        payload = machine()
        payload["states"]["reserve"] = {
            "on_enter": [{"action": "speech", "mode": "tts_local", "text": "备用"}],
        }
        validate_state_machine(payload, "dice")  # must not raise

    def test_dangling_transition_reference_is_a_hard_error(self):
        payload = machine()
        payload["states"]["rules"]["on_intent"]["confirm"] = {"to": "missing_state"}
        with self.assertRaises(StateMachineError):
            validate_state_machine(payload, "dice")

    def test_initial_must_name_a_declared_state(self):
        with self.assertRaises(StateMachineError):
            validate_state_machine(machine(initial="nowhere"), "dice")

    def test_on_expire_requires_duration(self):
        payload = machine()
        payload["states"]["rules"]["on_expire"] = {"to": "ready"}
        with self.assertRaises(StateMachineError):
            validate_state_machine(payload, "dice")

    def test_unknown_action_type_is_rejected(self):
        payload = machine()
        payload["states"]["rules"]["on_enter"] = [
            {"action": "robot", "command": "shake"},
        ]
        with self.assertRaises(StateMachineError):
            validate_state_machine(payload, "dice")

    def test_select_by_must_cover_all_outcomes_and_cannot_await(self):
        payload = machine()
        del payload["states"]["result"]["on_enter"][0]["cases"]["TIE"]
        with self.assertRaises(StateMachineError):
            validate_state_machine(payload, "dice")
        payload = machine()
        payload["states"]["result"]["on_enter"][0]["await"] = True
        with self.assertRaises(StateMachineError):
            validate_state_machine(payload, "dice")

    def test_speech_mode_and_path_rules_are_enforced(self):
        payload = machine()
        payload["states"]["rules"]["on_enter"] = [
            {"action": "speech", "mode": "tts", "text": "旧写法"},
        ]
        with self.assertRaises(StateMachineError):
            validate_state_machine(payload, "dice")
        payload = machine()
        payload["states"]["rules"]["on_enter"] = [
            {"action": "speech", "mode": "audio", "audio": "../secret.wav"},
        ]
        with self.assertRaises(StateMachineError):
            validate_state_machine(payload, "dice")
        payload = machine()
        payload["states"]["rules"]["on_enter"] = [
            {"action": "speech", "mode": "audio", "audio": "audio/a.wav", "provider": "tts_x"},
        ]
        with self.assertRaises(StateMachineError):
            validate_state_machine(payload, "dice")
        payload = machine()
        payload["states"]["rules"]["on_enter"] = [
            {"action": "speech", "mode": "tts_remote", "text": "远端", "provider": "tts_b"},
        ]
        validate_state_machine(payload, "dice")

    def test_intent_and_event_names_must_be_lowercase_identifiers(self):
        payload = machine()
        payload["states"]["rules"]["on_intent"]["确认"] = {"to": "ready"}
        with self.assertRaises(StateMachineError):
            validate_state_machine(payload, "dice")
        payload = machine()
        payload["states"]["analysis"]["on_event"]["Bad.Event"] = {"to": "result"}
        with self.assertRaises(StateMachineError):
            validate_state_machine(payload, "dice")

    def test_transition_must_pick_exactly_one_outcome(self):
        payload = machine()
        payload["states"]["rules"]["on_intent"]["confirm"] = {
            "to": "ready", "exit": True,
        }
        with self.assertRaises(StateMachineError):
            validate_state_machine(payload, "dice")
        payload = machine()
        payload["states"]["rules"]["on_intent"]["confirm"] = {}
        with self.assertRaises(StateMachineError):
            validate_state_machine(payload, "dice")

    def test_iter_speech_actions_covers_enter_intents_and_cases(self):
        actions = list(iter_speech_actions(machine()))
        texts = {action.get("text") or action.get("provider") for action in actions}
        self.assertIn("规则", texts)
        self.assertIn("开盖", texts)
        self.assertIn("玩家赢 {player_score}", texts)
        providers = [
            action["provider"] for action in actions
            if isinstance(action, dict) and "provider" in action
        ]
        self.assertEqual(providers, [])


class RoundEngineTests(unittest.TestCase):
    def make_round(self, *, machine_payload=None, adjudicate_fn=None, **kwargs):
        manifest = manifest_with(machine_payload)
        round_ = GameRound(
            game_id="dice",
            manifest=manifest,
            adjudicate_fn=adjudicate_fn,
            log=lambda line: None,
            **kwargs,
        )
        round_.start()
        return round_

    def events_of(self, round_, event_name):
        return [e for e in round_.snapshot()["events"] if e.get("event") == event_name]

    def test_start_enters_initial_state_and_emits_speech(self):
        round_ = self.make_round()
        self.assertTrue(wait_for(lambda: round_.snapshot()["state"] == "rules"))
        self.assertTrue(wait_for(lambda: self.events_of(round_, "speech")))
        speeches = self.events_of(round_, "speech")
        self.assertEqual(len(speeches), 1)
        self.assertEqual(speeches[0]["mode"], "tts_local")
        self.assertEqual(speeches[0]["text"], "规则")
        self.assertEqual(speeches[0]["voice"], "default")
        self.assertEqual(speeches[0]["speed"], 1.0)
        self.assertTrue(speeches[0]["directive_id"])

    def test_intent_routes_to_declared_transition(self):
        round_ = self.make_round()
        round_.submit_intent("confirm")
        self.assertTrue(wait_for(lambda: round_.snapshot()["state"] == "ready"))
        self.assertTrue(wait_for(lambda: self.events_of(round_, "speech")))

    def test_unknown_intent_is_rejected_with_conflict_status(self):
        round_ = self.make_round()
        with self.assertRaises(IntentRejectedError):
            round_.submit_intent("stop_shake")
        self.assertEqual(round_.snapshot()["state"], "rules")

    def test_repeat_intent_runs_actions_without_transition(self):
        round_ = self.make_round()
        self.assertTrue(wait_for(lambda: self.events_of(round_, "speech")))
        round_.submit_intent("repeat")
        self.assertTrue(wait_for(lambda: len(self.events_of(round_, "speech")) == 2))
        self.assertEqual(round_.snapshot()["state"], "rules")

    def test_exit_intent_ends_round_and_rejects_further_intents(self):
        round_ = self.make_round()
        round_.submit_intent("back")
        self.assertTrue(wait_for(lambda: round_.snapshot()["status"] == "exited"))
        with self.assertRaises(RoundClosedError):
            round_.submit_intent("confirm")

    def test_duration_expires_into_next_state_with_ticks(self):
        payload = machine()
        payload["states"]["rules"]["on_intent"]["confirm"] = {"to": "timed"}
        payload["states"]["timed"] = {
            "duration": 0.3,
            "tick_seconds": 0.1,
            "on_expire": {"to": "ready"},
        }
        round_ = self.make_round(machine_payload=payload)
        round_.submit_intent("confirm")
        self.assertTrue(wait_for(lambda: round_.snapshot()["state"] == "ready", timeout=3))
        ticks = self.events_of(round_, "tick")
        self.assertGreaterEqual(len(ticks), 1)
        self.assertEqual(ticks[0]["state"], "timed")
        self.assertGreater(ticks[0]["remaining_ms"], 0)
        # The tick carries the state's rhythm so the frontend can step one
        # number per tick_seconds.
        self.assertEqual(ticks[0]["tick_seconds"], 0.1)
        self.assertEqual(ticks[0]["duration_seconds"], 0.3)

    def test_early_intent_wins_against_expiring_timer(self):
        payload = machine()
        payload["states"]["rules"]["on_intent"]["confirm"] = {"to": "timed"}
        payload["states"]["timed"] = {
            "duration": 0.5,
            "tick_seconds": 0.1,
            "on_intent": {"skip": {"to": "ready"}},
            "on_expire": {"to": "analysis"},
        }
        round_ = self.make_round(machine_payload=payload)
        round_.submit_intent("confirm")
        self.assertTrue(wait_for(lambda: round_.snapshot()["state"] == "timed"))
        time.sleep(0.1)
        round_.submit_intent("skip")
        self.assertTrue(wait_for(lambda: round_.snapshot()["state"] == "ready", timeout=3))
        # The timer's expire transition must not override the intent's.
        time.sleep(0.7)
        self.assertEqual(round_.snapshot()["state"], "ready")

    def test_awaited_speech_blocks_sequence_until_acknowledged(self):
        round_ = self.make_round()
        round_.submit_intent("confirm")
        round_.submit_intent("start_shake")
        self.assertTrue(wait_for(lambda: round_.snapshot()["state"] == "shaking"))
        round_.submit_intent("stop_shake")
        self.assertTrue(wait_for(lambda: round_.snapshot()["state"] == "open_reveal"))
        self.assertTrue(
            wait_for(lambda: [s for s in self.events_of(round_, "speech") if s.get("audio")])
        )
        # "停" is awaited: the reveal line must not be emitted before the ack.
        time.sleep(0.3)
        speeches = self.events_of(round_, "speech")
        self.assertNotIn("开盖", [s.get("text") for s in speeches])
        first = next(s for s in speeches if s.get("audio") == "audio/stop.wav")
        self.assertEqual(first["mode"], "audio")
        self.assertTrue(first["await"])

        round_.submit_intent("speech_done", {"directive_id": first["directive_id"]})
        self.assertTrue(wait_for(lambda: "开盖" in [s.get("text") for s in self.events_of(round_, "speech")]))
        # Duration starts after the awaited sequence finishes issuing.
        self.assertTrue(wait_for(lambda: reached(round_, "analysis"), timeout=20))

    def test_await_fallback_timeout_advances_without_acknowledgement(self):
        round_ = self.make_round(await_fallback_seconds=0.3)
        round_.submit_intent("confirm")
        round_.submit_intent("start_shake")
        self.assertTrue(wait_for(lambda: round_.snapshot()["state"] == "shaking"))
        round_.submit_intent("stop_shake")
        # No speech_done is sent; the fallback must keep the round moving.
        self.assertTrue(wait_for(lambda: reached(round_, "analysis"), timeout=20))
        self.assertTrue(wait_for(lambda: self.events_of(round_, "speech_timeout")))

    def test_adjudication_result_routes_and_renders_placeholders(self):
        def adjudicate(manifest, on_event, is_cancelled, on_log):
            on_event({"event": "phase", "phase": "detecting"})
            return {
                "winner_role": "PLAYER",
                "player_score": 18,
                "agent_score": 12,
                "player_values": [6, 6, 6],
            }

        round_ = self.make_round(adjudicate_fn=adjudicate, await_fallback_seconds=0.2)
        round_.submit_intent("confirm")
        round_.submit_intent("start_shake")
        self.assertTrue(wait_for(lambda: round_.snapshot()["state"] == "shaking"))
        round_.submit_intent("stop_shake")
        self.assertTrue(wait_for(lambda: reached(round_, "analysis"), timeout=20))
        self.assertTrue(wait_for(lambda: round_.snapshot()["state"] == "result", timeout=20))
        snapshot = round_.snapshot()
        self.assertEqual(snapshot["result"]["winner_role"], "PLAYER")
        speeches = self.events_of(round_, "speech")
        announce = [s for s in speeches if "玩家赢" in s.get("text", "")]
        self.assertEqual(len(announce), 1)
        self.assertEqual(announce[0]["text"], "玩家赢 18")
        self.assertTrue(any(
            e.get("event") == "phase" and e.get("phase") == "detecting"
            for e in snapshot["events"]
        ))

    def test_diagnosis_routes_to_failed_state_and_retry_reenters_analysis(self):
        calls = []

        def adjudicate(manifest, on_event, is_cancelled, on_log):
            calls.append(1)
            return {"diagnosed": True, "retry_required": True,
                    "diagnosis": {"reason_code": "INCOMPLETE_OBJECTS", "message": "数量不完整"}}

        round_ = self.make_round(adjudicate_fn=adjudicate, await_fallback_seconds=0.2)
        round_.submit_intent("confirm")
        round_.submit_intent("start_shake")
        self.assertTrue(wait_for(lambda: round_.snapshot()["state"] == "shaking"))
        round_.submit_intent("stop_shake")
        self.assertTrue(wait_for(lambda: round_.snapshot()["state"] == "analysis_failed", timeout=20))
        round_.submit_intent("retry")
        self.assertTrue(wait_for(lambda: round_.snapshot()["state"] == "analysis_failed", timeout=20))
        self.assertGreaterEqual(len(calls), 2)
        self.assertIn("请重试", [s["text"] for s in self.events_of(round_, "speech")])

    def test_cancel_terminates_round_and_bridges_to_adjudication(self):
        import threading as _threading
        started = _threading.Event()

        def adjudicate(manifest, on_event, is_cancelled, on_log):
            started.set()
            while not is_cancelled():
                time.sleep(0.02)
            return {"diagnosed": True, "retry_required": True}

        round_ = self.make_round(adjudicate_fn=adjudicate, await_fallback_seconds=0.2)
        round_.submit_intent("confirm")
        round_.submit_intent("start_shake")
        self.assertTrue(wait_for(lambda: round_.snapshot()["state"] == "shaking"))
        round_.submit_intent("stop_shake")
        self.assertTrue(wait_for(lambda: reached(round_, "analysis"), timeout=20))
        self.assertTrue(started.wait(timeout=2))
        round_.cancel()
        snapshot = round_.snapshot()
        self.assertEqual(snapshot["status"], "cancelled")
        self.assertTrue(wait_for(lambda: snapshot["status"] == "cancelled"))
        # The cancelled adjudication must not drive a transition afterwards.
        time.sleep(0.3)
        self.assertEqual(round_.snapshot()["status"], "cancelled")
        with self.assertRaises(RoundClosedError):
            round_.submit_intent("retry")

    def test_adjudication_failure_ends_round_as_error(self):
        def adjudicate(manifest, on_event, is_cancelled, on_log):
            raise RuntimeError("YOLO runtime exited")

        round_ = self.make_round(adjudicate_fn=adjudicate, await_fallback_seconds=0.2)
        round_.submit_intent("confirm")
        round_.submit_intent("start_shake")
        self.assertTrue(wait_for(lambda: round_.snapshot()["state"] == "shaking"))
        round_.submit_intent("stop_shake")
        self.assertTrue(wait_for(lambda: round_.snapshot()["status"] == "error", timeout=20))
        self.assertIn("YOLO runtime exited", round_.snapshot()["error"])

    def test_late_speech_done_is_idempotent(self):
        round_ = self.make_round()
        # An acknowledgement for an unknown directive must not raise.
        round_.submit_intent("speech_done", {"directive_id": "nope"})
        self.assertEqual(round_.snapshot()["state"], "rules")


if __name__ == "__main__":
    unittest.main()
