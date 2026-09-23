"""Tests for robot actions: schema shape, engine dispatch, and routing.

The engine-level tests drive a scripted robot callable through a state graph
shaped like the production dice wiring (grasp during shake_countdown, shake
at shaking entry, command-scoped completion routes, arm_failed/manual
fallbacks, result gestures).
"""
from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.components import Component, _validate_component_contract  # noqa: E402
from core.robot import RobotProvider  # noqa: E402
from core.state_machine import GameRound  # noqa: E402
from core.state_schema import StateMachineError, validate_state_machine  # noqa: E402

from test_state_machine import machine, manifest_with, reached, wait_for  # noqa: E402


def robot_machine(**overrides):
    """A state graph shaped like the production dice robot wiring."""
    states = {
        "rules": {
            "on_enter": [{"action": "speech", "mode": "tts_local", "text": "规则"}],
            "on_intent": {"confirm": {"to": "ready"}, "back": {"exit": True}},
        },
        "ready": {
            "on_enter": [
                {"action": "speech", "mode": "tts_local", "text": "准备"},
            ],
            "on_intent": {
                "start_shake": {"to": "game_start"},
                "back": {"exit": True},
            },
        },
        "game_start": {
            "duration": 0.3,
            "on_enter": [
                {"action": "robot", "command": "grasp_cup"},
            ],
            "on_expire": {"to": "shake_countdown"},
            "on_event": {"robot.grasp_cup.failed": {"to": "arm_failed"}},
        },
        "shake_countdown": {
            "duration": 0.3,
            "tick_seconds": 0.1,
            "on_enter": [
                {"action": "speech", "mode": "audio", "audio": "audio/warm.wav"},
            ],
            "on_expire": {"to": "shaking"},
            "on_event": {"robot.grasp_cup.failed": {"to": "arm_failed"}},
        },
        "shaking": {
            "on_enter": [
                {"action": "speech", "mode": "tts_local", "text": "摇骰进行中"},
                {"action": "robot", "command": "shake_dice"},
            ],
            "on_event": {
                "robot.shake_dice.completed": {"to": "vision_countdown"},
                "robot.shake_dice.failed": {"to": "arm_failed"},
            },
        },
        "vision_countdown": {
            "duration": 0.2,
            "on_expire": {"to": "analysis"},
        },
        "analysis": {
            "on_enter": [{"action": "adjudicate"}],
            "on_event": {
                "adjudication.result": {"to": "result"},
                "adjudication.diagnosis": {"to": "analysis_failed"},
            },
        },
        "analysis_failed": {
            "on_enter": [{"action": "speech", "mode": "tts_local", "text": "请重试"}],
            "on_intent": {"retry": {"to": "analysis"}, "back": {"exit": True}},
        },
        "result": {
            "on_enter": [
                {
                    "action": "robot",
                    "command": "feedback",
                    "select_by": "winner_role",
                    "cases": {"PLAYER": "lose", "AGENT": "win", "TIE": "draw"},
                },
            ],
            "on_intent": {"new_round": {"to": "ready"}, "back": {"exit": True}},
        },
        "arm_failed": {
            "on_enter": [{"action": "speech", "mode": "tts_local", "text": "臂失败"}],
            "on_intent": {
                "retry": {"to": "game_start"},
                "manual": {"to": "manual_shaking"},
                "back": {"exit": True},
            },
        },
        "manual_shaking": {
            "duration": 30,
            "on_intent": {"stop_shake": {"to": "vision_countdown"}},
            "on_expire": {"to": "vision_countdown"},
        },
    }
    payload = {"schema_version": 1, "initial": "rules", "states": states}
    payload.update(overrides)
    return payload


class _ScriptedRobot:
    """Fake robot_fn with per-command scripted handlers and call recording."""

    def __init__(self):
        self.calls: list[dict[str, Any]] = []
        self._handlers: list[tuple[str, Callable[..., dict[str, Any]]]] = []

    def on(self, command: str, handler: Callable[..., dict[str, Any]]) -> None:
        self._handlers.append((command, handler))

    def __call__(self, action, on_event, is_cancelled, on_log):
        command = str(action.get("command") or "")
        self.calls.append(dict(action))
        for matched, handler in self._handlers:
            if matched == command:
                return handler(action, on_event, is_cancelled)
        return {"status": "completed"}


def _blocking(outcome: dict[str, Any], *, gate: threading.Event):
    """Handler body that waits on a gate before returning an outcome."""

    def handler(action, on_event, is_cancelled):
        gate.wait(10)
        return dict(outcome)

    return handler


def _failing(reason: str):
    def handler(action, on_event, is_cancelled):
        return {"status": "failed", "reason": reason}

    return handler


def start_robot_round(
    robot_fn,
    *,
    machine_payload=None,
    winner_role: str = "AGENT",
) -> GameRound:
    def adjudicate(manifest, on_event, is_cancelled, on_log):
        return {
            "winner_role": winner_role,
            "player_score": 3,
            "agent_score": 5,
        }

    round_ = GameRound(
        game_id="dice",
        manifest=manifest_with(machine_payload or robot_machine()),
        adjudicate_fn=adjudicate,
        robot_fn=robot_fn,
        log=lambda line: None,
    )
    round_.start()
    return round_


def drive_to_shaking(round_: GameRound) -> None:
    round_.submit_intent("confirm")
    round_.submit_intent("start_shake")
    assert wait_for(lambda: reached(round_, "shaking")), "never reached shaking"


class RobotSchemaTests(unittest.TestCase):
    def _machine_with_action(self, action):
        payload = machine()
        payload["states"]["rules"]["on_enter"] = [action]
        return payload

    def test_grasp_and_shake_actions_validate(self):
        validate_state_machine(
            self._machine_with_action({"action": "robot", "command": "grasp_cup"}), "dice"
        )
        validate_state_machine(
            self._machine_with_action(
                {"action": "robot", "command": "shake_dice", "timeout_seconds": 60}
            ),
            "dice",
        )

    def test_unknown_command_is_rejected(self):
        with self.assertRaises(StateMachineError):
            validate_state_machine(
                self._machine_with_action({"action": "robot", "command": "shake"}), "dice"
            )

    def test_robot_actions_cannot_await(self):
        with self.assertRaises(StateMachineError):
            validate_state_machine(
                self._machine_with_action(
                    {"action": "robot", "command": "grasp_cup", "await": True}
                ),
                "dice",
            )

    def test_extra_keys_are_rejected(self):
        with self.assertRaises(StateMachineError):
            validate_state_machine(
                self._machine_with_action(
                    {"action": "robot", "command": "grasp_cup", "kind": "win"}
                ),
                "dice",
            )
        with self.assertRaises(StateMachineError):
            validate_state_machine(
                self._machine_with_action(
                    {"action": "robot", "command": "reset_home", "timeout_seconds": 5}
                ),
                "dice",
            )

    def test_timeout_seconds_must_be_positive(self):
        for bad in (0, -3, "soon"):
            with self.assertRaises(StateMachineError):
                validate_state_machine(
                    self._machine_with_action(
                        {"action": "robot", "command": "shake_dice", "timeout_seconds": bad}
                    ),
                    "dice",
                )

    def test_feedback_requires_complete_winner_role_cases(self):
        good = {
            "action": "robot",
            "command": "feedback",
            "select_by": "winner_role",
            "cases": {"PLAYER": "lose", "AGENT": "win", "TIE": "draw"},
        }
        validate_state_machine(self._machine_with_action(good), "dice")

        no_select = dict(good)
        del no_select["select_by"]
        with self.assertRaises(StateMachineError):
            validate_state_machine(self._machine_with_action(no_select), "dice")

        missing_case = dict(good)
        cases = dict(good["cases"])
        del cases["TIE"]
        missing_case["cases"] = cases
        with self.assertRaises(StateMachineError):
            validate_state_machine(self._machine_with_action(missing_case), "dice")

        bad_kind = dict(good)
        bad_kind["cases"] = {"PLAYER": "lose", "AGENT": "win", "TIE": "cheer"}
        with self.assertRaises(StateMachineError):
            validate_state_machine(self._machine_with_action(bad_kind), "dice")

    def test_feedback_accepts_no_extra_keys(self):
        action = {
            "action": "robot",
            "command": "feedback",
            "select_by": "winner_role",
            "cases": {"PLAYER": "lose", "AGENT": "win", "TIE": "draw"},
            "timeout_seconds": 10,
        }
        with self.assertRaises(StateMachineError):
            validate_state_machine(self._machine_with_action(action), "dice")


class RobotEngineTests(unittest.TestCase):
    def test_robot_commands_run_alongside_the_countdown(self):
        robot = _ScriptedRobot()
        grasp_gate = threading.Event()
        robot.on("grasp_cup", _blocking({"status": "completed"}, gate=grasp_gate))
        round_ = start_robot_round(robot)

        round_.submit_intent("confirm")
        round_.submit_intent("start_shake")
        # The countdown (0.3s) expires while the grasp is still gated: the
        # state worker must not block on physical motion.
        self.assertTrue(wait_for(lambda: reached(round_, "shaking")))
        self.assertFalse(grasp_gate.is_set())
        self.assertEqual(round_.snapshot()["status"], "running")
        round_.cancel()
        grasp_gate.set()

    def test_late_grasp_completion_does_not_advance_shaking(self):
        """The routing regression this engine exists to prevent.

        The grasp outlives shake_countdown, so its completion fires while the
        round is in shaking.  A command-scoped route name means that outcome
        must fall through as unrouted — not be mistaken for the shake.
        """
        robot = _ScriptedRobot()
        grasp_gate = threading.Event()
        shake_gate = threading.Event()
        robot.on("grasp_cup", _blocking({"status": "completed"}, gate=grasp_gate))
        robot.on("shake_dice", _blocking({"status": "completed"}, gate=shake_gate))
        round_ = start_robot_round(robot)

        drive_to_shaking(round_)
        grasp_gate.set()  # grasp finishes *after* shaking began
        time.sleep(0.4)
        self.assertEqual(round_.snapshot()["state"], "shaking")
        self.assertFalse(reached(round_, "vision_countdown"))

        shake_gate.set()  # the real completion advances
        self.assertTrue(wait_for(lambda: reached(round_, "vision_countdown")))
        round_.cancel()

    def test_grasp_failure_routes_to_arm_failed(self):
        robot = _ScriptedRobot()
        robot.on("grasp_cup", _failing("Hand start exceeds 0.5 degree motion bound"))
        round_ = start_robot_round(robot)

        round_.submit_intent("confirm")
        round_.submit_intent("start_shake")
        self.assertTrue(wait_for(lambda: reached(round_, "arm_failed")))
        events = round_.snapshot()["events"]
        result_events = [e for e in events if e.get("event") == "robot_result"]
        self.assertTrue(
            any(
                e.get("route") == "robot.grasp_cup.failed"
                and "Hand start" in str(e.get("reason"))
                for e in result_events
            )
        )
        round_.cancel()

    def test_shake_failure_routes_to_arm_failed(self):
        robot = _ScriptedRobot()
        robot.on("shake_dice", _failing("phase LOWER failed"))
        round_ = start_robot_round(robot)

        drive_to_shaking(round_)
        self.assertTrue(wait_for(lambda: reached(round_, "arm_failed")))
        round_.cancel()

    def test_arm_failed_retry_reruns_the_grasp(self):
        robot = _ScriptedRobot()
        attempt = {"count": 0}
        lock = threading.Lock()

        def grasp(action, on_event, is_cancelled):
            with lock:
                attempt["count"] += 1
                return (
                    {"status": "failed", "reason": "first attempt fails"}
                    if attempt["count"] == 1
                    else {"status": "completed"}
                )

        robot.on("grasp_cup", grasp)
        round_ = start_robot_round(robot)

        round_.submit_intent("confirm")
        round_.submit_intent("start_shake")
        self.assertTrue(wait_for(lambda: reached(round_, "arm_failed")))
        round_.submit_intent("retry")
        self.assertTrue(wait_for(lambda: attempt["count"] >= 2))
        # The retried round runs the whole chain through adjudication.
        self.assertTrue(wait_for(lambda: reached(round_, "result"), timeout=15))
        round_.cancel()

    def test_manual_shaking_fallback_bypasses_the_robot(self):
        robot = _ScriptedRobot()
        robot.on("grasp_cup", _failing("cup not found"))
        round_ = start_robot_round(robot)

        round_.submit_intent("confirm")
        round_.submit_intent("start_shake")
        self.assertTrue(wait_for(lambda: reached(round_, "arm_failed")))
        round_.submit_intent("manual")
        self.assertTrue(wait_for(lambda: reached(round_, "manual_shaking")))
        round_.submit_intent("stop_shake")
        self.assertTrue(wait_for(lambda: reached(round_, "vision_countdown")))
        round_.cancel()

    def test_feedback_kind_follows_winner_role_and_unrouted_is_observed(self):
        robot = _ScriptedRobot()
        round_ = start_robot_round(robot, winner_role="AGENT")

        round_.submit_intent("confirm")
        round_.submit_intent("start_shake")
        self.assertTrue(wait_for(lambda: reached(round_, "result"), timeout=15))
        feedback_calls = [c for c in robot.calls if c.get("command") == "feedback"]
        self.assertEqual(len(feedback_calls), 1)
        self.assertEqual(feedback_calls[0].get("kind"), "win")

        # result declares no route for the gesture's completion: an unrouted
        # observation, and the round keeps running.
        self.assertTrue(
            wait_for(
                lambda: any(
                    e.get("event") == "robot_unrouted"
                    and e.get("route") == "robot.feedback.completed"
                    for e in round_.snapshot()["events"]
                )
            )
        )
        self.assertEqual(round_.snapshot()["status"], "running")
        round_.cancel()

    def test_robot_progress_events_reach_the_round_stream(self):
        robot = _ScriptedRobot()

        def grasp(action, on_event, is_cancelled):
            on_event({"event": "robot", "phase": "CAPTURE", "zh": "定位杯子", "progress": "2/10"})
            return {"status": "completed"}

        robot.on("grasp_cup", grasp)
        round_ = start_robot_round(robot)

        round_.submit_intent("confirm")
        round_.submit_intent("start_shake")
        self.assertTrue(
            wait_for(
                lambda: any(
                    e.get("event") == "robot" and e.get("phase") == "CAPTURE"
                    for e in round_.snapshot()["events"]
                )
            )
        )
        round_.cancel()

    def test_cancel_interrupts_the_active_robot_command(self):
        robot = _ScriptedRobot()
        interrupted = threading.Event()

        def shake(action, on_event, is_cancelled):
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if is_cancelled():
                    interrupted.set()
                    return {"status": "failed", "reason": "interrupted"}
                time.sleep(0.02)
            return {"status": "completed"}

        robot.on("shake_dice", shake)
        round_ = start_robot_round(robot)

        drive_to_shaking(round_)
        round_.cancel()
        self.assertTrue(interrupted.wait(5))

    def test_missing_robot_provider_fails_the_round(self):
        round_ = GameRound(
            game_id="dice",
            manifest=manifest_with(robot_machine()),
            adjudicate_fn=lambda m, e, c, l: {"winner_role": "TIE"},
            robot_fn=None,
            log=lambda line: None,
        )
        round_.start()
        # game_start 的 grasp 是本图第一个 robot 动作；没接 provider 的回合
        # 必须在真正用到机械臂的那一刻大声失败。
        round_.submit_intent("confirm")
        round_.submit_intent("start_shake")
        self.assertTrue(wait_for(lambda: round_.snapshot()["status"] == "error"))
        self.assertIn("robot", round_.snapshot()["error"])


class RobotProviderContractTests(unittest.TestCase):
    def test_reset_home_floor_is_noop(self):
        provider = RobotProvider()
        result = provider.reset_home(on_event=lambda event: None, is_cancelled=lambda: False)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["noop"])

    def test_robot_type_requires_the_robot_base_class(self):
        class FakeRobot(Component):
            id = "fake_robot"
            type = "robot"

        with self.assertRaises(ValueError):
            _validate_component_contract(FakeRobot())


if __name__ == "__main__":
    unittest.main()
