"""Protocol tests for the robot_arm_nero resident JSONL client.

Each test builds a throwaway "demo" directory whose ``run.sh`` execs a fake
resident speaking the dice_demo control protocol (ready/phase_started/
command_completed/rejected/failed/close).  Scenario modes are delivered via
a one-shot ``mode.txt`` consumed by the fake, so a mode like "fail_exit"
fires once and the next spawn behaves normally — exactly the
restart-and-recover paths the provider must handle.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from components.robot_arm_nero.provider import RobotArmNeroProvider  # noqa: E402

FAKE_PHASES = [
    "HOME", "CAPTURE", "PLAN", "APPROACH", "GRIP",
    "LIFT", "SHAKE", "LOWER", "OPEN", "RETURN_HOME",
]

FAKE_RESIDENT_SOURCE = '''
import json
import sys
import time
from pathlib import Path

PHASES = __FAKE_PHASES__

MODE_FILE = Path(__file__).parent / "mode.txt"


def read_mode(consume):
    try:
        mode = MODE_FILE.read_text().strip()
    except OSError:
        return ""
    if mode and consume:
        try:
            MODE_FILE.unlink()
        except OSError:
            pass
    return mode


def emit(event, **fields):
    sys.stdout.write(json.dumps(dict(event=event, **fields)) + "\\n")
    sys.stdout.flush()


# Startup modes are the only ones consumed at spawn time; per-command modes
# must survive the resident's boot.
if read_mode(consume=False) == "die_before_ready":
    read_mode(consume=True)
    sys.stderr.write("simulated can0 down\\n")
    sys.exit(2)

emit("ready", phases=PHASES, actions=["home", "yeah", "thumbs-up", "tie"],
     state_file="green_pipeline_state.json")

for line in sys.stdin:
    (MODE_FILE.parent / "received.log").open("a").write(line)
    try:
        request = json.loads(line)
    except json.JSONDecodeError:
        continue
    command = request.get("command")
    rid = request.get("id")
    mode = read_mode(consume=True)
    if command == "close":
        emit("closed", id=rid, state_file="green_pipeline_state.json")
        break
    if command == "reload":
        emit("actions_reloaded", id=rid, names=["home", "yeah", "thumbs-up", "tie"])
        continue
    if command == "status":
        emit("status", id=rid, next_phase="HOME", completed_phases=[])
        continue
    if command == "action":
        name = request.get("name")
        emit("action_started", id=rid, name=name)
        emit("action_completed", id=rid, name=name,
             receipt="/tmp/receipt.json", elapsed_s=0.01)
        continue
    if command == "advance":
        target = (request.get("until") or "GRIP").upper()
        if mode == "reject_advance":
            emit("rejected", id=rid, code="flow_in_progress", message="抓取流程进行中")
            continue
        if mode == "fail_exit":
            emit("failed", id=rid, phase="GRIP", error="ValueError: planned move rejected")
            sys.exit(2)
        if mode == "hand_start":
            emit("failed", id=rid, phase="GRIP",
                 error="ValueError: Hand start exceeds 0.5 degree motion bound")
            sys.exit(2)
        if mode == "hang":
            emit("phase_started", id=rid, phase="CAPTURE")
            time.sleep(600)
            continue
        for phase in PHASES[: PHASES.index(target) + 1]:
            emit("phase_started", id=rid, phase=phase)
            emit("phase_completed", id=rid, phase=phase, elapsed_s=0.01)
        if target == "RETURN_HOME":
            emit("run_completed", id=rid, runs=1)
        emit("command_completed", id=rid, through=target)
        continue
    emit("rejected", id=rid, code="invalid_command", message="unknown command")
'''.replace("__FAKE_PHASES__", repr(FAKE_PHASES))


class RobotArmNeroProtocolTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        (root / "configs").mkdir()
        (root / "configs" / "green_cup.json").write_text("{}", encoding="utf-8")
        (root / "configs" / "actions" / "gestures").mkdir(parents=True)
        (root / "configs" / "actions" / "gestures" / "result_feedback.json").write_text(
            "{}", encoding="utf-8"
        )
        (root / "configs" / "actions" / "home.json").write_text("{}", encoding="utf-8")
        (root / "vendor-site" / "pyAgxArm").mkdir(parents=True)
        (root / "vendor-site" / "pyrealsense2").mkdir(parents=True)
        (root / "cup_grasp_demo" / "datasets").mkdir(parents=True)
        (root / "fake_resident.py").write_text(FAKE_RESIDENT_SOURCE, encoding="utf-8")
        (root / "run.sh").write_text(
            '#!/usr/bin/env bash\nexec python3 "$(dirname "$0")/fake_resident.py"\n',
            encoding="utf-8",
        )
        self.demo_root = root
        self.provider = RobotArmNeroProvider()
        # Point at the fixture and keep every timeout short but generous for
        # slow CI machines; scenarios override per test where needed.
        self.provider._demo_root = root
        self.provider._ready_timeout = 15.0
        self.provider._sigint_grace = 5.0
        self.provider._timeouts = {key: 20.0 for key in self.provider._timeouts}
        self.events: list[dict] = []

    def tearDown(self):
        self.provider.shutdown()
        self._tmp.cleanup()

    def set_mode(self, mode: str) -> None:
        (self.demo_root / "mode.txt").write_text(mode, encoding="utf-8")

    def phases_seen(self) -> list[str]:
        return [str(e.get("phase")) for e in self.events if e.get("event") == "robot"]

    # ---- deployment ------------------------------------------------------

    def test_health_checks_demo_deployment(self):
        health = self.provider.health()
        self.assertTrue(health["ok"])
        self.assertEqual(health["demo_root"], str(self.demo_root))

        (self.demo_root / "run.sh").unlink()
        health = self.provider.health()
        self.assertFalse(health["ok"])
        self.assertIn("run.sh", str(health.get("error")))

    def test_health_flags_missing_gesture_library(self):
        """手势缺失时常驻会退化为无动作会话（手势/归位被拒）——部署期就该报出来。"""
        shutil.rmtree(self.demo_root / "configs" / "actions" / "gestures")
        health = self.provider.health()
        self.assertFalse(health["ok"])
        self.assertIn("gestures", str(health.get("error")))

    # ---- happy paths -----------------------------------------------------

    def test_prewarm_then_grasp_cup_stops_before_shaking(self):
        self.provider.ensure_started()
        outcome = self.provider.grasp_cup(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "completed")
        self.assertGreaterEqual(outcome.get("elapsed", 0), 0)
        phases = self.phases_seen()
        self.assertIn("CAPTURE", phases)
        self.assertIn("GRIP", phases)
        # The whole point of the split: no shake motion during the countdown.
        self.assertNotIn("SHAKE", phases)
        self.assertNotIn("LIFT", phases)
        zh = {e.get("phase"): e.get("zh") for e in self.events if e.get("event") == "robot"}
        self.assertEqual(zh.get("CAPTURE"), "定位杯子")
        grip_event = next(
            e for e in self.events
            if e.get("event") == "robot" and e.get("phase") == "GRIP"
        )
        self.assertEqual(grip_event.get("progress"), "5/10")

    def test_shake_dice_runs_the_full_chain_to_return_home(self):
        outcome = self.provider.shake_dice(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "completed")
        phases = self.phases_seen()
        for phase in ("LIFT", "SHAKE", "LOWER", "OPEN", "RETURN_HOME"):
            self.assertIn(phase, phases)
        return_event = next(
            e for e in self.events
            if e.get("event") == "robot" and e.get("phase") == "RETURN_HOME"
        )
        self.assertEqual(return_event.get("progress"), "10/10")

    def test_feedback_and_reset_home_use_static_actions(self):
        outcome = self.provider.feedback(
            "win", on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "completed")
        self.assertIn("手势 yeah", [str(e.get("zh")) for e in self.events])

        outcome = self.provider.reset_home(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "completed")

    def test_action_commands_refresh_the_gesture_table_first(self):
        """demo 2026-09-23 起手势表支持常驻热重载：动作命令前自动发 reload。

        demo 串行处理 stdin，reload 先到即先生效——改手势文件后下一个手势
        就用新表，无需重启任何东西。
        """
        outcome = self.provider.feedback(
            "win", on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "completed")
        received = [
            json.loads(line)
            for line in (self.demo_root / "received.log").read_text().splitlines()
        ]
        commands = [c.get("command") for c in received if isinstance(c, dict)]
        self.assertIn("reload", commands)
        self.assertIn("action", commands)
        self.assertLess(commands.index("reload"), commands.index("action"))

    def test_grasp_chain_never_sends_reload(self):
        """reload 只对静态动作有意义；抓取/摇骰链保持纯 advance 协议。"""
        outcome = self.provider.grasp_cup(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "completed")
        received = [
            json.loads(line)
            for line in (self.demo_root / "received.log").read_text().splitlines()
        ]
        commands = [c.get("command") for c in received if isinstance(c, dict)]
        self.assertNotIn("reload", commands)

    def test_unknown_feedback_kind_fails_without_spawning(self):
        outcome = self.provider.feedback(
            "cheer", on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "failed")
        self.assertIn("unknown feedback kind", outcome["reason"])
        self.assertIsNone(self.provider._resident)

    # ---- failure paths ---------------------------------------------------

    def test_rejected_command_reports_the_code(self):
        self.set_mode("reject_advance")
        outcome = self.provider.grasp_cup(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "failed")
        self.assertIn("rejected(flow_in_progress)", outcome["reason"])

    def test_phase_failure_fails_command_and_next_command_restarts(self):
        self.set_mode("fail_exit")
        outcome = self.provider.grasp_cup(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "failed")
        self.assertIn("planned move rejected", outcome["reason"])
        # A phase failure reaps the process and drops the resident eagerly.
        self.assertIsNone(self.provider._resident)

        outcome = self.provider.grasp_cup(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "completed")  # auto-restart worked

    def test_hand_start_failure_auto_retries_once(self):
        self.set_mode("hand_start")
        outcome = self.provider.grasp_cup(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "completed")
        self.assertTrue(outcome.get("auto_retried"))
        self.assertIn("AUTO_RETRY", self.phases_seen())

    def test_timeout_interrupts_the_resident(self):
        self.set_mode("hang")
        self.provider._timeouts["grasp_cup"] = 1.5
        outcome = self.provider.grasp_cup(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "failed")
        self.assertIn("timed out", outcome["reason"])
        # The interrupt path reaped the process even if the resident object
        # is only dropped lazily.
        self.assertTrue(
            self.provider._resident is None
            or self.provider._resident.process.poll() is not None
        )

    def test_cancel_interrupts_the_command(self):
        self.set_mode("hang")
        cancel_after = time.monotonic() + 0.5

        def is_cancelled() -> bool:
            return time.monotonic() >= cancel_after

        outcome = self.provider.grasp_cup(on_event=self.events.append, is_cancelled=is_cancelled)
        self.assertEqual(outcome["status"], "failed")
        self.assertIn("interrupted (cancelled)", outcome["reason"])

    def test_ready_failure_surfaces_the_log_tail(self):
        self.set_mode("die_before_ready")
        outcome = self.provider.grasp_cup(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "failed")
        self.assertIn("resident not ready", outcome["reason"])
        self.assertIn("simulated can0 down", outcome["reason"])

    def test_reset_home_skips_when_resident_not_running(self):
        outcome = self.provider.reset_home(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "completed")
        self.assertTrue(outcome.get("skipped"))

    def test_shutdown_closes_a_live_resident(self):
        self.provider.ensure_started()
        self.assertIsNotNone(self.provider._resident)
        self.provider.grasp_cup(on_event=self.events.append, is_cancelled=lambda: False)
        session_dir = self.provider._resident.session_dir
        self.provider.shutdown()
        self.assertIsNone(self.provider._resident)
        self.assertTrue((session_dir / "controller.log").exists())
        # A second shutdown is a harmless no-op.
        self.provider.shutdown()

    def test_commands_serialize_on_the_arm_lock(self):
        """Two concurrent commands must not interleave on one arm."""
        self.provider.ensure_started()
        results: list[dict] = []
        started = threading.Event()

        def run_grasp():
            started.set()
            results.append(
                self.provider.grasp_cup(on_event=self.events.append, is_cancelled=lambda: False)
            )

        first = threading.Thread(target=run_grasp)
        first.start()
        started.wait(5)
        second = threading.Thread(target=run_grasp)
        second.start()
        first.join(30)
        second.join(30)
        self.assertEqual([r["status"] for r in results], ["completed", "completed"])


if __name__ == "__main__":
    unittest.main()
