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
from unittest import mock

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
import select
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

# 有状态推进（镜像真实 ControlSession）：advance 从当前位置顺序跑到目标；
# RETURN_HOME 完成发 run_completed 并自动复位回空闲。
pos = 0

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
    if command == "query_pose":
        # Pose probe: mirror the demo's at-home verdict. A one-shot mode file
        # ("pose_fail") simulates a CAN/worker failure (rejected, session alive).
        if mode == "pose_fail":
            emit("rejected", id=rid, code="pose_unavailable",
                 message="RuntimeError: SDK worker exited")
            continue
        at_home = (MODE_FILE.parent / "pose_away").exists() is False
        joints = [0.0, -1.2217, -1.5708, 1.7453, -0.1745, -0.0873, 0.0873]
        emit("pose", id=rid, joints_rad=joints,
             delta_deg=[0.0] * 7 if at_home else [30.0] * 7, at_home=at_home)
        continue
    if command == "action":
        name = request.get("name")
        # reject_prep 用独立持久标记文件（set_mode 的一次性模式会被链内
        # 前置的 reload 命令提前消费掉）。
        if (MODE_FILE.parent / "reject_prep").exists() and name == "rps-ready":
            emit("rejected", id=rid, code="unknown_action", message="unknown action rps-ready")
            continue
        emit("action_started", id=rid, name=name)
        emit("action_completed", id=rid, name=name,
             receipt="/tmp/receipt.json", elapsed_s=0.01)
        continue
    if command == "advance":
        target = (request.get("until") or "GRIP").upper()
        if mode == "reject_advance":
            emit("rejected", id=rid, code="flow_in_progress", message="抓取流程进行中")
            continue
        if mode == "recovering_failure":
            # demo 2026-09-25 失败自愈：failed → recovery_started → 归位 →
            # recovered，进程不退出、继续常驻服务。
            emit("failed", id=rid, phase="GRIP", error="ValueError: planned move rejected")
            emit("recovery_started", id=rid, action="home", cause="ValueError: planned move rejected")
            emit("recovered", id=rid, action="home")
            continue
        if mode == "recovering_failure_delayed":
            emit("failed", id=rid, phase="GRIP", error="ValueError: planned move rejected")
            emit("recovery_started", id="unrelated-command", action="home")
            emit("recovery_started", id=rid, action="home")
            (MODE_FILE.parent / "recovery_waiting").write_text("", encoding="utf-8")
            while not (MODE_FILE.parent / "release_recovery").exists():
                if select.select([sys.stdin], [], [], 0)[0]:
                    (MODE_FILE.parent / "queued_during_recovery").write_text("", encoding="utf-8")
                time.sleep(0.01)
            emit("recovered", id=rid, action="home")
            pos = 0
            continue
        if mode == "recovery_home_failed":
            emit("failed", id=rid, phase="GRIP", error="ValueError: planned move rejected")
            emit("recovery_started", id=rid, action="home")
            emit("failed", id=rid, phase="recovery:home",
                 error="RuntimeError: home motion timed out")
            sys.exit(2)
        if mode == "fail_exit":
            emit("failed", id=rid, phase="GRIP", error="ValueError: planned move rejected")
            sys.exit(2)
        if mode == "hand_start":
            emit("failed", id=rid, phase="GRIP",
                 error="ValueError: Hand start exceeds 0.5 degree motion bound")
            sys.exit(2)
        if mode == "hand_start_recovering":
            emit("failed", id=rid, phase="GRIP",
                 error="ValueError: Hand start exceeds 0.5 degree motion bound")
            emit("recovery_started", id=rid, action="home")
            (MODE_FILE.parent / "recovery_waiting").write_text("", encoding="utf-8")
            while not (MODE_FILE.parent / "release_recovery").exists():
                time.sleep(0.01)
            emit("recovered", id=rid, action="home")
            pos = 0
            continue
        if mode == "hand_start_recovery_home_failed":
            emit("failed", id=rid, phase="GRIP",
                 error="ValueError: Hand start exceeds 0.5 degree motion bound")
            emit("recovery_started", id=rid, action="home")
            emit("failed", id=rid, phase="recovery:home",
                 error="RuntimeError: home motion timed out")
            sys.exit(2)
        if mode == "hang":
            emit("phase_started", id=rid, phase="CAPTURE")
            time.sleep(600)
            continue
        target_index = PHASES.index(target)
        if target_index < pos:
            emit("rejected", id=rid, code="already_completed",
                 message="目标阶段已经完成，不能重复执行")
            continue
        for phase in PHASES[pos : target_index + 1]:
            emit("phase_started", id=rid, phase=phase)
            emit("phase_completed", id=rid, phase=phase, elapsed_s=0.01)
        pos = target_index + 1
        if pos == len(PHASES):
            emit("run_completed", id=rid, runs=1)
            pos = 0
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

    def _wait_resident_dead(self, timeout=5.0):
        """等自愈失败（legacy exit）的常驻进程真正退出。

        failed 事件先于进程退出落地（几十毫秒窗口）；新契约不在 failed
        时刻打断进程，死句柄由下一次 _ensure_running 延迟清理——测试里
        先等 poll() 确认死透再断言后续行为。
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            resident = self.provider._resident
            if resident is None or resident.process.poll() is not None:
                return True
            time.sleep(0.02)
        return False

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

    def test_prewarm_then_grasp_cup_lifts_before_shaking(self):
        self.provider.ensure_started()
        outcome = self.provider.grasp_cup(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "completed")
        self.assertGreaterEqual(outcome.get("elapsed", 0), 0)
        phases = self.phases_seen()
        self.assertIn("CAPTURE", phases)
        self.assertIn("GRIP", phases)
        # 升起（LIFT）并入抓取链（用户 2026-09-23 晚拍板）：过场结束时臂
        # 举杯在空中待命；整个抓取链仍不得有任何摇的动作。
        self.assertIn("LIFT", phases)
        self.assertNotIn("SHAKE", phases)
        self.assertNotIn("LOWER", phases)
        zh = {e.get("phase"): e.get("zh") for e in self.events if e.get("event") == "robot"}
        self.assertEqual(zh.get("CAPTURE"), "定位杯子")
        lift_event = next(
            e for e in self.events
            if e.get("event") == "robot" and e.get("phase") == "LIFT"
        )
        self.assertEqual(lift_event.get("progress"), "6/10")

    def test_shake_dice_runs_the_full_chain_to_return_home(self):
        # 游戏真实序列：先 grasp（推进到 LIFT，杯已在空中），再 shake——
        # 有状态 resident 下 shake 从 SHAKE 直接开始，不再有抬起。
        self.provider.grasp_cup(on_event=self.events.append, is_cancelled=lambda: False)
        self.events.clear()
        outcome = self.provider.shake_dice(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "completed")
        phases = self.phases_seen()
        self.assertEqual(phases[0], "SHAKE")
        for phase in ("SHAKE", "LOWER", "OPEN", "RETURN_HOME"):
            self.assertIn(phase, phases)
        self.assertNotIn("LIFT", phases)
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

    def test_throw_gesture_maps_rps_vocabulary_to_demo_actions(self):
        """rps 出拳：游戏词表（石头/剪刀/布）→ demo 动作 rock/scissors/paper，
        每次出拳前先连贯执行 rps-ready 预备动作（单锁两段链）；未知词表值
        直接失败（不发送任何动作）。"""
        for gesture, action in (("石头", "rock"), ("剪刀", "scissors"), ("布", "paper")):
            outcome = self.provider.throw_gesture(
                gesture, on_event=self.events.append, is_cancelled=lambda: False
            )
            self.assertEqual(outcome["status"], "completed")
            self.assertEqual(outcome["gesture"], gesture)
        received = [
            json.loads(line)
            for line in (self.demo_root / "received.log").read_text().splitlines()
        ]
        names = [c.get("name") for c in received if c.get("command") == "action"]
        self.assertEqual(
            names,
            ["rps-ready", "rock", "rps-ready", "scissors", "rps-ready", "paper"],
        )
        commands = [c.get("command") for c in received if isinstance(c, dict)]
        self.assertEqual(commands.count("reload"), 6)  # 每段静态动作前都热重载

        outcome = self.provider.throw_gesture(
            "布匹", on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "failed")
        self.assertIn("unknown throw gesture", str(outcome["reason"]))

    def test_throw_gesture_prep_failure_fails_the_chain(self):
        """预备段（rps-ready）被拒/失败 → 整链失败，拳形动作不发送。"""
        (self.demo_root / "reject_prep").write_text("", encoding="utf-8")
        outcome = self.provider.throw_gesture(
            "石头", on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "failed")
        self.assertIn("prep action", str(outcome.get("reason")))
        received = [
            json.loads(line)
            for line in (self.demo_root / "received.log").read_text().splitlines()
        ]
        # 只有预备段（及其 reload）被发送，rock 从未到达。
        sent = [c.get("name") for c in received if c.get("command") == "action"]
        self.assertEqual(sent, ["rps-ready"])

    def test_throw_gesture_none_picks_a_random_gesture_and_reports_it(self):
        """gesture=None（play 提前出拳，拳形尚不可知）：臂侧从词表随机
        选一种，outcome 必须回报实际执行的那份拳形——单一事实源。"""
        with mock.patch(
            "components.robot_arm_nero.provider.random.choice",
            side_effect=["石头", "剪刀", "布"],
        ):
            outcomes = [
                self.provider.throw_gesture(
                    None, on_event=self.events.append, is_cancelled=lambda: False
                )
                for _ in range(3)
            ]
        self.assertEqual([o["gesture"] for o in outcomes], ["石头", "剪刀", "布"])
        for outcome in outcomes:
            self.assertEqual(outcome["status"], "completed")
        received = [
            json.loads(line)
            for line in (self.demo_root / "received.log").read_text().splitlines()
        ]
        names = [c.get("name") for c in received if c.get("command") == "action"]
        self.assertEqual(
            names,
            ["rps-ready", "rock", "rps-ready", "scissors", "rps-ready", "paper"],
        )

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
        # 2026-09-25 新契约：failed 不再立即 reap（demo 自愈世界不能打断
        # 活常驻）；进程死透由延迟清理兜底，下条命令自动重启。
        self.assertTrue(self._wait_resident_dead())

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

    def test_hand_start_retry_waits_for_live_recovery(self):
        self.provider.ensure_started()
        resident = self.provider._resident
        self.set_mode("hand_start_recovering")
        result = {}
        worker = threading.Thread(target=lambda: result.setdefault(
            "outcome", self.provider.grasp_cup(
                on_event=self.events.append, is_cancelled=lambda: False
            )
        ))
        worker.start()
        release = self.demo_root / "release_recovery"
        try:
            deadline = time.monotonic() + 5.0
            while not (self.demo_root / "recovery_waiting").exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue((self.demo_root / "recovery_waiting").exists())
            release.write_text("", encoding="utf-8")
            worker.join(timeout=5.0)
            self.assertFalse(worker.is_alive())
            self.assertEqual(result["outcome"]["status"], "completed")
            self.assertTrue(result["outcome"].get("auto_retried"))
            self.assertIs(self.provider._resident, resident)
            self.assertIsNone(resident.process.poll())
        finally:
            release.write_text("", encoding="utf-8")
            worker.join(timeout=5.0)

    def test_hand_start_does_not_retry_after_recovery_home_fails(self):
        self.provider.ensure_started()
        self.set_mode("hand_start_recovery_home_failed")
        outcome = self.provider.grasp_cup(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "failed")
        self.assertIn("home motion timed out", outcome["reason"])
        self.assertEqual(self._received_commands().count("advance"), 1)
        self.assertTrue(self._wait_resident_dead())

    def test_hand_start_cancelled_before_retry_sends_no_motion(self):
        self.provider.ensure_started()
        self.set_mode("hand_start_recovering")
        cancelled = threading.Event()
        release = self.demo_root / "release_recovery"

        def on_event(event):
            if event.get("phase") == "AUTO_RETRY":
                cancelled.set()

        def release_recovery():
            deadline = time.monotonic() + 5.0
            while not (self.demo_root / "recovery_waiting").exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            release.write_text("", encoding="utf-8")

        worker = threading.Thread(target=release_recovery)
        worker.start()
        try:
            outcome = self.provider.grasp_cup(
                on_event=on_event, is_cancelled=cancelled.is_set
            )
            self.assertTrue(cancelled.is_set())
            self.assertEqual(outcome["status"], "failed")
            self.assertEqual(self._received_commands().count("advance"), 1)
        finally:
            release.write_text("", encoding="utf-8")
            worker.join(timeout=5.0)

    def test_hand_start_cancelled_during_recovery_keeps_homing(self):
        self.provider.ensure_started()
        resident = self.provider._resident
        self.set_mode("hand_start_recovering")
        cancelled = threading.Event()
        result = {}
        worker = threading.Thread(target=lambda: result.setdefault(
            "outcome", self.provider.grasp_cup(
                on_event=self.events.append, is_cancelled=cancelled.is_set
            )
        ))
        release = self.demo_root / "release_recovery"
        worker.start()
        try:
            deadline = time.monotonic() + 5.0
            while not (self.demo_root / "recovery_waiting").exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue((self.demo_root / "recovery_waiting").exists())
            cancelled.set()
            worker.join(timeout=5.0)
            self.assertFalse(worker.is_alive())
            self.assertEqual(result["outcome"]["status"], "failed")
            self.assertIsNone(resident.process.poll())
            release.write_text("", encoding="utf-8")
            self.assertTrue(resident.recovery_done.wait(5.0))
            self.assertIsNone(resident.process.poll())
            self.assertEqual(self._received_commands().count("advance"), 1)
        finally:
            release.write_text("", encoding="utf-8")
            worker.join(timeout=5.0)

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

    def test_reset_home_revives_a_dead_resident(self):
        """归位不变量：流程失败自杀后，reset_home 也要能拉起新常驻完成归位。

        旧语义是"resident 死了就 skipped"——失败后臂停在原位（握着杯子）
        无人处理。新语义：reset_home 是唯一允许复活死 resident 的命令。
        """
        outcome = self.provider.reset_home(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "completed")
        self.assertFalse(outcome.get("skipped"))
        self.assertIsNotNone(self.provider._resident)
        received = [
            json.loads(line)
            for line in (self.demo_root / "received.log").read_text().splitlines()
        ]
        commands = [c.get("command") for c in received if isinstance(c, dict)]
        self.assertIn("action", commands)
        homes = [c for c in received if c.get("command") == "action" and c.get("name") == "home"]
        self.assertTrue(homes)

    def test_reset_home_after_phase_failure_still_homes(self):
        """阶段失败退出 demo 后（真实失败链路），归位照样复活执行。"""
        self.set_mode("fail_exit")
        outcome = self.provider.grasp_cup(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "failed")
        # 新契约：不立即 reap；等进程死透后归位走复活链（session 换新）。
        self.assertTrue(self._wait_resident_dead())
        old_session = self.provider._resident.session_dir if self.provider._resident else None
        outcome = self.provider.reset_home(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "completed")
        if old_session is not None:
            self.assertIsNot(self.provider._resident.session_dir, old_session)  # 复活

    def test_reset_home_refuses_after_shutdown(self):
        self.provider.shutdown()
        outcome = self.provider.reset_home(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "failed")
        self.assertIn("shutting down", outcome["reason"])

    def test_query_pose_reports_at_home_without_reloading_actions(self):
        """只读姿态探针：pose 事件解析出 at_home；不发 reload（非 action 命令）。"""
        self.provider.ensure_started()
        outcome = self.provider.query_pose(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "completed")
        self.assertTrue(outcome["at_home"])
        received = [
            json.loads(line)
            for line in (self.demo_root / "received.log").read_text().splitlines()
        ]
        commands = [c.get("command") for c in received if isinstance(c, dict)]
        self.assertEqual(commands.count("reload"), 0)

    def test_query_pose_reports_away_and_probe_failure(self):
        # 离家判定
        self.provider.ensure_started()
        (self.demo_root / "pose_away").write_text("", encoding="utf-8")
        outcome = self.provider.query_pose(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "completed")
        self.assertFalse(outcome["at_home"])
        # 探针失败（CAN/worker）→ failed + 常驻仍活着（rejected 不杀会话）
        self.set_mode("pose_fail")
        outcome = self.provider.query_pose(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "failed")
        self.assertIn("pose_unavailable", str(outcome.get("reason")))
        self.assertTrue(self.provider._resident is not None)

    def test_query_pose_never_revives_a_dead_resident(self):
        """巡检探针绝不复活：死常驻 = skipped（复活只属于 reset_home）。"""
        self.set_mode("fail_exit")
        outcome = self.provider.grasp_cup(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "failed")
        self.assertTrue(self._wait_resident_dead())
        outcome = self.provider.query_pose(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "skipped")

    # ---- ensure_home（2026-09-25 检查式归位） ----

    def _received_commands(self):
        return [
            json.loads(line).get("command")
            for line in (self.demo_root / "received.log").read_text().splitlines()
            if line.strip().startswith("{")
        ]

    def test_ensure_home_skips_motion_when_already_home(self):
        """在家 → 只发探针（query_pose），零动作不发 action home。"""
        self.provider.ensure_started()
        outcome = self.provider.ensure_home(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "completed")
        self.assertTrue(outcome["at_home"])
        commands = self._received_commands()
        self.assertEqual(commands.count("query_pose"), 1)
        self.assertEqual(
            sum(1 for c in commands if c == "action"), 0
        )

    def test_ensure_home_homes_when_off_home(self):
        """不在家 → 探针后走归位链（action home 发出）。"""
        self.provider.ensure_started()
        (self.demo_root / "pose_away").write_text("", encoding="utf-8")
        outcome = self.provider.ensure_home(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "completed")
        commands = self._received_commands()
        self.assertEqual(commands.count("query_pose"), 1)
        homes = [
            json.loads(line)
            for line in (self.demo_root / "received.log").read_text().splitlines()
            if "query_pose" not in line and '"home"' in line
        ]
        self.assertTrue(homes)

    def test_ensure_home_homes_conservatively_on_probe_failure(self):
        """探针失败（CAN/瞬断）→ 不知道状态就归位，绝不跳过。"""
        self.provider.ensure_started()
        self.set_mode("pose_fail")
        outcome = self.provider.ensure_home(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "completed")
        commands = self._received_commands()
        self.assertEqual(commands.count("query_pose"), 1)

    def test_ensure_home_revives_a_dead_resident(self):
        """死常驻 → 直接走归位链（复活语义），不发探针。"""
        self.set_mode("fail_exit")
        outcome = self.provider.grasp_cup(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "failed")
        outcome = self.provider.ensure_home(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "completed")
        commands = self._received_commands()
        self.assertEqual(commands.count("query_pose"), 0)

    def test_failed_then_recovery_keeps_the_resident_alive(self):
        """失败恢复协议（ChatGPT 2026-09-25 报的 bug）：demo 失败自愈后
        进程**不退出**——main 不得 wait+interrupt 打断归位、不得丢弃活常驻；
        下一条命令复用同一进程（自愈成果保住）。"""
        self.provider.ensure_started()
        session_dir = self.provider._resident.session_dir
        self.set_mode("recovering_failure")
        outcome = self.provider.grasp_cup(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "failed")
        self.assertIn("planned move rejected", str(outcome.get("reason")))
        # 完整序列转成 RECOVERY 进度（failed → recovery_started →
        # recovered；resolve 后通知仍在管道里，有界等齐）。
        deadline = time.monotonic() + 5.0
        zh = []
        while time.monotonic() < deadline:
            zh = [str(e.get("zh")) for e in self.events if e.get("event") == "robot"]
            if "失败自愈，机械臂归位中" in zh and "机械臂已恢复就位" in zh:
                break
            time.sleep(0.02)
        self.assertIn("失败自愈，机械臂归位中", zh)
        self.assertIn("机械臂已恢复就位", zh)
        # 修复前：process_exiting 标志会让 provider wait(5s)+interrupt 杀掉
        # 活着的自愈常驻并把 _resident 置 None。修复后常驻保留且存活。
        self.assertIsNotNone(self.provider._resident)
        self.assertTrue(self.provider._resident.alive)
        self.assertIs(self.provider._resident.process.poll(), None)
        # 下一条命令复用同一常驻（同 session），不再重启。
        outcome = self.provider.reset_home(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "completed")
        self.assertIs(self.provider._resident.session_dir, session_dir)

    def test_followup_waits_for_recovery_before_sending(self):
        self.provider.ensure_started()
        resident = self.provider._resident
        first_events = []
        second_events = []
        self.set_mode("recovering_failure_delayed")
        first = self.provider.grasp_cup(
            on_event=first_events.append, is_cancelled=lambda: False
        )
        self.assertEqual(first["status"], "failed")
        result = {}
        worker = threading.Thread(target=lambda: result.setdefault(
            "outcome", self.provider.reset_home(
                on_event=second_events.append, is_cancelled=lambda: False
            )
        ))
        release = self.demo_root / "release_recovery"
        worker.start()
        try:
            deadline = time.monotonic() + 5.0
            while not (self.demo_root / "recovery_waiting").exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue((self.demo_root / "recovery_waiting").exists())
            time.sleep(0.2)
            self.assertTrue(worker.is_alive())
            self.assertFalse((self.demo_root / "queued_during_recovery").exists())
            self.assertIsNone(resident.process.poll())
            release.write_text("", encoding="utf-8")
            worker.join(timeout=5.0)
            self.assertFalse(worker.is_alive())
            self.assertEqual(result["outcome"]["status"], "completed")
            self.assertEqual(
                [e.get("zh") for e in first_events].count("失败自愈，机械臂归位中"), 1
            )
            self.assertIn("机械臂已恢复就位", [e.get("zh") for e in first_events])
            self.assertNotIn("机械臂已恢复就位", [e.get("zh") for e in second_events])
        finally:
            release.write_text("", encoding="utf-8")
            worker.join(timeout=5.0)

    def test_followup_timeout_leaves_recovery_running(self):
        self.provider.ensure_started()
        resident = self.provider._resident
        self.set_mode("recovering_failure_delayed")
        first = self.provider.grasp_cup(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(first["status"], "failed")
        release = self.demo_root / "release_recovery"
        try:
            followup = self.provider.reset_home(
                on_event=self.events.append, is_cancelled=lambda: False,
                timeout_seconds=0.15,
            )
            self.assertEqual(followup["status"], "failed")
            self.assertIn("recovery still in progress", followup["reason"])
            self.assertFalse((self.demo_root / "queued_during_recovery").exists())
            self.assertIsNone(resident.process.poll())
            release.write_text("", encoding="utf-8")
            self.assertTrue(resident.recovery_done.wait(5.0))
            self.assertIsNone(resident.process.poll())
            retry = self.provider.reset_home(
                on_event=self.events.append, is_cancelled=lambda: False
            )
            self.assertEqual(retry["status"], "completed")
            self.assertIs(self.provider._resident, resident)
        finally:
            release.write_text("", encoding="utf-8")

    def test_followup_cancelled_at_recovery_handoff_sends_no_motion(self):
        self.provider.ensure_started()
        resident = self.provider._resident
        self.set_mode("recovering_failure_delayed")
        first = self.provider.grasp_cup(
            on_event=self.events.append, is_cancelled=lambda: False
        )
        self.assertEqual(first["status"], "failed")
        cancelled = threading.Event()
        result = {}
        worker = threading.Thread(target=lambda: result.setdefault(
            "outcome", self.provider.reset_home(
                on_event=self.events.append, is_cancelled=cancelled.is_set
            )
        ))
        release = self.demo_root / "release_recovery"
        worker.start()
        try:
            deadline = time.monotonic() + 5.0
            while not (self.demo_root / "recovery_waiting").exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue((self.demo_root / "recovery_waiting").exists())
            cancelled.set()
            release.write_text("", encoding="utf-8")
            worker.join(timeout=5.0)
            self.assertFalse(worker.is_alive())
            self.assertEqual(result["outcome"]["status"], "failed")
            self.assertEqual(self._received_commands().count("action"), 0)
            self.assertIsNone(resident.process.poll())
        finally:
            release.write_text("", encoding="utf-8")
            worker.join(timeout=5.0)

    def test_recovery_home_failure_is_reported(self):
        self.provider.ensure_started()
        self.set_mode("recovery_home_failed")
        events = []
        outcome = self.provider.grasp_cup(
            on_event=events.append, is_cancelled=lambda: False
        )
        self.assertEqual(outcome["status"], "failed")
        deadline = time.monotonic() + 5.0
        while not any(e.get("phase") == "RECOVERY_FAILED" for e in events) and time.monotonic() < deadline:
            time.sleep(0.01)
        failures = [e for e in events if e.get("phase") == "RECOVERY_FAILED"]
        self.assertEqual(len(failures), 1)
        self.assertIn("home motion timed out", failures[0]["reason"])
        self.assertTrue(self._wait_resident_dead())

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
        """Two concurrent commands must not interleave on one arm.

        用游戏的真实序列 grasp→shake：有状态 resident 下连续两次 grasp
        会被 already_completed 拒绝（与真实 ControlSession 一致），所以
        第二条命令换成 shake——它必须等第一条的臂锁释放后才开跑。
        """
        self.provider.ensure_started()
        results: list[dict] = []
        first_started = threading.Event()

        def run_grasp():
            first_started.set()
            results.append(
                self.provider.grasp_cup(on_event=self.events.append, is_cancelled=lambda: False)
            )

        def run_shake():
            results.append(
                self.provider.shake_dice(on_event=self.events.append, is_cancelled=lambda: False)
            )

        first = threading.Thread(target=run_grasp)
        first.start()
        first_started.wait(5)
        second = threading.Thread(target=run_shake)
        second.start()
        first.join(30)
        second.join(30)
        self.assertEqual([r["status"] for r in results], ["completed", "completed"])
        # 顺序保障：LIFT（grasp 链尾）先于 SHAKE（shake 链头）到达。
        phases = [str(e.get("phase")) for e in self.events if e.get("event") == "robot"]
        self.assertLess(phases.index("LIFT"), phases.index("SHAKE"))


if __name__ == "__main__":
    unittest.main()
