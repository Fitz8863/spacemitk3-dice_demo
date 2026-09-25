"""NERO robot-arm provider over the dice_demo resident JSONL controller.

Spawns one long-lived ``bash run.sh control --execute`` subprocess per board
session (contract: dice_demo ``docs/INTEGRATION.md`` §2) and drives it with
line-JSON commands. The resident pre-connects SDK/CAN, prewarms the camera
and loads the model once. A failed command may be followed by automatic home
recovery; the resident is reused after ``recovered`` and replaced if it exits.

Command mapping:

* ``grasp_cup``  → ``advance until LIFT``      (HOME→CAPTURE→PLAN→APPROACH→GRIP→LIFT; grasp and lift, no shaking)
* ``shake_dice`` → ``advance until RETURN_HOME`` (SHAKE→LOWER→OPEN→RETURN_HOME; the cup is already aloft)
* ``feedback``   → ``action name=yeah|thumbs-up|tie``
* ``throw_gesture`` → ``action name=rps-ready`` → ``action name=rock|paper|scissors`` (rps agent move: prep pose then the gesture, one locked chain; pose is held, no auto-home)
* ``reset_home`` → ``action name=home`` — parks the arm; the one command that
  revives a dead resident (a phase failure exits the demo, and the parking
  invariant still demands the arm be homed)
* ``query_pose`` → ``query_pose`` — read-only at-home probe for the idle
  patrol; the one command that must NOT revive (dead resident = skip)
* ``ensure_home`` → probe then maybe ``action name=home`` — checked homing
  (2026-09-25): at-home answers with zero motion, anything else falls
  through to the full reset_home chain (revive included)

Completion is judged strictly by id-correlated events (``command_completed`` /
``action_completed`` / ``rejected`` / ``failed``); progress phases are relayed
to the round stream as ``{"event": "robot", ...}``. ``Hand start`` failures
are retried once after recovery finishes or the resident exits. Round cancels
are tiered: only the long advance chains interrupt the resident; short safe
actions run to completion so a cancel never kills the resident mid-homing.
"""
from __future__ import annotations

import json
import os
import random
import signal
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from core.robot import (
    RobotCancelledFn,
    RobotEventFn,
    RobotProvider,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

# Pipeline phases in demo order; also the frontend progress denominator.
_PHASES = (
    "HOME",
    "CAPTURE",
    "PLAN",
    "APPROACH",
    "GRIP",
    "LIFT",
    "SHAKE",
    "LOWER",
    "OPEN",
    "RETURN_HOME",
)
_PHASE_LABELS_ZH = {
    "HOME": "归位",
    "CAPTURE": "定位杯子",
    "PLAN": "规划",
    "APPROACH": "靠近杯子",
    "GRIP": "抓取杯子",
    "LIFT": "抬起骰盅",
    "SHAKE": "摇骰",
    "LOWER": "放回骰盅",
    "OPEN": "松开手指",
    "RETURN_HOME": "收尾归位",
}
_FEEDBACK_ACTIONS = {"win": "yeah", "lose": "thumbs-up", "draw": "tie"}
# rps 游戏词表（games/rps/result.py GESTURES）→ demo 静态手势动作名。
_THROW_ACTIONS = {"石头": "rock", "剪刀": "scissors", "布": "paper"}
# 出拳前的预备动作（手前伸，与 home 仅肘关节之差）——先预备再亮拳，
# 两段连贯构成一次 throw_gesture。
_THROW_PREP_ACTION = "rps-ready"

# 取消语义分级：advance 类多阶段运动（抓取/摇骰）取消＝急停（SIGINT 进程组，
# demo 没有逐命令 stop，这是唯一的运动停止手段）；action/query_pose 类短安全
# 命令（归位/手势/探针，1-3s）取消时只放弃等待、让 demo 把动作执行完——打断
# 一个正在执行的归位，换来的是 8-13s 复活链加新会话目录（2026-09-25 常驻
# 重启 bug：三个会话全死在 KeyboardInterrupt + active_action=home）。
# 超时仍一律急停：那是解锁卡死命令、释放臂锁的唯一逃生门。
_CANCEL_INTERRUPT_COMMANDS = {"grasp_cup", "shake_dice"}

_LOG_TAIL_LINES = 15


def _log(line: str) -> None:
    print(f"[robot] {line}", flush=True)


class _Pending:
    """One in-flight command awaiting its id-correlated terminal event."""

    def __init__(self, command_id: str, on_event: RobotEventFn) -> None:
        self.command_id = command_id
        self.on_event = on_event
        self.done = threading.Event()
        self.outcome: dict[str, Any] | None = None

    def resolve(self, outcome: dict[str, Any]) -> None:
        self.outcome = outcome
        self.done.set()


class _Resident:
    """One resident ``run.sh control --execute`` subprocess plus its reader."""

    def __init__(
        self,
        demo_root: Path,
        env: dict[str, str],
        sigint_grace: float,
    ) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = (
            demo_root / "cup_grasp_demo" / "datasets" / f"game_{stamp}_{uuid.uuid4().hex[:8]}"
        )
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.session_dir / "controller.log"
        self._log_file = self.log_path.open("w", encoding="utf-8")
        run_env = dict(env)
        run_env["DICE_RUN"] = str(self.session_dir)
        self.process = subprocess.Popen(
            ["bash", str(demo_root / "run.sh"), "control", "--execute"],
            cwd=str(demo_root),
            env=run_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log_file,
            text=True,
            bufsize=1,
            start_new_session=True,  # cancel must reach the SDK worker children too
        )
        self.sigint_grace = sigint_grace
        self.ready = threading.Event()
        self.recovery_done = threading.Event()
        self.recovery_done.set()
        self.actions: list[str] = []
        self.alive = True
        self._pending: dict[str, _Pending] = {}
        self._pending_lock = threading.Lock()
        self._recovery_id: str | None = None
        self._recovery_on_event: RobotEventFn | None = None
        self.recovery_started = False
        self.recovery_error = ""
        self._reader = threading.Thread(
            target=self._read_loop, name="robot-resident-reader", daemon=True
        )
        self._reader.start()

    # ---- reader --------------------------------------------------------

    def _read_loop(self) -> None:
        try:
            for line in self.process.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    _log(f"non-JSON resident stdout: {line[:200]}")
                    continue
                if isinstance(event, dict):
                    self._dispatch(event)
        except Exception as exc:  # a dead reader must not leak pending commands
            _log(f"resident reader failed: {exc}")
        finally:
            self.alive = False
            self.recovery_done.set()
            # Unblock a ready-wait, then fail anything still in flight.
            self.ready.set()
            with self._pending_lock:
                pendings = list(self._pending.values())
                self._pending.clear()
            for pending in pendings:
                pending.resolve(
                    {
                        "status": "failed",
                        "reason": (
                            f"resident exited unexpectedly (code {self.process.poll()}); "
                            f"log: {self.log_path}"
                        ),
                    }
                )

    def _dispatch(self, event: dict[str, Any]) -> None:
        name = str(event.get("event") or "")
        command_id = event.get("id")
        if name == "ready":
            self.actions = [str(a) for a in (event.get("actions") or [])]
            self.ready.set()
            return
        if name == "run_completed":
            # Auto-reset to idle; the id-matching command_completed follows.
            _log(f"resident run completed (cycle {event.get('cycle')})")
            return
        pending: _Pending | None = None
        if command_id is not None:
            with self._pending_lock:
                pending = self._pending.get(str(command_id))
        if name in ("recovery_started", "recovered"):
            if str(command_id) != self._recovery_id:
                _log(f"ignored {name} for unexpected command id {command_id}")
                return
            if name == "recovery_started":
                self.recovery_started = True
            if name == "recovered":
                self.recovery_done.set()
            self._notify_recovery(
                "RECOVERY",
                "失败自愈，机械臂归位中" if name == "recovery_started" else "机械臂已恢复就位",
            )
            _log(f"resident {name} (recovery)")
            return
        if name == "phase_started":
            if pending is not None:
                pending.on_event(self._progress_event(str(event.get("phase") or "")))
            return
        if name == "action_started":
            if pending is not None:
                pending.on_event(
                    {
                        "event": "robot",
                        "phase": "ACTION",
                        "zh": f"手势 {event.get('name')}",
                    }
                )
            return
        if name in ("command_completed", "action_completed", "actions_reloaded"):
            if pending is not None:
                self._forget_pending(command_id)
                pending.resolve({"status": "completed"})
            return
        if name == "pose":
            # Read-only probe receipt (query_pose): carries the at_home verdict.
            if pending is not None:
                self._forget_pending(command_id)
                pending.resolve(
                    {
                        "status": "completed",
                        "at_home": bool(event.get("at_home")),
                        "joints_rad": list(event.get("joints_rad") or []),
                        "delta_deg": list(event.get("delta_deg") or []),
                    }
                )
            return
        if name == "rejected":
            if pending is not None:
                self._forget_pending(command_id)
                pending.resolve(
                    {
                        "status": "failed",
                        "reason": (
                            f"rejected({event.get('code')}): {event.get('message')}"
                        ),
                    }
                )
            return
        if name == "failed":
            if pending is not None:
                self._recovery_id = str(command_id)
                self._recovery_on_event = pending.on_event
                self.recovery_started = False
                self.recovery_error = ""
                self.recovery_done.clear()
                self._forget_pending(command_id)
                pending.resolve(
                    {
                        "status": "failed",
                        "reason": f"{event.get('phase')}: {event.get('error')}",
                    }
                )
            elif str(command_id) == self._recovery_id and event.get("phase") == "recovery:home":
                reason = str(event.get("error") or "recovery home failed")
                self.recovery_error = reason
                self._notify_recovery("RECOVERY_FAILED", "机械臂自动归位失败", reason=reason)
                _log(f"resident recovery home failed: {reason}")
            return
        if name == "closed":
            if pending is not None:
                self._forget_pending(command_id)
                pending.resolve({"status": "failed", "reason": "resident closed"})
            return
        # status / actions / perception_reset / preview: informational only.

    def _notify_recovery(self, phase: str, zh: str, *, reason: str = "") -> None:
        if self._recovery_on_event is None:
            return
        event = {"event": "robot", "phase": phase, "zh": zh}
        if reason:
            event["reason"] = reason
        try:
            self._recovery_on_event(event)
        except Exception as exc:
            _log(f"recovery event callback failed: {exc}")

    def _forget_pending(self, command_id: Any) -> None:
        """Drop one resolved pending so the map stays bounded."""
        if command_id is None:
            return
        with self._pending_lock:
            self._pending.pop(str(command_id), None)

    @staticmethod
    def _progress_event(phase: str) -> dict[str, Any]:
        zh = _PHASE_LABELS_ZH.get(phase, phase)
        index = _PHASES.index(phase) + 1 if phase in _PHASES else 0
        progress = f"{index}/{len(_PHASES)}" if index else ""
        event: dict[str, Any] = {"event": "robot", "phase": phase, "zh": zh}
        if progress:
            event["progress"] = progress
        return event

    # ---- command plumbing ----------------------------------------------

    def wait_ready(self, timeout: float, is_cancelled: RobotCancelledFn) -> str:
        """Empty string once ready; otherwise a human-readable error."""
        deadline = time.monotonic() + max(0.1, timeout)
        while not self.ready.wait(0.2):
            if not self.alive:
                return (
                    f"resident exited before ready (code {self.process.poll()}): "
                    f"{self.log_tail()}"
                )
            if is_cancelled():
                return "cancelled before ready"
            if time.monotonic() >= deadline:
                return f"resident not ready within {timeout:.0f}s: {self.log_tail()}"
        if not self.alive:
            return f"resident exited before ready: {self.log_tail()}"
        return ""

    def send(self, payload: dict[str, Any], on_event: RobotEventFn) -> _Pending:
        command_id = uuid.uuid4().hex[:12]
        pending = _Pending(command_id, on_event)
        with self._pending_lock:
            self._pending[command_id] = pending
        try:
            assert self.process.stdin is not None
            self.process.stdin.write(json.dumps({**payload, "id": command_id}) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            with self._pending_lock:
                self._pending.pop(command_id, None)
            pending.resolve(
                {"status": "failed", "reason": f"resident stdin broken: {exc}"}
            )
        return pending

    def interrupt(self) -> None:
        """SIGINT the process group, then SIGKILL after the grace period."""
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGINT)
        except (ProcessLookupError, OSError):
            pass
        try:
            self.process.wait(timeout=self.sigint_grace)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    def close(self) -> None:
        """Ask the resident to release SDK/camera and exit (server shutdown)."""
        if self.process.poll() is None:
            try:
                assert self.process.stdin is not None
                self.process.stdin.write(json.dumps({"id": "close-game", "command": "close"}) + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            try:
                self.process.wait(timeout=self.sigint_grace)
            except subprocess.TimeoutExpired:
                self.interrupt()
        try:
            self._log_file.close()
        except OSError:
            pass

    def log_tail(self) -> str:
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            return " | ".join(lines[-_LOG_TAIL_LINES:])[-800:]
        except OSError:
            return "<no log>"


class RobotArmNeroProvider(RobotProvider):
    """Drive the NERO arm through the dice_demo resident controller."""

    id = "robot_arm_nero"
    type = "robot"

    def __init__(self, manifest: dict[str, Any] | None = None) -> None:
        super().__init__(manifest)
        config = json.loads(
            (Path(__file__).parent / "config.json").read_text(encoding="utf-8")
        )
        self._demo_root = (REPO_ROOT / str(config.get("demo_root", "../dice_demo"))).resolve()
        self._python_bin = str(config.get("python_bin", "/usr/bin/python3"))
        self._ready_timeout = float(config.get("ready_timeout_seconds", 60))
        self._timeouts = {
            "grasp_cup": float(config.get("grasp_timeout_seconds", 45)),
            "shake_dice": float(config.get("shake_timeout_seconds", 120)),
            "feedback": float(config.get("action_timeout_seconds", 30)),
            "throw_gesture": float(config.get("action_timeout_seconds", 30)),
            "reset_home": float(config.get("action_timeout_seconds", 30)),
            "query_pose": float(config.get("action_timeout_seconds", 30)),
            "ensure_home": float(config.get("action_timeout_seconds", 30)),
        }
        self._sigint_grace = float(config.get("sigint_grace_seconds", 15))
        # 归位不变量的开机档：服务起来后把臂拉回 home（没接臂的部署关掉，
        # 免得白拉一对常驻进程）。组件 config 改动需重启后端。
        self._home_on_boot = bool(config.get("home_on_boot", True))
        # One arm: commands serialize here, including the ready wait and any
        # auto-retry, so the demo never sees overlapping motion.
        self._arm_lock = threading.Lock()
        self._resident: _Resident | None = None
        self._shutdown = False

    # ---- component lifecycle -------------------------------------------

    @property
    def home_on_boot(self) -> bool:
        return self._home_on_boot

    def health(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "ok": True,
            "ready": True,
            "demo_root": str(self._demo_root),
        }
        checks = {
            "run.sh": self._demo_root / "run.sh",
            "configs/green_cup.json": self._demo_root / "configs" / "green_cup.json",
            "configs/actions/gestures": self._demo_root / "configs" / "actions" / "gestures",
            "configs/actions/home.json": self._demo_root / "configs" / "actions" / "home.json",
            "vendor-site/pyAgxArm": self._demo_root / "vendor-site" / "pyAgxArm",
            "vendor-site/pyrealsense2": self._demo_root / "vendor-site" / "pyrealsense2",
        }
        missing = [name for name, path in checks.items() if not path.exists()]
        if missing:
            payload["ok"] = False
            payload["ready"] = False
            payload["error"] = f"demo deployment incomplete under {self._demo_root}: missing {missing}"
        return payload

    def ensure_started(self) -> None:
        """Spawn the resident early so the rules speech covers its warmup.

        Fire-and-forget by design: only the Popen happens here (fast), the
        6-10s ready wait is paid by whichever command comes first.  Never
        blocks the caller — if a command is in flight the resident is clearly
        up already.
        """
        if not self._arm_lock.acquire(blocking=False):
            return
        try:
            if self._shutdown:
                return
            if (
                self._resident is not None
                and self._resident.alive
                and self._resident.process.poll() is None
            ):
                return  # already warming or ready
            self._resident = _Resident(
                self._demo_root, self._build_env(), self._sigint_grace,
            )
            _log(f"resident prewarmed (session {self._resident.session_dir.name})")
        except Exception as exc:
            _log(f"resident prewarm failed: {exc}")
        finally:
            self._arm_lock.release()

    def shutdown(self) -> None:
        """Release the arm's SDK/camera; called on server shutdown."""
        self._shutdown = True
        with self._arm_lock:
            resident, self._resident = self._resident, None
        if resident is not None:
            resident.close()
            _log(f"resident closed (session {resident.session_dir.name})")

    # ---- robot commands --------------------------------------------------

    def grasp_cup(
        self,
        *,
        on_event: RobotEventFn,
        is_cancelled: RobotCancelledFn,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        return self._run_command(
            "grasp_cup",
            {"command": "advance", "until": "LIFT"},
            timeout_seconds,
            on_event,
            is_cancelled,
        )

    def shake_dice(
        self,
        *,
        on_event: RobotEventFn,
        is_cancelled: RobotCancelledFn,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        return self._run_command(
            "shake_dice",
            {"command": "advance", "until": "RETURN_HOME"},
            timeout_seconds,
            on_event,
            is_cancelled,
        )

    def feedback(
        self,
        kind: str,
        *,
        on_event: RobotEventFn,
        is_cancelled: RobotCancelledFn,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        action = _FEEDBACK_ACTIONS.get(kind)
        if action is None:
            return {"status": "failed", "reason": f"unknown feedback kind {kind!r}"}
        return self._run_command(
            "feedback",
            {"command": "action", "name": action},
            timeout_seconds,
            on_event,
            is_cancelled,
        )

    def throw_gesture(
        self,
        gesture: str | None = None,
        *,
        on_event: RobotEventFn,
        is_cancelled: RobotCancelledFn,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        # gesture=None → 臂侧随机选拳（play 提前出拳时拳形尚不可知）。
        # 单一事实源是 outcome["gesture"]：实际执行的那份拳形回报给消费方，
        # 判定用的拳 == 观众看到的拳。
        if gesture is None:
            gesture = random.choice(sorted(_THROW_ACTIONS))
        action = _THROW_ACTIONS.get(gesture)
        if action is None:
            return {"status": "failed", "reason": f"unknown throw gesture {gesture!r}"}
        # 连贯两段链：rps-ready 预备（手前伸）→ 拳形。整链一把臂锁（镜像
        # reset_home 的持锁直调，两段间不松锁——不与终局归位 watcher 交错）；
        # 预备段失败即整链失败，第二段的 action_completed 是整链完成判据。
        with self._arm_lock:
            if self._shutdown:
                return {"status": "failed", "reason": "provider is shutting down"}
            prep = self._run_command_locked(
                "throw_gesture",
                {"command": "action", "name": _THROW_PREP_ACTION},
                timeout_seconds,
                on_event,
                is_cancelled,
            )
            if not (isinstance(prep, dict) and prep.get("status") == "completed"):
                if isinstance(prep, dict):
                    prep["reason"] = (
                        f"prep action {_THROW_PREP_ACTION!r} failed: {prep.get('reason') or '动作未完成'}"
                    )
                    return prep
                return {"status": "failed", "reason": "prep failed"}
            outcome = self._run_command_locked(
                "throw_gesture",
                {"command": "action", "name": action},
                timeout_seconds,
                on_event,
                is_cancelled,
            )
        if isinstance(outcome, dict):
            outcome.setdefault("gesture", gesture)
        return outcome

    def reset_home(
        self,
        *,
        on_event: RobotEventFn,
        is_cancelled: RobotCancelledFn,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        # The parking invariant: any exit from the game flow must home the
        # arm, including right after a failure (which exits the demo). That
        # means reset_home is the one command allowed to revive a dead
        # resident — _run_command_locked's _ensure_running spawns a fresh one.
        with self._arm_lock:
            if self._shutdown:
                return {"status": "failed", "reason": "provider is shutting down"}
            return self._run_command_locked(
                "reset_home",
                {"command": "action", "name": "home"},
                timeout_seconds,
                on_event,
                is_cancelled,
            )

    def query_pose(
        self,
        *,
        on_event: RobotEventFn,
        is_cancelled: RobotCancelledFn,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Read-only at-home probe for the patrol loop — never revives.

        The mirror image of reset_home: a dead resident means "cannot judge,
        skip this round" (the patrol must not become a second revive source);
        a live one answers ``{"status": "completed", "at_home": bool}``.
        Probe failures surface as failed/rejected outcomes, never a spawn.
        """
        with self._arm_lock:
            if self._shutdown:
                return {"status": "failed", "reason": "provider is shutting down"}
            # 三重验活：failed 事件先于进程退出落地时（demo 自愈失败走
            # legacy exit 的窗口），alive 标志和句柄都还没更新——poll 现场
            # 问操作系统，绝不在这个窗口里 spawn 复活。
            if (
                self._resident is None
                or not self._resident.alive
                or self._resident.process.poll() is not None
            ):
                return {"status": "skipped", "reason": "resident not running"}
            return self._run_command_locked(
                "query_pose",
                {"command": "query_pose"},
                timeout_seconds,
                on_event,
                is_cancelled,
            )

    def ensure_home(
        self,
        *,
        on_event: RobotEventFn,
        is_cancelled: RobotCancelledFn,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Checked homing: probe first (≈60ms cached), home only if off-home.

        The 2026-09-25 parking-insurance command: an at-home arm answers
        completed with zero motion (no redundant ~2s homing gesture), while
        anything else — off-home, probe failure, or a dead resident — falls
        through to the full reset_home chain (which may revive).  Err on the
        side of homing: a probe that cannot judge must not skip the motion.
        The probe cannot see fingers (snapshot carries joints only), which is
        exactly why this stays a separate command: reset_home stays
        unconditional for the "must open the hand" paths (dice failure page).
        """
        with self._arm_lock:
            if self._shutdown:
                return {"status": "failed", "reason": "provider is shutting down"}
            resident = self._resident
            if resident is not None and not resident.recovery_done.is_set():
                deadline = time.monotonic() + float(
                    timeout_seconds or self._timeouts["ensure_home"]
                )
                error = self._await_recovery(resident, is_cancelled, deadline)
                if error:
                    return {"status": "failed", "reason": error}
            if (
                self._resident is not None
                and self._resident.alive
                and self._resident.process.poll() is None
            ):
                # The probe must not eat the homing budget: cap it short so a
                # hung snapshot still leaves the full timeout for the motion.
                probe_budget = min(float(timeout_seconds or 30.0), 10.0)
                probe = self._run_command_locked(
                    "query_pose",
                    {"command": "query_pose"},
                    probe_budget,
                    on_event,
                    is_cancelled,
                )
                if (
                    isinstance(probe, dict)
                    and probe.get("status") == "completed"
                    and probe.get("at_home")
                ):
                    return {"status": "completed", "at_home": True}
            return self._run_command_locked(
                "reset_home",
                {"command": "action", "name": "home"},
                timeout_seconds,
                on_event,
                is_cancelled,
            )

    # ---- command engine (arm lock held) ---------------------------------

    def _run_command(
        self,
        name: str,
        payload: dict[str, Any],
        timeout_seconds: float | None,
        on_event: RobotEventFn,
        is_cancelled: RobotCancelledFn,
    ) -> dict[str, Any]:
        with self._arm_lock:
            if self._shutdown:
                return {"status": "failed", "reason": "provider is shutting down"}
            return self._run_command_locked(
                name, payload, timeout_seconds, on_event, is_cancelled
            )

    def _run_command_locked(
        self,
        name: str,
        payload: dict[str, Any],
        timeout_seconds: float | None,
        on_event: RobotEventFn,
        is_cancelled: RobotCancelledFn,
    ) -> dict[str, Any]:
        started = time.monotonic()
        deadline = started + float(timeout_seconds or self._timeouts.get(name, 60))
        attempt = 0
        auto_retried = False
        while True:
            attempt += 1
            resident, error = self._ensure_running(is_cancelled, deadline)
            if resident is None:
                return {"status": "failed", "reason": error}
            if is_cancelled():
                return {"status": "failed", "reason": "cancelled before command dispatch"}
            if payload.get("command") == "action":
                # Static gestures live in configs/actions/gestures/ and the
                # resident hot-reloads that table on demand (demo 2026-09-23).
                # A fire-and-forget reload ahead of every static action makes
                # a recipe edit apply to the very next gesture — the demo
                # serializes stdin commands, so the action runs behind the
                # reload, and a failed reload keeps the previous table.
                resident.send({"command": "reload"}, on_event)
            pending = resident.send(payload, on_event)
            outcome = self._wait_pending(
                resident,
                pending,
                deadline,
                is_cancelled,
                interrupt_on_cancel=name in _CANCEL_INTERRUPT_COMMANDS,
            )
            elapsed = round(time.monotonic() - started, 2)
            if outcome is None:
                reason = (
                    "command interrupted (cancelled)"
                    if is_cancelled()
                    else "command timed out"
                )
                return {"status": "failed", "reason": reason, "elapsed": elapsed}
            if outcome.get("status") == "completed":
                outcome.setdefault("elapsed", elapsed)
                if auto_retried:
                    outcome["auto_retried"] = True
                return outcome
            reason = str(outcome.get("reason") or "")
            # The finger-position check trips occasionally right after a
            # shake; one automatic rerun recovers it on the real machine.
            if (
                "Hand start" in reason
                and attempt == 1
                and not is_cancelled()
                and time.monotonic() < deadline
            ):
                while not resident.recovery_done.wait(0.1):
                    if is_cancelled():
                        return {"status": "failed", "reason": "command interrupted (cancelled)", "elapsed": elapsed}
                    if time.monotonic() >= deadline:
                        return {"status": "failed", "reason": f"{reason}; resident recovery still in progress", "elapsed": elapsed}
                if is_cancelled():
                    return {"status": "failed", "reason": f"{reason}; command interrupted (cancelled)", "elapsed": elapsed}
                if resident.recovery_error or (resident.recovery_started and not resident.alive):
                    detail = resident.recovery_error or "resident exited during recovery"
                    return {"status": "failed", "reason": f"{reason}; recovery failed: {detail}", "elapsed": elapsed}
                if time.monotonic() >= deadline:
                    return {"status": "failed", "reason": f"{reason}; retry budget exhausted after recovery", "elapsed": elapsed}
                auto_retried = True
                _log(f"{name}: hand-start check tripped; auto-retrying once ({reason})")
                on_event(
                    {"event": "robot", "phase": "AUTO_RETRY", "zh": "手指校验未过，自动重试"}
                )
                # The demo may be homing after failed; _ensure_running waits
                # for recovered or process exit before retrying.
                continue
            if outcome.get("status") != "completed" and self._resident is not None:
                # demo 2026-09-25 起失败自愈（recovery 归位后继续常驻）：
                # failed ≠ 进程退出。只在进程**真的死了**（自愈失败走
                # legacy exit，或启动即崩）时才 reap；活着的常驻正带着
                # RECOVERY 进度，绝不能 wait+interrupt 打断它的自愈归位。
                # 半死不活的句柄由下一次 _ensure_running 的验活兜底清理。
                resident = self._resident
                if resident.process.poll() is not None:
                    try:
                        resident.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        resident.interrupt()
                    self._resident = None
            outcome.setdefault("elapsed", elapsed)
            return outcome

    def _wait_pending(
        self,
        resident: _Resident,
        pending: _Pending,
        deadline: float,
        is_cancelled: RobotCancelledFn,
        *,
        interrupt_on_cancel: bool = True,
    ) -> dict[str, Any] | None:
        """None when cancelled/timed out. Timeout always interrupts the
        resident (the only escape hatch for a hung command); cancel only
        does when ``interrupt_on_cancel`` — the short safe actions let the
        demo finish the motion instead of dying mid-homing."""
        while not pending.done.wait(0.1):
            if is_cancelled():
                if interrupt_on_cancel:
                    resident.interrupt()
                return None
            if time.monotonic() >= deadline:
                resident.interrupt()
                return None
        return pending.outcome or {"status": "failed", "reason": "no outcome"}

    def _ensure_running(
        self, is_cancelled: RobotCancelledFn, deadline: float
    ) -> tuple[_Resident | None, str]:
        """A live, ready resident — spawning a fresh one when necessary."""
        resident = self._resident
        if resident is not None:
            error = self._await_recovery(resident, is_cancelled, deadline)
            if error:
                return None, error
        if resident is not None and (
            not resident.alive or resident.process.poll() is not None
        ):
            # Dead or exiting (the reader's EOF may lag the process exit).
            self._resident = None
        if self._resident is not None and self._resident.ready.is_set():
            return self._resident, ""
        if self._resident is None:
            if is_cancelled():
                return None, "cancelled"
            self._resident = _Resident(
                self._demo_root, self._build_env(), self._sigint_grace,
            )
            _log(f"resident spawned (session {self._resident.session_dir.name})")
        # Either freshly spawned or still warming from a prewarm.
        remaining = max(0.1, deadline - time.monotonic())
        timeout = min(self._ready_timeout, remaining)
        resident = self._resident
        error = resident.wait_ready(timeout, is_cancelled)
        if error:
            tail = resident.log_tail()
            resident.interrupt()
            self._resident = None
            return None, f"resident not ready: {error}" + (f" | {tail}" if tail else "")
        return resident, ""

    @staticmethod
    def _await_recovery(
        resident: _Resident, is_cancelled: RobotCancelledFn, deadline: float
    ) -> str:
        while not resident.recovery_done.wait(0.1):
            if is_cancelled():
                return "cancelled while resident is recovering"
            if time.monotonic() >= deadline:
                return "resident recovery still in progress"
        if is_cancelled():
            return "cancelled while resident is recovering"
        if time.monotonic() >= deadline:
            return "command timeout elapsed while resident was recovering"
        return ""

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["DICE_CONFIG"] = str(self._demo_root / "configs" / "green_cup.json")
        env["DICE_VISION_PYTHON"] = self._python_bin
        env["DICE_SDK_PYTHON"] = self._python_bin
        env["NERO_SDK_DIR"] = str(self._demo_root / "vendor-site" / "pyAgxArm")
        pythonpath = [
            str(self._demo_root / "vendor-site"),
            str(self._demo_root / "vendor-site-deps"),
        ]
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = ":".join([*pythonpath, existing]) if existing else ":".join(pythonpath)
        return env
