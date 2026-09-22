"""NERO robot-arm provider over the dice_demo resident JSONL controller.

Spawns one long-lived ``bash run.sh control --execute`` subprocess per board
session (contract: dice_demo ``docs/INTEGRATION.md`` §2) and drives it with
line-JSON commands.  The resident pre-connects SDK/CAN, prewarms the camera
and loads the model once, so in-game commands run at full speed; it stays
alive across rounds and is only replaced when a phase failure kills it (the
demo exits on ``failed``) or the round is cancelled (SIGINT, exit 130).

Command mapping:

* ``grasp_cup``  → ``advance until GRIP``      (HOME→CAPTURE→PLAN→APPROACH→GRIP; no shaking)
* ``shake_dice`` → ``advance until RETURN_HOME`` (LIFT→SHAKE→LOWER→OPEN→RETURN_HOME)
* ``feedback``   → ``action name=yeah|thumbs-up|tie``
* ``reset_home`` → ``action name=home`` — best-effort, never revives a dead resident

Completion is judged strictly by id-correlated events (``command_completed`` /
``action_completed`` / ``rejected`` / ``failed``); progress phases are relayed
to the round stream as ``{"event": "robot", ...}``.  The one known transient
hardware failure (``Hand start`` finger-position check after a shake) is
retried once automatically, which real-machine testing showed recovers it.
"""
from __future__ import annotations

import json
import os
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
        self.actions: list[str] = []
        self.alive = True
        self._pending: dict[str, _Pending] = {}
        self._pending_lock = threading.Lock()
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
        if name in ("command_completed", "action_completed"):
            if pending is not None:
                pending.resolve({"status": "completed"})
            return
        if name == "rejected":
            if pending is not None:
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
                pending.resolve(
                    {
                        "status": "failed",
                        "reason": f"{event.get('phase')}: {event.get('error')}",
                        # The demo exits after a phase failure by contract.
                        "process_exiting": True,
                    }
                )
            return
        if name == "closed":
            if pending is not None:
                pending.resolve({"status": "failed", "reason": "resident closed"})
            return
        # status / actions / perception_reset / preview: informational only.

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
            "reset_home": float(config.get("action_timeout_seconds", 30)),
        }
        self._sigint_grace = float(config.get("sigint_grace_seconds", 15))
        # One arm: commands serialize here, including the ready wait and any
        # auto-retry, so the demo never sees overlapping motion.
        self._arm_lock = threading.Lock()
        self._resident: _Resident | None = None
        self._shutdown = False

    # ---- component lifecycle -------------------------------------------

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
            if self._resident is not None and self._resident.alive:
                return  # already warming or ready
            self._resident = _Resident(
                self._demo_root, self._build_env(), self._sigint_grace
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
            {"command": "advance", "until": "GRIP"},
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

    def reset_home(
        self,
        *,
        on_event: RobotEventFn,
        is_cancelled: RobotCancelledFn,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        # Best-effort by design: the post-round watcher calls this, and it
        # must not pay a full resident restart just to park the arm.
        with self._arm_lock:
            if self._shutdown:
                return {"status": "failed", "reason": "provider is shutting down"}
            if self._resident is None or not self._resident.alive:
                return {"status": "completed", "skipped": True}
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
            pending = resident.send(payload, on_event)
            outcome = self._wait_pending(resident, pending, deadline, is_cancelled)
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
                auto_retried = True
                _log(f"{name}: hand-start check tripped; auto-retrying once ({reason})")
                on_event(
                    {"event": "robot", "phase": "AUTO_RETRY", "zh": "手指校验未过，自动重试"}
                )
                # A phase failure exits the demo; make sure the retry rides a
                # fresh process instead of racing the old one's shutdown.
                if self._resident is not None:
                    self._resident.interrupt()
                    self._resident = None
                continue
            if outcome.get("status") != "completed" and self._resident is not None:
                # A phase failure (or an unexpected exit) kills the process:
                # reap it and drop the resident now, so health checks and
                # best-effort skips see a consistent provider state instead
                # of waiting for the next command to clean up lazily.
                resident = self._resident
                if outcome.get("process_exiting") or resident.process.poll() is not None:
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
    ) -> dict[str, Any] | None:
        """None when cancelled/timed out (the resident was interrupted)."""
        while not pending.done.wait(0.1):
            if is_cancelled():
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
        if resident is not None and (
            not resident.alive or resident.process.poll() is not None
        ):
            # Dead or exiting (the reader's EOF may lag the process exit);
            # never reuse it — a phase failure exits the demo by contract.
            self._resident = None
        if self._resident is not None and self._resident.ready.is_set():
            return self._resident, ""
        if self._resident is None:
            if is_cancelled():
                return None, "cancelled"
            self._resident = _Resident(
                self._demo_root, self._build_env(), self._sigint_grace
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
