"""YOLOv8 dice-adjudicator adapter around the ``yolov8_camera`` process.

This adapter owns camera capture, OpenCL preprocessing, SpaceMIT EP inference,
stable-frame filtering, side scoring, winner calculation, and LLM
verification. It deliberately returns an adjudication result, not generic
object coordinates; a future YOLO-based localizer belongs behind the separate
``VisionLocalizerProvider`` interface.
"""
from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from core.vision import VisionAdjudicatorProvider
from core.env import load_board_env
from core.errors import VisionError

load_board_env()

ROOT = Path(__file__).resolve().parents[3]  # repo root (main/)
VISION_ROOT = ROOT / "vision" / "yolov8_objdetect"
DEFAULT_BINARY = VISION_ROOT / "build" / "yolov8_camera"
_protocol_cache: tuple[str, int, bool] | None = None


def yolo_binary() -> Path:
    return Path(os.environ.get("DICE_YOLO_BINARY", str(DEFAULT_BINARY))).resolve()


def supports_structured_events(binary: Path | None = None) -> bool:
    """Probe ``--event-fd`` once per binary mtime."""
    global _protocol_cache
    binary = binary or yolo_binary()
    try:
        mtime_ns = binary.stat().st_mtime_ns
    except OSError:
        return False
    key = str(binary)
    if _protocol_cache and _protocol_cache[:2] == (key, mtime_ns):
        return _protocol_cache[2]
    try:
        help_output = subprocess.run(
            [str(binary), "--help"], cwd=VISION_ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=2,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        help_output = ""
    supported = "--event-fd" in help_output
    _protocol_cache = (key, mtime_ns, supported)
    return supported


def configured_llm() -> bool:
    if os.environ.get("DICE_LLM_API_KEY", "").strip():
        return True
    try:
        config = json.loads((VISION_ROOT / "config.json").read_text(encoding="utf-8"))
        return bool(config.get("llm", {}).get("api_key", "").strip())
    except (OSError, ValueError, AttributeError):
        return False


def _terminate(process: subprocess.Popen[str]) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass


def _parse_result_line(line: str) -> dict[str, Any] | None:
    raw = line.strip()
    if raw.startswith("[RESULT] "):
        raw = raw[9:]
    if not raw.startswith("{"):
        return None
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _legacy_phase_of(line: str) -> str | None:
    """Map old human-readable output to coarse progress during upgrades only."""
    if line.startswith("[YOLO]") or "OpenCL GPU" in line or "Model loaded" in line:
        return "detecting"
    if line.startswith("[LLM]") or "calling LLM" in line:
        return "verifying"
    return None


def _consume_legacy_log_line(
    line: str,
    on_log: Callable[[str], None],
    on_event: Callable[[dict[str, Any]], None],
) -> dict[str, Any] | None:
    """Parse only the old explicit ``[RESULT]`` compatibility envelope.

    Arbitrary JSON-looking diagnostic lines stay logs. This prevents stdout
    from silently becoming a second business-event protocol.
    """
    if not line.startswith("[RESULT] "):
        on_log(line)
        return None
    parsed = _parse_result_line(line)
    if parsed is None:
        on_log(line)
        return None
    event = dict(parsed)
    event.setdefault("event", "result")
    on_event(event)
    return event


def run_adjudication(
    on_log: Callable[[str], None],
    on_event: Callable[[dict[str, Any]], None],
    is_cancelled: Callable[[], bool],
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run ``yolov8_camera`` and return its verified adjudication result.

    Rebuilt binaries send JSONL over the dedicated ``--event-fd`` pipe while
    stdout/stderr remains diagnostic text. During a rolling upgrade, an older
    binary may still return the explicit ``[RESULT]`` stdout envelope; only
    that tagged line is parsed as a compatibility result.
    """
    binary = yolo_binary()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise VisionError(f"YOLOv8 executable not found or not executable: {binary}")
    if not configured_llm():
        raise VisionError("LLM is not configured; set DICE_LLM_API_KEY or .dice-arena.env")

    # The rebuilt binary writes JSONL events to a dedicated inherited pipe.
    # Until that binary is deployed, use the explicit legacy result envelope.
    structured_events = supports_structured_events(binary)
    event_read = -1
    event_write = -1
    if structured_events:
        event_read, event_write = os.pipe()

    command = [
        str(binary), "--config", "config.json", "--no-display",
        "--rejudge-on-change",
    ]
    if structured_events:
        command += ["--event-fd", str(event_write)]

    env = os.environ.copy()
    process: subprocess.Popen[str] | None = None
    event_stream = None
    last_result: dict[str, Any] | None = None
    try:
        process = subprocess.Popen(
            command,
            cwd=VISION_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
            pass_fds=(event_write,) if structured_events else (),
        )
        if event_write >= 0:
            os.close(event_write)
            event_write = -1
        if structured_events:
            event_stream = os.fdopen(event_read, "r", encoding="utf-8", buffering=1)
            event_read = -1
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "log")
        if event_stream is not None:
            selector.register(event_stream, selectors.EVENT_READ, "event")
        deadline = time.monotonic() + timeout_seconds

        def consume_event(raw_line: str) -> None:
            nonlocal last_result
            try:
                value = json.loads(raw_line)
            except (ValueError, TypeError):
                on_log(f"[vision-event] invalid JSON: {raw_line.rstrip()}")
                return
            if not isinstance(value, dict):
                on_log(f"[vision-event] ignored non-object event: {raw_line.rstrip()}")
                return
            on_event(value)
            if value.get("event") == "result" and value.get("verified"):
                last_result = value

        legacy_phase: str | None = None

        def consume_log(line: str) -> dict[str, Any] | None:
            nonlocal legacy_phase
            if structured_events:
                # In the current protocol stdout/stderr is diagnostics only.
                on_log(line)
                return None
            phase = _legacy_phase_of(line)
            if phase and phase != legacy_phase:
                legacy_phase = phase
                on_event({"event": "phase", "phase": phase, "source": "legacy-log"})
            return _consume_legacy_log_line(line, on_log, on_event)

        while True:
            if is_cancelled():
                _terminate(process)
                raise VisionError("analysis cancelled")
            if time.monotonic() > deadline:
                _terminate(process)
                raise VisionError(f"YOLOv8 analysis timed out after {timeout_seconds}s")

            for key, _ in selector.select(timeout=0.05):
                stream = key.fileobj
                line = stream.readline()
                if not line:
                    try:
                        selector.unregister(stream)
                    except Exception:
                        pass
                    continue
                if key.data == "event":
                    consume_event(line)
                    if last_result is not None:
                        selector.close()
                        _terminate(process)
                        return last_result
                else:
                    parsed = consume_log(line)
                    if parsed is not None and parsed.get("verified"):
                        last_result = parsed
                        selector.close()
                        _terminate(process)
                        return last_result

            if process.poll() is not None:
                for line in process.stdout:
                    parsed = consume_log(line)
                    if parsed is not None and parsed.get("verified"):
                        last_result = parsed
                if event_stream is not None:
                    for line in event_stream:
                        consume_event(line)
                break

        selector.close()
        return_code = process.wait(timeout=5)
        if last_result:
            return last_result
        if return_code != 0:
            raise VisionError(f"YOLOv8 process exited with code {return_code}")
        raise VisionError("YOLOv8 stopped before an LLM-verified result was produced")

    finally:
        if process is not None and process.poll() is None:
            _terminate(process)
        if event_write >= 0:
            os.close(event_write)
        if event_read >= 0:
            os.close(event_read)
        if event_stream is not None:
            try:
                event_stream.close()
            except OSError:
                pass


class DiceYoloAdjudicator(VisionAdjudicatorProvider):
    id = "vision_yolo"
    type = "vision"
    role = "adjudicator"
    name = "YOLOv8 Dice Adjudicator"
    version = "1.1"

    def health(self) -> dict[str, Any]:
        binary = yolo_binary()
        return {
            "id": self.id,
            "type": self.type,
            "role": self.role,
            "ok": binary.is_file() and os.access(binary, os.X_OK) and configured_llm(),
            "ready": binary.is_file() and os.access(binary, os.X_OK),
            "binary": str(binary),
            "llm_configured": configured_llm(),
            "config": str(VISION_ROOT / "config.json"),
            "protocol": (
                "jsonl-events-v1" if supports_structured_events(binary)
                else "legacy-result-line"
            ),
        }

    def adjudicate(
        self,
        *,
        on_log: Callable[[str], None],
        on_event: Callable[[dict[str, Any]], None],
        is_cancelled: Callable[[], bool],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return run_adjudication(
            on_log=on_log,
            on_event=on_event,
            is_cancelled=is_cancelled,
            timeout_seconds=timeout_seconds,
        )
