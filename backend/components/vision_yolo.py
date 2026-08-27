"""YOLOv8 vision component: wraps the ``yolov8_camera`` C++ subprocess.

The component owns camera capture, OpenCL preprocessing, SpaceMIT EP
inference, stable-frame filtering, and LLM verification by shelling out to the
checked-in ``yolov8_camera`` binary. It turns frames into a verified dice
result and knows nothing about game rules.
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

from core.components import Component
from core.env import load_board_env
from core.errors import VisionError

load_board_env()

ROOT = Path(__file__).resolve().parents[2]  # repo root (main/)
VISION_ROOT = ROOT / "vision" / "yolov8_objdetect"
DEFAULT_BINARY = VISION_ROOT / "build" / "yolov8_camera"


def yolo_binary() -> Path:
    return Path(os.environ.get("DICE_YOLO_BINARY", str(DEFAULT_BINARY))).resolve()


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


def run_analysis(
    result_path: Path,
    on_log: Callable[[str], None],
    is_cancelled: Callable[[], bool],
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run ``yolov8_camera`` and return its verified result dict.

    ``on_log`` receives each stdout line as it arrives (the job layer maps
    lines to phases). ``is_cancelled`` lets the job layer abort the subprocess.
    Raises :class:`VisionError` when no verified result can be produced.
    """
    binary = yolo_binary()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise VisionError(f"YOLOv8 executable not found or not executable: {binary}")
    if not configured_llm():
        raise VisionError("LLM is not configured; set DICE_LLM_API_KEY or .dice-arena.env")

    command = [
        str(binary),
        "--config", "config.json",
        "--no-display",
        "--rejudge-on-change",
        "--require-llm",
        "--result-file", str(result_path),
        "--exit-on-result",
    ]
    env = os.environ.copy()
    # Keep the secret only in this child process. The browser/API response never
    # contains it, and the C++ verifier removes it before invoking curl.
    process = subprocess.Popen(
        command,
        cwd=VISION_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + timeout_seconds
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        while True:
            if is_cancelled():
                _terminate(process)
                raise VisionError("analysis cancelled")
            if time.monotonic() > deadline:
                _terminate(process)
                raise VisionError(f"YOLOv8 analysis timed out after {timeout_seconds}s")
            events = selector.select(timeout=0.5)
            for key, _ in events:
                line = key.fileobj.readline()
                if line:
                    on_log(line)
            if process.poll() is not None:
                for line in process.stdout:
                    on_log(line)
                break
        selector.close()
        return_code = process.wait(timeout=5)
        if result_path.is_file():
            try:
                return json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise VisionError(f"YOLOv8 wrote an invalid result: {exc}") from exc
        if return_code != 0:
            raise VisionError(f"YOLOv8 process exited with code {return_code}")
        raise VisionError("YOLOv8 stopped before an LLM-verified result was produced")
    finally:
        try:
            result_path.unlink(missing_ok=True)
        except OSError:
            pass


class VisionYolo(Component):
    id = "vision_yolo"
    type = "vision"

    def health(self) -> dict[str, Any]:
        binary = yolo_binary()
        return {
            "id": self.id,
            "type": self.type,
            "binary": str(binary),
            "ready": binary.is_file() and os.access(binary, os.X_OK),
            "llm_configured": configured_llm(),
            "config": str(VISION_ROOT / "config.json"),
        }
