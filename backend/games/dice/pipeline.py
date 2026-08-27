"""Dice-game backend pipeline: one YOLOv8 + LLM verified analysis.

This is the orchestration entrypoint the job layer calls for ``/api/analyze``.
Today it is a single vision step; future games (or a robot-arm dice game)
declare richer multi-component sequences here while keeping components generic.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Callable

from components.vision_yolo import run_analysis

GAME_ID = "dice"


def run(
    on_log: Callable[[str], None],
    is_cancelled: Callable[[], bool],
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run a single verified dice analysis and return its result dict."""
    result_path = Path("/tmp") / f"dice-arena-{uuid.uuid4().hex}.json"
    return run_analysis(
        result_path=result_path,
        on_log=on_log,
        is_cancelled=is_cancelled,
        timeout_seconds=timeout_seconds,
    )
