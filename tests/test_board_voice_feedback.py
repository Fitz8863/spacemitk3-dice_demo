"""板端键盘合成事件的语音浮条分流回归（Node 驱动真实 app.js）。

「听到「board」，已生效」误弹（2026-10-10）：板端按键被包装成语音总线
事件，前端把所有总线事件都当语音渲染。修法=合成事件带 source:'board'，
前端对 board 来源只做动作、不弹浮条。服务端 source 字段的契约在
tests/test_board_input.py；本壳跑 JS 行为回归。
"""

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tests/js/board_source_feedback_regressions.mjs"


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 Node 才能驱动前端模块")
def test_board_sourced_bus_events_do_not_show_voice_feedback():
    result = subprocess.run(
        ["node", str(HARNESS), str(ROOT)],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
