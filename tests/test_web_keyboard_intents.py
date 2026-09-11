"""按键→意图映射的回归测试（用 Node 跑真实前端模块）。

字符串断言只能证明代码里有某个词，证明不了"按下这个键会发生什么"。
web/games/dice.js 是纯 ES 模块、依赖全部由 register(engine) 注入，所以
tests/js/dice_keyboard_intents.mjs 可以在 Node 里把 onKey 真正跑起来，
并断言按 Esc 与点击屏幕按钮产生完全相同的意图。
"""

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tests/js/dice_keyboard_intents.mjs"
DICE_JS = ROOT / "web/games/dice.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 Node 才能驱动前端模块")
def test_dice_keyboard_and_buttons_submit_the_same_intents():
    result = subprocess.run(
        ["node", str(HARNESS), str(DICE_JS)],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
