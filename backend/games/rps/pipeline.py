"""Rock-paper-scissors adjudication pipeline（视觉接入前的骨架桩）.

手势模型训练好之前，两只手都由桩给出：

* ``agent_gesture`` 在这里随机生成——这**正是将来给机械臂下发指令的位置**
  （臂出什么手势是程序自己定的，视觉不需要识别它）；
* ``player_gesture`` 同样随机桩出——它是**唯一将来要换成视觉检测的件**
  （换成检测调用或 ``run_vision_game`` 证据，``games/rps/result.py``
  的比较与投影原样保留）。

期间发出的分阶段进度事件与真实视觉流的 phase 形状一致（detecting →
verifying → holding），分析页的动画行为在接入前后不变。
"""
from __future__ import annotations

import random
import time
from typing import Any, Callable, Mapping

from games.rps.result import GESTURES, project

# (事件, 停留秒数)——合计约 2 秒，让判定页有节奏感。
_STAGES = (
    ({"event": "phase", "phase": "detecting"}, 0.6),
    ({"event": "phase", "phase": "verifying", "llm": False}, 0.6),
    ({"event": "phase", "phase": "holding", "remaining_ms": 800}, 0.4),
)


def run(
    on_log: Callable[[str], None],
    is_cancelled: Callable[[], bool],
    timeout_seconds: float,
    *,
    components: Any,
    manifest: Mapping[str, Any],
    on_event: Callable[[Mapping[str, Any]], None],
) -> dict[str, Any]:
    _ = on_log, timeout_seconds
    for event, hold_seconds in _STAGES:
        if is_cancelled():
            break
        on_event(event)
        time.sleep(hold_seconds)
    # 机械臂手势 = 程序随机（将来在此处向臂下发指令）；玩家手势 = 桩随机
    # （将来换成视觉检测，唯一待替换件）。
    agent_gesture = random.choice(GESTURES)
    player_gesture = random.choice(GESTURES)
    return project(player_gesture, agent_gesture, manifest["participants"], source="stub")
