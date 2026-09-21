"""Rock-paper-scissors adjudication pipeline（真实视觉版）.

玩家手势来自视觉：``observe()`` 拿 v10 runtime 的稳定观测（类别已在 C++
折叠成 Rock/Paper/Scissors），最高置信度的检测就是玩家的出拳。agent 手势
仍是程序随机——这一处**就是将来给机械臂下发指令的位置**（臂出什么手势是
程序自己定的，视觉不需要识别它；ROI 已把臂所在半幅确定性排除）。

胜负不走 provider 的双侧规则（``categorical_relation`` 要求两侧都有值，
而 agent 侧根本不进画面），比较与投影由本模块调用
``games/rps/result.py`` 完成——那是这个游戏的常驻规则层。
"""
from __future__ import annotations

import random
import time
import uuid
from typing import Any, Callable, Mapping

from core.vision import VisionAdjudicationRequest
from core.games import resolve_provider_id
from games.rps.result import GESTURES, project

# 折叠后的游戏标签（C++ rps_map 的键，按 label 字符串映射，不依赖 id 顺序）
# → result.py 的中文手势词（直接进台词占位符与前端展示）。
_LABEL_TO_GESTURE = {
    "Rock": "石头",
    "Paper": "布",
    "Scissors": "剪刀",
}


def run(
    on_log: Callable[[str], None],
    is_cancelled: Callable[[], bool],
    timeout_seconds: float,
    *,
    components: Any,
    manifest: Mapping[str, Any],
    on_event: Callable[[Mapping[str, Any]], None],
) -> dict[str, Any]:
    game_id = str(manifest.get("id", "rps"))
    profile = manifest.get("vision_profile")
    if not isinstance(profile, Mapping):
        raise RuntimeError(f"vision profile is required for {game_id} game")
    provider_id = resolve_provider_id(manifest, "vision_adjudicator", "vision_yolov8_objdetect")
    adjudicator = components.require(
        provider_id, expected_type="vision", expected_role="adjudicator"
    )
    observe = getattr(adjudicator, "observe", None)
    if not callable(observe):
        raise RuntimeError(f"vision adjudicator {provider_id} does not implement observe()")
    request = VisionAdjudicationRequest(
        game_id=game_id,
        profile=profile,
        request_id=uuid.uuid4().hex,
        timeout_seconds=timeout_seconds,
    )
    outcome = observe(
        request,
        on_log=on_log,
        on_event=on_event,
        is_cancelled=is_cancelled,
        timeout_seconds=timeout_seconds,
    )
    # 无稳定观测（超时/无手）：provider 的诊断契约原样上抛——状态机据此
    # 走 adjudication.diagnosis → analysis_failed（重试路径已有）。
    if outcome.get("diagnosed"):
        return outcome

    observations = outcome.get("observations") or {}
    if not observations:
        raise RuntimeError("vision observation round ended without evidence")
    # 单视图游戏：取默认视图（多视图时字典序第一个，与 provider 的
    # ordered 语义一致）。
    view_id = sorted(observations)[0] if "default" not in observations else "default"
    observation = observations[view_id]
    detections = observation.get("detections")
    detections = [d for d in detections if isinstance(d, Mapping)] if isinstance(detections, list) else []
    if not detections:
        # 稳定观测按契约非空（C++ 只有非空多重集才计稳定帧）；防御分支。
        raise RuntimeError("stable observation carried no detections")
    best = max(detections, key=lambda d: float(d.get("confidence", 0.0) or 0.0))
    label = str(best.get("label", ""))
    player_gesture = _LABEL_TO_GESTURE.get(label)
    if player_gesture is None:
        raise RuntimeError(
            f"vision observation carried label {label!r} outside the rps fold table"
        )
    if is_cancelled():
        raise RuntimeError("cancelled")

    on_event({"event": "phase", "phase": "verifying", "llm": False})
    # 机械臂手势 = 程序随机（将来在此处向臂下发指令，接口形状不变）。
    agent_gesture = random.choice(GESTURES)
    result = project(player_gesture, agent_gesture, manifest["participants"], source="yolo_only")
    on_event({"event": "result", **result})
    # post-result hold 与 dice 同款：结果已出、推流保温，倒计时归展示层。
    hold = float((profile.get("lifecycle") or {}).get("post_result_hold_seconds", 0) or 0)
    if hold > 0:
        end = time.monotonic() + hold
        while time.monotonic() < end:
            if is_cancelled():
                raise RuntimeError("cancelled")
            remaining = max(0, end - time.monotonic())
            on_event({"event": "phase", "phase": "holding", "remaining_ms": int(remaining * 1000)})
            time.sleep(min(0.25, remaining))
    return result
