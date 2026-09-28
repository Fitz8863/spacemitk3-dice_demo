"""Rock-paper-scissors adjudication pipeline（真实视觉版）.

玩家手势来自视觉：``observe()`` 拿 v10 runtime 的稳定观测（类别已在 C++
折叠成 Rock/Paper/Scissors），最高置信度的检测就是玩家的出拳。agent 手势
由机械臂执行：预备动作完成后，play 状态同步派发口令与 ``throw_gesture``
（臂侧随机选拳），本管线通过回合共享的 ``arm_throw`` 槽消费**臂实际出的
那份拳形**——随机数单一事实源在臂的 outcome 回报，判定的 agent 拳形与观众
看到的拳形恒一致。槽缺失（独立调用/旧装配）时走管线直发兜底：现场随机、
observe 锁定玩家后下发。两条路出拳失败都返回诊断型结果，引擎据此走
analysis_failed（蓝键重试=完整重放口令与出拳）。

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

_ARM_THROW_DIAGNOSIS_REASON = "arm_throw_failed"

_ARM_THROW_PENDING_WAIT_SECONDS = 8.0


def _wait_prefetched_gesture(
    arm_throw: Any,
    *,
    is_cancelled: Callable[[], bool],
    timeout_wait_budget_seconds: float,
) -> tuple[str, str | None]:
    """Wait for the round's prefetched throw slot to settle.

    Returns ``(source, gesture)``:

    * ``("prefetched", gesture)`` — the arm already threw during play; the
      gesture is the one it played.
    * ``("diagnose", None)`` — the slot exists but the throw failed, carried
      no usable gesture, or stayed pending past the wait budget: interrupt
      the round (analysis_failed), the retry replays the chant and throw.
    * ``("legacy", None)`` — no slot at all (standalone call / old wiring):
      the caller falls back to the pipeline-direct throw.

    A pending slot only waits until ``deadline`` so the adjudication budget
    stays in charge.
    """
    if not isinstance(arm_throw, Mapping):
        return "legacy", None
    budget = min(
        _ARM_THROW_PENDING_WAIT_SECONDS, max(0.0, timeout_wait_budget_seconds)
    )
    wait_until = time.monotonic() + budget
    while True:
        status = str(arm_throw.get("status") or "")
        gesture = arm_throw.get("gesture")
        if status == "completed" and gesture in GESTURES:
            return "prefetched", str(gesture)
        if status in {"failed", "completed"}:
            # failed, or completed without a usable gesture — interrupt.
            return "diagnose", None
        if is_cancelled():
            raise RuntimeError("cancelled")
        if time.monotonic() >= wait_until:
            return "diagnose", None
        time.sleep(0.05)


def _arm_throw_failure(
    on_log: Callable[[str], None],
    on_event: Callable[[Mapping[str, Any]], None],
    reason: str,
) -> dict[str, Any]:
    """Fabricate the arm-failure outcome in the observe diagnosis shape.

    与 observe 的诊断契约同形（``diagnosed``/``retry_required``），引擎据此
    路由 ``adjudication.diagnosis`` → analysis_failed；``diagnosis.message``
    由 rps.js 渲染到失败页状态行。
    """
    on_log(f"robot throw_gesture failed: {reason}")
    diagnosis = {
        "adjudicated": False,
        "verified": False,
        "diagnosed": True,
        "retry_required": True,
        "diagnosis": {
            "reason": _ARM_THROW_DIAGNOSIS_REASON,
            "message": f"机械臂未完成出拳（{reason}），可按绿色按钮再来一局",
        },
        "source": "local",
    }
    on_event({"event": "diagnosis", **diagnosis})
    return diagnosis


def _throw_agent_gesture(
    manifest: Mapping[str, Any],
    components: Any,
    gesture: str,
    *,
    on_event: Callable[[Mapping[str, Any]], None],
    is_cancelled: Callable[[], bool],
    on_log: Callable[[str], None],
    timeout_seconds: float,
) -> dict[str, Any] | None:
    """让机械臂亮出与裁决同一份随机拳形（静态手势，保持姿态不回 home）。

    臂是必需执行器：槽位缺失、provider 未实现 ``throw_gesture``、no-op 地板
    返回 ``skipped``、动作失败或抛异常都返回诊断型结果——本局走
    analysis_failed。成功返回 ``None``，裁决继续。
    """
    try:
        provider_id = resolve_provider_id(manifest, "robot_arm", "")
        if not provider_id:
            return _arm_throw_failure(on_log, on_event, "robot_arm 槽位未配置")
        robot = components.require(provider_id, expected_type="robot")
        throw = getattr(robot, "throw_gesture", None)
        if not callable(throw):
            return _arm_throw_failure(
                on_log, on_event, f"provider {provider_id} 未实现 throw_gesture"
            )
        outcome = throw(
            gesture,
            on_event=on_event,
            is_cancelled=is_cancelled,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        return _arm_throw_failure(on_log, on_event, repr(exc))
    if not (
        isinstance(outcome, dict)
        and outcome.get("status") == "completed"
        and not outcome.get("skipped")
    ):
        reason = (
            str(outcome.get("reason"))
            if isinstance(outcome, dict) and outcome.get("reason")
            else "动作未完成"
        )
        return _arm_throw_failure(on_log, on_event, reason)
    return None


def run(
    on_log: Callable[[str], None],
    is_cancelled: Callable[[], bool],
    timeout_seconds: float,
    *,
    components: Any,
    manifest: Mapping[str, Any],
    on_event: Callable[[Mapping[str, Any]], None],
    arm_throw: Any = None,
) -> dict[str, Any]:
    # A failed preparation/throw must not be masked by a no-hand diagnosis.
    if isinstance(arm_throw, Mapping) and arm_throw.get("status") == "failed":
        return _arm_throw_failure(
            on_log, on_event, str(arm_throw.get("reason") or "机械臂预备或出拳动作未完成")
        )
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
    if isinstance(arm_throw, Mapping) and arm_throw.get("status") == "failed":
        return _arm_throw_failure(
            on_log, on_event, str(arm_throw.get("reason") or "出拳动作未完成")
        )
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
    # agent 拳形：正常流在 play 状态已随口令 wav 提前出拳（臂侧随机、
    # outcome 回报同一份拳形），这里只消费；出拳失败/卡住=中断本局（诊断型
    # 结果 → analysis_failed，重试完整重放）；仅槽缺失（独立调用/旧装配）
    # 才回退管线直发——现场随机、observe 已锁定玩家。
    source, agent_gesture = _wait_prefetched_gesture(
        arm_throw,
        is_cancelled=is_cancelled,
        timeout_wait_budget_seconds=timeout_seconds,
    )
    if source == "prefetched":
        on_log(f"agent gesture {agent_gesture} consumed from prefetched arm throw")
    elif source == "legacy":
        agent_gesture = random.choice(GESTURES)
        arm_failure = _throw_agent_gesture(
            manifest,
            components,
            agent_gesture,
            on_event=on_event,
            is_cancelled=is_cancelled,
            on_log=on_log,
            timeout_seconds=timeout_seconds,
        )
        if arm_failure is not None:
            return arm_failure
    else:
        return _arm_throw_failure(
            on_log, on_event, str(arm_throw.get("reason") or "出拳失败或超时未回执")
        )
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
