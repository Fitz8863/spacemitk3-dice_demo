"""Rock-paper-scissors outcome logic and public result projection.

``decide`` 和 ``project`` 是这个游戏的常驻规则层：等手势模型训练好、
机械臂接入后，只有 ``pipeline.py`` 里的证据来源要换，这里的比较与投影
保持不变。
"""
from __future__ import annotations

from typing import Any, Mapping

from core.participants import project_categorical_result

# 拳形用中文短词：直接进台词占位符（{player_choice}）与前端展示；将来视觉
# class_map 的值同样允许中文（profile 校验器不限制 class_map 的值类型）。
GESTURES = ("石头", "剪刀", "布")

# 拳形 -> 它赢的拳形（石头赢剪刀、剪刀赢布、布赢石头）。
BEATS = {"石头": "剪刀", "剪刀": "布", "布": "石头"}


def decide(player_choice: str, agent_choice: str) -> str:
    """Return the role-level verdict: PLAYER, AGENT, or TIE."""
    if player_choice == agent_choice:
        return "TIE"
    if BEATS.get(player_choice) == agent_choice:
        return "PLAYER"
    return "AGENT"


def project(
    player_choice: str,
    agent_choice: str,
    participants: Mapping[str, str],
    *,
    source: str,
) -> dict[str, Any]:
    """Project one adjudicated round onto the public round result.

    ``decide`` 说的是玩家/Agent（游戏叙事的角色）；框架的类别投影说的是
    LEFT/RIGHT（摄像头看到的物理侧），所以胜者先经 participants 映射成
    物理侧，再交给 ``project_categorical_result`` 补齐角色字段。视觉裁决
    器将来不会替非 dice 游戏构造顶层 winner——这一步是 rps 自己的责任。
    """
    verdict = decide(player_choice, agent_choice)
    winner = "TIE" if verdict == "TIE" else str(participants[verdict.lower()])
    choices_by_side = {
        str(participants["player"]): player_choice,
        str(participants["agent"]): agent_choice,
    }
    return project_categorical_result(
        {
            "winner": winner,
            "outcome": {"kind": "winner", "value": winner},
            "left_choice": choices_by_side["LEFT"],
            "right_choice": choices_by_side["RIGHT"],
            "source": source,
        },
        participants,
    )
