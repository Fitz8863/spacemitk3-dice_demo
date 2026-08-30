"""Project physical dice adjudication into player and Agent roles."""
from __future__ import annotations

from numbers import Real
from typing import Any, Mapping

from core.participants import normalize_participants, role_for_winner


def _side_values(result: Mapping[str, Any], side: str) -> list[Any]:
    value = result.get(f"{side.lower()}_values")
    if not isinstance(value, list):
        raise ValueError(f"dice result is missing {side.lower()}_values")
    return list(value)


def _side_score(result: Mapping[str, Any], side: str) -> Real:
    value = result.get(f"{side.lower()}_sum")
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"dice result is missing numeric {side.lower()}_sum")
    return value


def project_participant_result(
    result: Mapping[str, Any], participants: Mapping[str, Any]
) -> dict[str, Any]:
    """Add role-based fields without changing physical compatibility fields."""
    if not isinstance(result, Mapping):
        raise ValueError("dice result must be an object")
    sides = normalize_participants(participants)
    winner = result.get("winner")
    if not isinstance(winner, str):
        raise ValueError("dice result winner must be LEFT, RIGHT, or TIE")
    projected = dict(result)
    projected.update({
        "winner_role": role_for_winner(winner, sides),
        "player_side": sides["player"],
        "agent_side": sides["agent"],
        "player_values": _side_values(result, sides["player"]),
        "agent_values": _side_values(result, sides["agent"]),
        "player_score": _side_score(result, sides["player"]),
        "agent_score": _side_score(result, sides["agent"]),
    })
    return projected
