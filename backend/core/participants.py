"""Two-role game participant placement and physical winner mapping.

Besides the mapping itself this owns the **game-agnostic half of result
projection**: turning a physical ``LEFT``/``RIGHT``/``TIE`` verdict into the
player/Agent-facing role fields every vision game needs (:func:`project_roles`).
Game-specific evidence — dice pips, gesture categories — stays in the game's
own result module and layers on top.
"""
from __future__ import annotations

from typing import Any, Mapping


SIDES = {"LEFT", "RIGHT"}
ROLES = ("player", "agent")


def normalize_participants(value: Any) -> dict[str, str]:
    """Validate and normalize the player/Agent physical-side mapping."""
    if not isinstance(value, Mapping):
        raise ValueError("participants must map player and agent to physical sides")
    normalized: dict[str, str] = {}
    for role in ROLES:
        side = value.get(role)
        if side not in SIDES:
            raise ValueError(f"participants.{role} must be LEFT or RIGHT")
        normalized[role] = str(side)
    if normalized["player"] == normalized["agent"]:
        raise ValueError("participants.player and participants.agent must use different sides")
    return normalized


def role_for_winner(winner: str, participants: Mapping[str, Any]) -> str:
    """Map a physical LEFT/RIGHT/TIE winner to PLAYER/AGENT/TIE."""
    sides = normalize_participants(participants)
    if winner == "TIE":
        return "TIE"
    if winner == sides["player"]:
        return "PLAYER"
    if winner == sides["agent"]:
        return "AGENT"
    raise ValueError("winner must be LEFT, RIGHT, or TIE")


def project_roles(
    result: Mapping[str, Any], participants: Mapping[str, Any]
) -> dict[str, str]:
    """Validate a physical verdict and map it onto the player/Agent roles.

    Returns only the three role fields; callers merge them with their own
    evidence fields.  Shared by every vision game so the participant mapping is
    interpreted in exactly one place.
    """
    if not isinstance(result, Mapping):
        raise ValueError("result must be an object")
    sides = normalize_participants(participants)
    winner = result.get("winner")
    if not isinstance(winner, str):
        raise ValueError("result winner must be LEFT, RIGHT, or TIE")
    outcome = result.get("outcome")
    if outcome is not None and (
        not isinstance(outcome, Mapping) or outcome.get("value") != winner
    ):
        raise ValueError("result outcome.value must match winner")
    return {
        "winner_role": role_for_winner(winner, sides),
        "player_side": sides["player"],
        "agent_side": sides["agent"],
    }


def project_categorical_result(
    result: Mapping[str, Any],
    participants: Mapping[str, Any],
    *,
    choice_field: str = "choice",
) -> dict[str, Any]:
    """Project a categorical game's verdict (rock / scissors / paper, ...).

    The counterpart of the numeric path for games whose evidence is a **label**
    per side instead of a number: it reads ``left_<choice_field>`` /
    ``right_<choice_field>`` strings and therefore imposes no range or numeric
    validation.  Deciding the winner is still the rule engine's job
    (``categorical_relation``); this only re-labels the physical sides.
    """
    if not isinstance(result, Mapping):
        raise ValueError("result must be an object")
    roles = project_roles(result, participants)
    sides = normalize_participants(participants)
    choices: dict[str, str] = {}
    for side in ("LEFT", "RIGHT"):
        field = f"{side.lower()}_{choice_field}"
        value = result.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"result {field} must be a non-empty string")
        choices[side] = value.strip()
    projected = dict(result)
    projected.update(roles)
    projected.update({
        "player_choice": choices[sides["player"]],
        "agent_choice": choices[sides["agent"]],
    })
    return projected
