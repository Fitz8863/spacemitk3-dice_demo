"""Two-role game participant placement and physical winner mapping."""
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
