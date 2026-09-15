"""Project physical dice adjudication into player and Agent roles."""
from __future__ import annotations

from numbers import Integral, Real
from typing import Any, Mapping

from core.participants import normalize_participants, project_roles


def _side_values(result: Mapping[str, Any], side: str) -> list[int]:
    field = f"{side.lower()}_values"
    value = result.get(field)
    if not isinstance(value, list) or not value:
        raise ValueError(f"dice result is missing valid {field}")
    if any(
        not isinstance(item, Integral)
        or isinstance(item, bool)
        or not 1 <= int(item) <= 6
        for item in value
    ):
        raise ValueError(f"dice result {field} must contain integers from 1 to 6")
    return [int(item) for item in value]


def _validate_score_field(result: Mapping[str, Any], field: str, expected: int) -> None:
    if field not in result:
        return
    value = result[field]
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or value != expected
    ):
        raise ValueError(f"dice result {field} must match its dice values")


def _validate_values_field(
    result: Mapping[str, Any], field: str, expected: list[int]
) -> None:
    if field in result and result[field] != expected:
        raise ValueError(f"dice result {field} must match its physical side")


def project_participant_result(
    result: Mapping[str, Any], participants: Mapping[str, Any]
) -> dict[str, Any]:
    """Add role-based fields without changing physical compatibility fields.

    The verdict and participant mapping are validated by the shared
    ``project_roles``; everything below is dice-specific evidence (five pips
    per side, each 1-6, summed).
    """
    roles = project_roles(result, participants)
    sides = normalize_participants(participants)
    values = {side: _side_values(result, side) for side in ("LEFT", "RIGHT")}
    scores = {side: sum(values[side]) for side in ("LEFT", "RIGHT")}
    _validate_score_field(result, "left_sum", scores["LEFT"])
    _validate_score_field(result, "right_sum", scores["RIGHT"])
    _validate_values_field(result, "first_dice", values["LEFT"])
    _validate_values_field(result, "second_dice", values["RIGHT"])
    _validate_score_field(result, "first_sum", scores["LEFT"])
    _validate_score_field(result, "second_sum", scores["RIGHT"])

    projected = dict(result)
    projected.update(roles)
    projected.update({
        "player_values": list(values[sides["player"]]),
        "agent_values": list(values[sides["agent"]]),
        "player_score": scores[sides["player"]],
        "agent_score": scores[sides["agent"]],
    })
    return projected
