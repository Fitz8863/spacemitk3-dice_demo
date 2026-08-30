from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.participants import normalize_participants, role_for_winner
from games.dice.result import project_participant_result


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"player": "LEFT", "agent": "RIGHT"}, {"player": "LEFT", "agent": "RIGHT"}),
        ({"player": "RIGHT", "agent": "LEFT"}, {"player": "RIGHT", "agent": "LEFT"}),
    ],
)
def test_normalize_participants_accepts_both_physical_layouts(raw, expected):
    assert normalize_participants(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {},
        {"player": "LEFT"},
        {"player": "LEFT", "agent": "LEFT"},
        {"player": "CENTER", "agent": "RIGHT"},
    ],
)
def test_normalize_participants_rejects_ambiguous_layouts(raw):
    with pytest.raises(ValueError, match="participants"):
        normalize_participants(raw)


def test_role_for_winner_preserves_tie_and_maps_both_sides():
    sides = {"player": "RIGHT", "agent": "LEFT"}
    assert role_for_winner("RIGHT", sides) == "PLAYER"
    assert role_for_winner("LEFT", sides) == "AGENT"
    assert role_for_winner("TIE", sides) == "TIE"
    with pytest.raises(ValueError, match="winner"):
        role_for_winner("UNKNOWN", sides)


def physical_result(winner="RIGHT"):
    return {
        "winner": winner,
        "outcome": {"kind": "winner", "value": winner},
        "left_values": [4, 4, 1, 1, 1],
        "right_values": [5, 4, 6, 2, 2],
        "left_sum": 11,
        "right_sum": 19,
        "first_dice": [4, 4, 1, 1, 1],
        "second_dice": [5, 4, 6, 2, 2],
        "first_sum": 11,
        "second_sum": 19,
    }


def test_project_participant_result_default_layout():
    result = project_participant_result(
        physical_result(), {"player": "LEFT", "agent": "RIGHT"}
    )
    assert result["winner"] == "RIGHT"
    assert result["winner_role"] == "AGENT"
    assert result["player_values"] == [4, 4, 1, 1, 1]
    assert result["agent_values"] == [5, 4, 6, 2, 2]
    assert result["player_score"] == 11
    assert result["agent_score"] == 19


def test_project_participant_result_swapped_layout_preserves_physical_fields():
    original = physical_result()
    result = project_participant_result(original, {"player": "RIGHT", "agent": "LEFT"})
    assert result["winner"] == "RIGHT"
    assert result["winner_role"] == "PLAYER"
    assert result["player_values"] == original["right_values"]
    assert result["agent_values"] == original["left_values"]
    assert result["player_score"] == 19
    assert result["agent_score"] == 11
    assert result["first_dice"] == original["first_dice"]
    assert result["second_dice"] == original["second_dice"]
    assert result["first_sum"] == original["first_sum"]
    assert result["second_sum"] == original["second_sum"]


def test_project_participant_result_maps_tie():
    result = project_participant_result(
        physical_result("TIE"), {"player": "LEFT", "agent": "RIGHT"}
    )
    assert result["winner_role"] == "TIE"


def test_project_participant_result_rejects_missing_side_evidence():
    result = physical_result()
    del result["left_values"]
    with pytest.raises(ValueError, match="left_values"):
        project_participant_result(result, {"player": "LEFT", "agent": "RIGHT"})
