from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.participants import normalize_participants, role_for_winner


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
