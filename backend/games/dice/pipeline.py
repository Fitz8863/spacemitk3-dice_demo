"""Dice game orchestration.

The shared half of a vision game (provider slot resolution, the LLM fallback,
the request object, the diagnosis short-circuit) lives in
``core.vision_pipeline``; this module only says which game it is and how its
physical result becomes a player/Agent result.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

from core.vision_pipeline import run_vision_game
from games.dice.result import project_participant_result

GAME_ID = "dice"


def run(
    on_log: Callable[[str], None],
    is_cancelled: Callable[[], bool],
    timeout_seconds: float,
    *,
    components: Any,
    manifest: Mapping[str, Any],
    on_event: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    return run_vision_game(
        on_log,
        is_cancelled,
        timeout_seconds,
        components=components,
        manifest=manifest,
        on_event=on_event,
        game_id=GAME_ID,
        projector=project_participant_result,
    )
