"""Dice game orchestration using a visual adjudicator selected by manifest."""
from __future__ import annotations

from typing import Any, Callable

from core.games import resolve_provider_id

GAME_ID = "dice"


def run(
    on_log: Callable[[str], None],
    is_cancelled: Callable[[], bool],
    timeout_seconds: float,
    *,
    components: Any,
    manifest: dict[str, Any],
    on_event: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    provider_id = resolve_provider_id(manifest, "vision_adjudicator", "vision_yolo")
    adjudicator = components.require(
        provider_id,
        expected_type="vision",
        expected_role="adjudicator",
    )
    adjudicate = getattr(adjudicator, "adjudicate", None)
    if not callable(adjudicate):
        raise RuntimeError(
            f"vision adjudicator {provider_id} does not implement adjudicate()"
        )
    return adjudicate(
        on_log=on_log,
        on_event=on_event,
        is_cancelled=is_cancelled,
        timeout_seconds=timeout_seconds,
    )
