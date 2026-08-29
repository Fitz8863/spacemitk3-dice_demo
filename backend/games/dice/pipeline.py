"""Dice game orchestration using a visual adjudicator selected by manifest."""
from __future__ import annotations

from typing import Any, Callable
import uuid

from core.games import resolve_provider_id
from core.vision import VisionAdjudicationRequest

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
    provider_id = resolve_provider_id(manifest, "vision_adjudicator", "vision_yolov8_adjudicator")
    profile = manifest.get("vision_profile")
    if not isinstance(profile, dict) or profile.get("game_id") != GAME_ID:
        raise ValueError("vision profile is required for dice game")
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
    request = VisionAdjudicationRequest(
        game_id=GAME_ID,
        profile=profile,
        request_id=uuid.uuid4().hex,
        timeout_seconds=timeout_seconds,
    )
    try:
        return adjudicate(
            request,
            on_log=on_log,
            on_event=on_event,
            is_cancelled=is_cancelled,
            timeout_seconds=timeout_seconds,
        )
    except TypeError as exc:
        # Migration compatibility for providers implementing the former
        # keyword-only interface; new adapters must accept the request object.
        if "positional" not in str(exc) and "required positional" not in str(exc):
            raise
        return adjudicate(
            on_log=on_log,
            on_event=on_event,
            is_cancelled=is_cancelled,
            timeout_seconds=timeout_seconds,
        )
