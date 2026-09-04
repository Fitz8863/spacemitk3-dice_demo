"""Dice game orchestration using a visual adjudicator selected by manifest."""
from __future__ import annotations

from typing import Any, Callable
import uuid

from core.errors import DiceArenaError
from core.games import resolve_provider_id
from core.vision import VisionAdjudicationRequest
from games.dice.result import project_participant_result

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
    # The LLM engine for verification/diagnosis is resolved per round from the
    # ``llm`` slot (game manifest override > arena default, hot-reloaded).
    # A missing slot means YOLO-only rounds; a broken slot id must not kill
    # the round either — verification disables itself and the detector-only
    # result stands.
    llm_id = resolve_provider_id(manifest, "llm", "")
    llm_provider = None
    if llm_id:
        try:
            llm_provider = components.require(llm_id, expected_type="llm")
        except DiceArenaError as exc:
            on_log(f"[dice] llm provider {llm_id} unavailable: {exc.message}; round runs YOLO-only")
    request = VisionAdjudicationRequest(
        game_id=GAME_ID,
        profile=profile,
        request_id=uuid.uuid4().hex,
        timeout_seconds=timeout_seconds,
        llm_provider=llm_provider,
    )
    try:
        physical_result = adjudicate(
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
        physical_result = adjudicate(
            on_log=on_log,
            on_event=on_event,
            is_cancelled=is_cancelled,
            timeout_seconds=timeout_seconds,
        )
    # A failed YOLO round can still return an explainable diagnosis.  It is a
    # terminal retry result, not a physical winner, so do not force it through
    # the winner/score projection layer.
    if isinstance(physical_result, dict) and physical_result.get("diagnosed"):
        return physical_result
    return project_participant_result(physical_result, manifest["participants"])
