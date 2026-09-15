"""Game-agnostic visual adjudication pipeline.

Every vision game runs the same three steps: resolve the adjudicator slot for
this manifest, run one bounded adjudication, then project the physical
``LEFT``/``RIGHT``/``TIE`` outcome into player/Agent roles.  Only the last step
is game-specific — dice sums pips, a gesture game compares categories — so the
shared half lives here and each game module passes its own projector.

A game keeps its own tiny ``games/<id>/pipeline.py`` because
``core.games.run_game`` imports ``games.<game_id>.pipeline`` by convention.
That wrapper should stay a few lines: resolving providers, owning the LLM
fallback or deciding how to project a result here would force every future
game to repeat it.
"""
from __future__ import annotations

import uuid
from typing import Any, Callable, Mapping

from core.errors import DiceArenaError
from core.games import resolve_provider_id
from core.vision import VisionAdjudicationRequest

#: Callable that turns one physical adjudication result into the round's result.
#: Signature: ``(physical_result, participants) -> dict``.
ResultProjector = Callable[[Mapping[str, Any], Mapping[str, Any]], dict[str, Any]]


def run_vision_game(
    on_log: Callable[[str], None],
    is_cancelled: Callable[[], bool],
    timeout_seconds: float,
    *,
    components: Any,
    manifest: Mapping[str, Any],
    on_event: Callable[[dict[str, Any]], None],
    game_id: str,
    projector: ResultProjector,
) -> dict[str, Any]:
    """Run one vision game's adjudication and project its result.

    ``projector`` receives the provider's physical result plus the manifest's
    participant mapping, and returns the round result.  Everything before that
    — slot resolution, the soft LLM fallback, the request object and the
    provider-interface compatibility shim — is identical for every vision game
    and stays here.
    """
    provider_id = resolve_provider_id(
        manifest, "vision_adjudicator", "vision_yolov8_adjudicator"
    )
    profile = manifest.get("vision_profile")
    if not isinstance(profile, Mapping) or profile.get("game_id") != game_id:
        raise ValueError(f"vision profile is required for {game_id} game")
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
    # The LLM engine is resolved per round from the ``llm`` slot (game manifest
    # override > arena default, hot-reloaded).  A missing slot means YOLO-only
    # rounds; a broken slot id must not kill the round either — verification
    # disables itself and the detector-only result stands.
    llm_id = resolve_provider_id(manifest, "llm", "")
    llm_provider = None
    if llm_id:
        try:
            llm_provider = components.require(llm_id, expected_type="llm")
        except DiceArenaError as exc:
            on_log(
                f"[{game_id}] llm provider {llm_id} unavailable: {exc.message}; "
                "round runs YOLO-only"
            )
    request = VisionAdjudicationRequest(
        game_id=game_id,
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
    # A failed round can still return an explainable diagnosis.  It is a
    # terminal retry result, not a physical winner, so it must not be forced
    # through a game's winner/score projection layer.
    if isinstance(physical_result, dict) and physical_result.get("diagnosed"):
        return physical_result
    return projector(physical_result, manifest["participants"])
