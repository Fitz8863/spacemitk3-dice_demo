"""Game-facing contract for robot-arm providers.

A robot provider drives exactly one physical arm.  The state machine calls it
through short imperative commands — grasp the dice cup (no shaking), shake the
held cup and put it back down, perform a result gesture, throw one static game
gesture, return home — and the
provider owns every process-level detail of the demo runtime behind those
verbs (resident process lifecycle, protocol framing, retries, interruption).

Command contract shared by all implementations:

* A command returns ``{"status": "completed", ...}`` or
  ``{"status": "failed", "reason": "<human readable>"}``.
* A command must respond to ``is_cancelled`` by aborting promptly and
  returning a failure (``reason`` mentioning the interruption).  It must never
  raise past the boundary: an arm exception belongs in arm_failed, not in a
  dead round.
* Progress reaches the frontend through
  ``on_event({"event": "robot", "phase": ..., "zh": ..., "progress": ...})``;
  completion is judged strictly by the return value, never by parsing logs.
* Providers serialize commands internally (one arm, one motion at a time);
  the engine may legally start the next command while the previous one is
  still winding down.
"""
from __future__ import annotations

from typing import Any, Callable

from core.components import Component

RobotEventFn = Callable[[dict[str, Any]], None]
RobotCancelledFn = Callable[[], bool]
RobotLogFn = Callable[[str], None]


class RobotProvider(Component):
    """Base class every robot-arm component must implement."""

    type = "robot"
    role = ""

    def grasp_cup(
        self,
        *,
        on_event: RobotEventFn,
        is_cancelled: RobotCancelledFn,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Locate the cup, grasp it and lift it — stopping before any shaking."""
        raise NotImplementedError

    def shake_dice(
        self,
        *,
        on_event: RobotEventFn,
        is_cancelled: RobotCancelledFn,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Shake the lifted cup, place it back down, and return the arm home."""
        raise NotImplementedError

    def feedback(
        self,
        kind: str,
        *,
        on_event: RobotEventFn,
        is_cancelled: RobotCancelledFn,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Perform one result gesture; ``kind`` is win/lose/draw."""
        raise NotImplementedError

    def throw_gesture(
        self,
        gesture: str | None = None,
        *,
        on_event: RobotEventFn,
        is_cancelled: RobotCancelledFn,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Throw one static game gesture (e.g. rock/paper/scissors).

        ``gesture`` uses the calling game's own vocabulary; ``None`` means the
        provider picks randomly (the prefetch flow dispatches before the
        verdict exists).  The outcome must report the gesture actually played
        as ``outcome["gesture"]`` — that value is the single source of truth
        for adjudication.  Non-abstract like reset_home: the floor reports
        ``skipped`` so arm-optional games keep working, while an arm-required
        game (rps) treats ``skipped`` as a failure.
        """
        return {"status": "completed", "skipped": True}

    def reset_home(
        self,
        *,
        on_event: RobotEventFn,
        is_cancelled: RobotCancelledFn,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Best-effort return to the home pose with fingers open.

        Non-abstract on purpose: the floor implementation keeps fixture and
        fallback providers zero-change.  Real providers override it; the
        revive policy is theirs to pick (a parking-invariant provider may
        revive a dead runtime — home is home even after a failure).
        """
        return {"status": "completed", "noop": True}

    def query_pose(
        self,
        *,
        on_event: RobotEventFn,
        is_cancelled: RobotCancelledFn,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Read-only "is the arm at home" probe for the idle patrol loop.

        Floor implementation: providers without pose feedback report
        ``skipped`` (cannot judge), which the patrol treats as "do nothing"
        — never a reason to move or restart the arm.
        """
        return {"status": "skipped", "reason": "provider has no pose feedback"}
