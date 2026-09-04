"""Role-specific interfaces for visual providers.

The implementation algorithm (YOLO, another detector, or a multimodal model)
does not define the provider interface. The role does:

* an adjudicator produces a verified game outcome;
* a localizer produces target coordinates for spatial perception.

Keeping these roles separate prevents a coordinate detector from being wired
into a game-decision slot merely because both packages happen to use YOLO.
"""
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from core.components import Component
from core.llm import LlmProvider


class VisionProvider(Component):
    """Marker interface shared by visual adapters."""

    type = "vision"
    role = ""


@dataclass(frozen=True)
class VisionAdjudicationRequest:
    """Trusted, game-agnostic input for one visual adjudication round."""

    game_id: str
    profile: Mapping[str, Any]
    request_id: str
    timeout_seconds: float
    # LLM engine for this round's verification/diagnosis, resolved by the
    # pipeline from the ``llm`` provider slot (game manifest override >
    # arena default; hot-reloaded per round).  ``None`` means the deployment
    # resolved no LLM — providers treat that as "verification disabled" and
    # fall back to detector-only results, never as a round failure.
    llm_provider: LlmProvider | None = None


class VisionAdjudicatorProvider(VisionProvider):
    """Visual adapter that returns a verified game adjudication result."""

    role = "adjudicator"

    @abstractmethod
    def adjudicate(
        self,
        request: VisionAdjudicationRequest,
        *,
        on_log: Callable[[str], None],
        on_event: Callable[[dict[str, Any]], None],
        is_cancelled: Callable[[], bool],
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Run one bounded adjudication and return its final business result."""
        raise NotImplementedError


class VisionLocalizerProvider(VisionProvider):
    """Visual adapter that locates targets for spatial perception.

    The coordinate-frame and object schema should be finalized when the first
    localizer is integrated. This separate interface exists now so a localizer
    can never be selected as a game adjudicator by accident.
    """

    role = "localizer"

    @abstractmethod
    def locate(
        self,
        request: dict[str, Any],
        *,
        on_log: Callable[[str], None],
        on_event: Callable[[dict[str, Any]], None],
        is_cancelled: Callable[[], bool],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Locate requested targets and return structured coordinate data."""
        raise NotImplementedError
