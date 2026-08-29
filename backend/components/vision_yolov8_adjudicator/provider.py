"""YOLOv8 adjudicator package entry point."""
from __future__ import annotations

from typing import Any

from core.vision import VisionAdjudicationRequest, VisionAdjudicatorProvider


class VisionYolov8Adjudicator(VisionAdjudicatorProvider):
    id = "vision_yolov8_adjudicator"
    type = "vision"
    role = "adjudicator"
    name = "YOLOv8 Vision Adjudicator"
    version = "2.0"

    def adjudicate(
        self,
        request: VisionAdjudicationRequest,
        *,
        on_log,
        on_event,
        is_cancelled,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError("YOLOv8 runtime adapter is implemented in a later task")
