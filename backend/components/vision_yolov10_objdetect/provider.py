"""Thin YOLOv10 vision provider: the v8 adjudicator pointed at the v10 runtime.

The two vision runtimes share the whole Python contract — resident process
lifecycle (``vision-control-v1`` + ``jsonl-events-v2``), profile schema,
rules engine, observation collection and the rule/LLM evaluation tail — so
this package only repoints the binary and the identity.  Everything else is
inherited from :mod:`components.vision_yolov8_objdetect.provider`.

What differs by design:

* the runtime binary / working dir come from **this package's** config.json
  (``vision/yolov10_objdetect``), not the v8 component's;
* games that select this provider get ``observe()``-style single-view
  evidence (the v10 runtime folds the vocabulary in C++ and emits labels),
  while ``adjudicate()`` keeps working unchanged for rule-driven games;
* the C++ side consumes a game ``adjudicator_config.json`` that also carries
  the model vocabulary (``classes`` / ``rps_map`` / ``filter_no_gesture``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import os

from components.vision_yolov8_objdetect.provider import (
    VisionYolov8Objdetect,
    resolve_runtime_binary,
)
from components.vision_yolov10_objdetect.process import YoloRuntimeProcess

COMPONENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = COMPONENT_DIR.parents[2]


def _default_runtime_factory(view_id: str = "default") -> YoloRuntimeProcess:
    """Build a runtime adapter bound to this package's deployment config.

    The v10 ``YoloRuntimeProcess`` resolves its default binary and working
    dir from **this** component's config.json, so the factory is a plain
    constructor call; the seam exists to keep the inherited provider's
    runtime-injection contract (tests pass their own factories).
    """
    return YoloRuntimeProcess()


class VisionYolov10Objdetect(VisionYolov8Objdetect):
    id = "vision_yolov10_objdetect"
    type = "vision"
    role = "adjudicator"
    name = "YOLOv10 Object Detection"
    version = "1.0"

    def __init__(self, manifest: dict[str, Any] | None = None, *, runtime_factory: Callable[..., Any] | None = None, verifier: Any | None = None) -> None:
        super().__init__(manifest, runtime_factory=runtime_factory or _default_runtime_factory, verifier=verifier)

    def health(self) -> dict[str, Any]:
        """Deployment readiness of the v10 binary (same contract as v8).

        Overridden only to resolve the binary from this package's config
        instead of the inherited v8 default.
        """
        payload: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "role": self.role,
        }
        try:
            binary = resolve_runtime_binary(component_dir=COMPONENT_DIR)
        except Exception as exc:
            return {**payload, "ok": False, "ready": False, "binary": "", "error": str(exc)}
        ready = binary.is_file() and os.access(binary, os.X_OK)
        return {**payload, "ok": ready, "ready": ready, "binary": str(binary)}
