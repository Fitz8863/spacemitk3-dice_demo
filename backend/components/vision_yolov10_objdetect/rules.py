"""Rules engine for the YOLOv10 package.

Shared with the v8 package (same schema, same evaluate/diagnose/project
contract); re-exported here so games resolving helpers from their selected
vision provider's package work unchanged.
"""
from components.vision_yolov8_objdetect.rules import (  # noqa: F401
    RuleError,
    diagnose_detection_failure,
    evaluate_rule,
    finalize_outcome,
    fuse_yolo_outcomes,
    project_result,
)
