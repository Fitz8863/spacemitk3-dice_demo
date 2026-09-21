"""Vision profile helpers for the YOLOv10 package.

The v8 and v10 vision packages share one profile schema (runtime_config
declaration, vision/rule/llm/video/multi_view sections, timeouts).  The v8
module owns the implementation; this module re-exports it so callers can
import from either package.  A future schema divergence would start here.
"""
from components.vision_yolov8_objdetect.profile import (  # noqa: F401
    ProfileError,
    compose_video_url,
    load_component_config,
    load_profile,
    load_runtime_config,
    resolve_project_path,
    resolve_runtime_config_path,
    validate_profile,
)
