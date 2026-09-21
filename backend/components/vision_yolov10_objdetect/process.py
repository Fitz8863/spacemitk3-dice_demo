"""Runtime adapters for the YOLOv10 package.

The v8 module owns the full resident-process orchestration; the only part
that must live **here** is the deployment-default resolution — a v10 runtime
must resolve ``runtime.binary`` / ``working_dir`` from *this* package's
config.json (``vision/yolov10_objdetect``), never from the v8 component's.
The subclass repoints exactly that; the protocol handshake (jsonl-events-v2)
is identical.
"""
from __future__ import annotations

from pathlib import Path

from components.vision_yolov8_objdetect.process import (  # noqa: F401
    EVENTS_PROTOCOL,
    SnapshotError,
    YoloRuntimeProcess as _V8YoloRuntimeProcess,
    build_rtsp_args,
    load_runtime_defaults,
    verify_snapshot,
)
from components.vision_yolov8_objdetect.profile import load_component_config

COMPONENT_DIR = Path(__file__).resolve().parent


class YoloRuntimeProcess(_V8YoloRuntimeProcess):
    """Resident YOLOv10 runtime adapter; deployment defaults from this package.

    The inherited ``start()`` calls ``load_runtime_defaults(Path(__file__).parent,
    profile)`` with the *v8* module path when it needs deployment defaults.
    Overriding the loader is not possible through that call shape, so this
    subclass pins binary/working_dir at construction time from this package's
    config.json; ``start()`` then takes the injected-binary path (the one
    production provider calls already use via the runtime factory).
    """

    def __init__(self, binary=None, working_dir=None) -> None:
        component = load_component_config(COMPONENT_DIR)
        runtime = component.get("runtime")
        runtime = runtime if isinstance(runtime, dict) else {}
        configured_binary = runtime.get("binary")
        configured_dir = runtime.get("working_dir")
        if not isinstance(configured_binary, str) or not configured_binary.strip():
            raise ValueError("vision_yolov10_objdetect config must declare runtime.binary")
        root = COMPONENT_DIR.parents[2]
        resolved_binary = str((root / configured_binary).resolve())
        resolved_dir = str((root / configured_dir).resolve()) if configured_dir else None
        super().__init__(
            binary=binary if binary is not None else resolved_binary,
            working_dir=working_dir if working_dir is not None else resolved_dir,
        )
