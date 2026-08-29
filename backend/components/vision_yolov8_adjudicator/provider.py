"""Resident, game-agnostic YOLOv8 adjudication provider."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any, Callable, Mapping

from core.vision import VisionAdjudicationRequest, VisionAdjudicatorProvider
from .llm import OpenAICompatibleVisionVerifier
from .process import YoloRuntimeProcess
from .rules import evaluate_rule, finalize_outcome, project_result, fuse_yolo_outcomes


class VisionYolov8Adjudicator(VisionAdjudicatorProvider):
    id = "vision_yolov8_adjudicator"
    type = "vision"
    role = "adjudicator"
    name = "YOLOv8 Vision Adjudicator"
    version = "2.0"

    def __init__(self, manifest: dict[str, Any] | None = None, *, runtime_factory: Callable[..., Any] | None = None, verifier: Any | None = None) -> None:
        super().__init__(manifest)
        self.runtime_factory = runtime_factory or (lambda view_id="default": YoloRuntimeProcess())
        self.verifier = verifier or OpenAICompatibleVisionVerifier()

    def adjudicate(self, request: VisionAdjudicationRequest, *, on_log: Callable[[str], None], on_event: Callable[[dict[str, Any]], None], is_cancelled: Callable[[], bool], timeout_seconds: float | None = None) -> dict[str, Any]:
        profile = request.profile; runtimes = []
        multi = profile.get("multi_view", {}) if isinstance(profile.get("multi_view"), Mapping) else {}
        views = multi.get("views") if multi.get("enabled") else None
        if not isinstance(views, list) or not views: views = [{"id": "default"}]
        try:
            def start(v):
                vid = str(v.get("id", "default")); rt = self.runtime_factory(vid); rt.start(profile, vid, prewarm=True); return rt
            with ThreadPoolExecutor(max_workers=len(views)) as pool: runtimes.extend(pool.map(start, views))
            for rt in runtimes: rt.send({"command":"START_ADJUDICATION", "request_id":request.request_id, "profile_id":profile.get("game_id")})
            on_event({"event":"phase", "phase":"detecting"}); observations = {}
            def collect(rt, view):
                vid = str(view.get("id", "default")); found = None
                for event in rt.events():
                    if event.get("event") == "video": on_event(dict(event, view_id=vid))
                    elif event.get("event") == "observation" and event.get("stable"):
                        found = dict(event, view_id=vid); break
                return vid, found
            deadline = time.monotonic() + float(timeout_seconds or request.timeout_seconds)
            with ThreadPoolExecutor(max_workers=len(runtimes)) as pool:
                futures = [pool.submit(collect, rt, view) for rt, view in zip(runtimes, views)]
                try:
                    for future in futures:
                        remaining = max(0.0, deadline - time.monotonic())
                        vid, found = future.result(timeout=remaining)
                        if found is not None: observations[vid] = found
                except FuturesTimeout as exc:
                    raise TimeoutError("YOLO adjudication timed out") from exc
            if is_cancelled():
                raise RuntimeError("cancelled")
            if not observations: raise RuntimeError("no stable observation")
            ordered = [observations[k] for k in sorted(observations)]
            vals = [str(o["yolo_outcome"]) for o in ordered if o.get("yolo_outcome")]
            yolo = (fuse_yolo_outcomes(vals) if len(vals) > 1 else (vals[0] if vals else None)) or evaluate_rule(profile.get("rule", {}), ordered)
            on_event({"event":"phase", "phase":"verifying"}); cfg = profile.get("llm", {}); status, out = "timeout", None
            if cfg.get("enabled", True):
                paths = [o.get("snapshot", {}).get("path") for o in ordered if isinstance(o.get("snapshot"), Mapping)]
                vr = self.verifier.verify(image_paths=paths, system_prompt=cfg.get("system_prompt", ""), user_prompt=cfg.get("user_prompt_template", ""), allowed_outcomes=cfg.get("allowed_outcomes", []), timeout_seconds=float(cfg.get("timeout_seconds", timeout_seconds or request.timeout_seconds)))
                status, out = vr.status, vr.outcome
            decision = finalize_outcome(yolo_outcome=yolo, llm_outcome=out, llm_status=status)
            final = project_result(profile, decision, {**ordered[0], "views": ordered})
            on_event({"event":"result", **final}); hold = float(profile.get("lifecycle", {}).get("post_result_hold_seconds", 0))
            if hold > 0:
                on_event({"event":"phase", "phase":"holding", "remaining_ms":int(hold*1000)}); end = time.monotonic() + hold
                while time.monotonic() < end:
                    if is_cancelled(): raise RuntimeError("cancelled")
                    time.sleep(min(0.25, end-time.monotonic()))
            on_event({"event":"complete", "phase":"complete"}); return final
        finally:
            for rt in runtimes:
                try: rt.stop()
                except Exception: pass
