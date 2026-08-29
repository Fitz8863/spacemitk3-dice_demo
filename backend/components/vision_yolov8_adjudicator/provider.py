"""Resident, game-agnostic YOLOv8 adjudication provider."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any, Callable, Mapping

from core.vision import VisionAdjudicationRequest, VisionAdjudicatorProvider
from components.vision_yolov8_adjudicator.llm import OpenAICompatibleVisionVerifier
from components.vision_yolov8_adjudicator.process import YoloRuntimeProcess
from components.vision_yolov8_adjudicator.rules import (
    evaluate_rule,
    finalize_outcome,
    project_result,
    fuse_yolo_outcomes,
)


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
        self._runtime_cache: dict[str, Any] = {}

    def adjudicate(self, request: VisionAdjudicationRequest, *, on_log: Callable[[str], None], on_event: Callable[[dict[str, Any]], None], is_cancelled: Callable[[], bool], timeout_seconds: float | None = None) -> dict[str, Any]:
        profile = request.profile; runtimes = []
        multi = profile.get("multi_view", {}) if isinstance(profile.get("multi_view"), Mapping) else {}
        views = multi.get("views") if multi.get("enabled") else None
        if not isinstance(views, list) or not views: views = [{"id": "default"}]
        try:
            def start(v):
                vid = str(v.get("id", "default")); rt = self._runtime_cache.get(vid)
                if rt is None:
                    rt = self.runtime_factory(vid); rt.start(profile, vid, prewarm=True); self._runtime_cache[vid] = rt
                return rt
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
            min_views = int(multi.get("min_views", 1))
            if len(observations) < min_views: raise RuntimeError("minimum views not reached")
            ordered = [observations[k] for k in sorted(observations)]
            vals = [str(o["yolo_outcome"]) for o in ordered if o.get("yolo_outcome")]
            rule = dict(profile.get("rule", {}) if isinstance(profile.get("rule"), Mapping) else {})
            vision = profile.get("vision", {}) if isinstance(profile.get("vision"), Mapping) else {}
            if "expected_count" not in rule and vision.get("expected_count") is not None:
                rule["expected_count"] = vision["expected_count"]
            participants = vision.get("participants")
            normalized = []
            for observation in ordered:
                if isinstance(participants, list) and isinstance(observation.get("participants"), Mapping):
                    obs = dict(observation); values = obs["participants"]
                    obs["participants"] = {name: values[name] for name in participants if name in values}
                    normalized.append(obs)
                else:
                    normalized.append(observation)
            if len(vals) > 1:
                yolo = fuse_yolo_outcomes(vals)
                if yolo is None: raise RuntimeError("no strict majority across views")
            else:
                yolo = vals[0] if vals else evaluate_rule(rule, normalized)
            on_event({"event":"phase", "phase":"verifying"}); cfg = profile.get("llm", {}); status, out = "timeout", None
            if cfg.get("enabled", True):
                paths = []
                for observation in ordered:
                    snapshot = observation.get("snapshot")
                    raw = snapshot.get("path") if isinstance(snapshot, Mapping) else None
                    if not isinstance(raw, str) or not raw.strip() or not Path(raw).is_absolute():
                        raise ValueError("snapshot.path must be an absolute path")
                    path = Path(raw).resolve()
                    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"} or not path.is_file():
                        raise ValueError("snapshot.path must reference an existing JPEG or PNG")
                    paths.append(path)
                verifier = self.verifier
                if not isinstance(verifier, OpenAICompatibleVisionVerifier):
                    vr = verifier.verify(image_paths=paths, system_prompt=cfg.get("system_prompt", ""), user_prompt=cfg.get("user_prompt_template", ""), allowed_outcomes=cfg.get("allowed_outcomes", []), timeout_seconds=float(cfg.get("timeout_seconds", timeout_seconds or request.timeout_seconds)))
                else:
                    endpoint = str(cfg.get("url", cfg.get("endpoint", "")))
                    if endpoint and not endpoint.rstrip("/").endswith("chat/completions"): endpoint = endpoint.rstrip("/") + "/chat/completions"
                    verifier = OpenAICompatibleVisionVerifier(endpoint, model=cfg.get("model"), api_key=cfg.get("api_key"))
                    vr = verifier.verify(image_paths=paths, system_prompt=cfg.get("system_prompt", ""), user_prompt=cfg.get("user_prompt_template", ""), allowed_outcomes=cfg.get("allowed_outcomes", []), timeout_seconds=float(cfg.get("timeout_seconds", timeout_seconds or request.timeout_seconds)))
                status, out = vr.status, vr.outcome
            decision = finalize_outcome(yolo_outcome=yolo, llm_outcome=out, llm_status=status)
            final = project_result(profile, decision, {**normalized[0], "views": normalized})
            on_event({"event":"result", **final}); hold = float(profile.get("lifecycle", {}).get("post_result_hold_seconds", 0))
            if hold > 0:
                end = time.monotonic() + hold
                while time.monotonic() < end:
                    if is_cancelled(): raise RuntimeError("cancelled")
                    remaining = max(0, end-time.monotonic()); on_event({"event":"phase", "phase":"holding", "remaining_ms":int(remaining*1000)})
                    time.sleep(min(0.25, remaining))
            on_event({"event":"complete", "phase":"complete"}); return final
        finally:
            for observation in locals().get("ordered", []):
                snapshot = observation.get("snapshot") if isinstance(observation, Mapping) else None
                raw = snapshot.get("path") if isinstance(snapshot, Mapping) else None
                if isinstance(raw, str):
                    try: Path(raw).unlink(missing_ok=True)
                    except OSError: pass
            # Resident runtimes stay alive between rounds; only stop on errors.
            pass
