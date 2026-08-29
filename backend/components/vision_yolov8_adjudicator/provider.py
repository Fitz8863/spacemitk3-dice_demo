"""Resident, game-agnostic YOLOv8 adjudication provider."""
from __future__ import annotations

import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any, Callable, Mapping

from core.vision import VisionAdjudicationRequest, VisionAdjudicatorProvider
from components.vision_yolov8_adjudicator.llm import OpenAICompatibleVisionVerifier
from components.vision_yolov8_adjudicator.process import YoloRuntimeProcess, _snapshot_path
from components.vision_yolov8_adjudicator.rules import (
    evaluate_rule,
    finalize_outcome,
    project_result,
    fuse_yolo_outcomes,
)
from components.vision_yolov8_adjudicator.profile import compose_video_url, load_component_config


def _parse_result_line(line: str) -> dict[str, Any] | None:
    raw = line.strip()
    if raw.startswith("[RESULT] "):
        raw = raw[9:]
    if not raw.startswith("{"):
        return None
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _consume_legacy_log_line(
    line: str,
    on_log: Callable[[str], None],
    on_event: Callable[[dict[str, Any]], None],
) -> dict[str, Any] | None:
    """Parse only the explicitly tagged result envelope from old runtimes."""
    if not line.startswith("[RESULT] "):
        on_log(line)
        return None
    parsed = _parse_result_line(line)
    if parsed is None:
        on_log(line)
        return None
    event = dict(parsed)
    event.setdefault("event", "result")
    on_event(event)
    return event


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
        # A resident process receives its snapshot root at process creation;
        # subsequent rounds reuse the same camera process.  Keep one private
        # root per view alive for that process lifetime and remove each
        # single-use snapshot after verification.
        self._runtime_snapshot_dirs: dict[str, Path] = {}

    @staticmethod
    def _video_event(profile: Mapping[str, Any], view_id: str, event: Mapping[str, Any]) -> dict[str, Any] | None:
        """Project runtime video notifications to the public WebRTC endpoint.

        Runtime URLs may be RTSP, loopback, or otherwise deployment-private;
        only the MediaMTX base URL from component config and profile-owned path
        are allowed to cross the provider boundary.
        """
        multi = profile.get("multi_view", {})
        path: Any = None
        if isinstance(multi, Mapping) and multi.get("enabled"):
            for view in multi.get("views", []):
                if isinstance(view, Mapping) and str(view.get("id")) == view_id:
                    video = view.get("video")
                    path = video.get("path") if isinstance(video, Mapping) else None
                    break
        if path is None:
            video = profile.get("video")
            path = video.get("path") if isinstance(video, Mapping) else None
        if not isinstance(path, str) or not path.strip():
            return None
        config = load_component_config(Path(__file__).parent)
        # A deployment may override the board's MediaMTX host without
        # changing game profiles.  The path remains profile-owned and is
        # validated by ``compose_video_url`` before crossing the API boundary.
        mediamtx = config.get("mediamtx", {})
        base = os.environ.get("DICE_MEDIAMTX_WEBRTC_BASE_URL", "") or (
            mediamtx.get("webrtc_base_url") if isinstance(mediamtx, Mapping) else ""
        )
        if not isinstance(base, str) or not base.strip():
            return None
        return {"event": "video", "url": compose_video_url(base, path), "view_id": view_id}

    @staticmethod
    def _llm_transport_config(profile: Mapping[str, Any]) -> dict[str, Any]:
        """Resolve deployment LLM endpoint/key separately from game prompts.

        Profiles carry only game-level prompt and output rules.  Endpoint and
        credentials are deployment concerns and come from component config or
        environment variables; environment values take precedence and are
        never included in a result or health payload.
        """
        component = load_component_config(Path(__file__).parent)
        configured = component.get("llm", {})
        configured = configured if isinstance(configured, Mapping) else {}
        game_llm = profile.get("llm", {})
        game_llm = game_llm if isinstance(game_llm, Mapping) else {}
        endpoint = (
            os.environ.get("DICE_LLM_ENDPOINT", "").strip()
            or os.environ.get("DICE_LLM_URL", "").strip()
            or str(configured.get("endpoint", configured.get("url", "")) or "").strip()
        )
        key = os.environ.get("DICE_LLM_API_KEY", "").strip() or str(
            configured.get("api_key", "") or ""
        ).strip()
        # ``model`` is harmless metadata, and allowing a profile value keeps
        # existing game profiles source-compatible while deployments can set a
        # common default in component config.
        model = str(game_llm.get("model") or configured.get("model") or "").strip() or None
        return {"endpoint": endpoint, "api_key": key or None, "model": model}

    def adjudicate(self, request: VisionAdjudicationRequest, *, on_log: Callable[[str], None], on_event: Callable[[dict[str, Any]], None], is_cancelled: Callable[[], bool], timeout_seconds: float | None = None) -> dict[str, Any]:
        profile = request.profile; runtimes = []
        multi = profile.get("multi_view", {}) if isinstance(profile.get("multi_view"), Mapping) else {}
        views = multi.get("views") if multi.get("enabled") else None
        if not isinstance(views, list) or not views: views = [{"id": "default"}]
        ordered: list[dict[str, Any]] = []
        cleanup_paths: set[Path] = set()
        strict_snapshot_roots: dict[str, Path] = {}
        round_completed = False
        round_started = False
        try:
            def start(v):
                vid = str(v.get("id", "default")); rt = self._runtime_cache.get(vid)
                if rt is None:
                    rt = self.runtime_factory(vid)
                    snapshot_dir = Path(tempfile.mkdtemp(prefix=f"vision-runtime-{vid}-"))
                    try:
                        rt.start(profile, vid, prewarm=True, snapshot_dir=snapshot_dir)
                    except TypeError:
                        # Keep injected test/fallback runtimes source-compatible
                        # while the production adapter receives the per-job
                        # directory above.
                        rt.start(profile, vid, prewarm=True)
                    self._runtime_cache[vid] = rt
                    self._runtime_snapshot_dirs[vid] = snapshot_dir
                if isinstance(rt, YoloRuntimeProcess):
                    root = self._runtime_snapshot_dirs.get(vid)
                    if root is not None:
                        strict_snapshot_roots[vid] = root.resolve()
                return rt
            with ThreadPoolExecutor(max_workers=len(views)) as pool: runtimes.extend(pool.map(start, views))
            for rt in runtimes:
                rt.send({"command":"START_ADJUDICATION", "request_id":request.request_id, "profile_id":profile.get("game_id")})
            round_started = True
            on_event({"event":"phase", "phase":"detecting"}); observations = {}
            def collect(rt, view):
                vid = str(view.get("id", "default")); found = None
                for event in rt.events():
                    if event.get("event") == "video":
                        video_event = self._video_event(profile, vid, event)
                        if video_event is not None:
                            on_event(video_event)
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
            on_event({"event":"phase", "phase":"verifying"}); cfg = profile.get("llm", {})
            cfg = cfg if isinstance(cfg, Mapping) else {}
            status, out = "timeout", None
            if cfg.get("enabled", True):
                paths = []
                for observation in ordered:
                    snapshot = observation.get("snapshot")
                    raw = snapshot.get("path") if isinstance(snapshot, Mapping) else None
                    if not isinstance(raw, str) or not raw.strip() or not Path(raw).is_absolute():
                        raise ValueError("snapshot.path must be an absolute path")
                    path = Path(raw).resolve()
                    # Real runtimes are constrained to this round's private
                    # directory.  Custom injected runtimes may use their own
                    # fixture directory for compatibility with older callers;
                    # those paths are still absolute, regular image files.
                    view_id = str(observation.get("view_id", "default"))
                    if view_id in strict_snapshot_roots:
                        try:
                            path = _snapshot_path(observation, strict_snapshot_roots[view_id])
                        except Exception:
                            raise ValueError("snapshot.path must stay inside task directory")
                    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"} or not path.is_file():
                        raise ValueError("snapshot.path must reference an existing JPEG or PNG")
                    paths.append(path)
                    cleanup_paths.add(path)
                verifier = self.verifier
                transport = self._llm_transport_config(profile)
                if not isinstance(verifier, OpenAICompatibleVisionVerifier):
                    vr = verifier.verify(image_paths=paths, system_prompt=cfg.get("system_prompt", ""), user_prompt=cfg.get("user_prompt_template", ""), allowed_outcomes=cfg.get("allowed_outcomes", []), timeout_seconds=float(cfg.get("timeout_seconds", timeout_seconds or request.timeout_seconds)), model=transport["model"])
                else:
                    endpoint = transport["endpoint"]
                    if endpoint and not endpoint.rstrip("/").endswith("chat/completions"): endpoint = endpoint.rstrip("/") + "/chat/completions"
                    verifier = OpenAICompatibleVisionVerifier(endpoint, model=transport["model"], api_key=transport["api_key"])
                    vr = verifier.verify(image_paths=paths, system_prompt=cfg.get("system_prompt", ""), user_prompt=cfg.get("user_prompt_template", ""), allowed_outcomes=cfg.get("allowed_outcomes", []), timeout_seconds=float(cfg.get("timeout_seconds", timeout_seconds or request.timeout_seconds)), model=transport["model"])
                status, out = vr.status, vr.outcome
            decision = finalize_outcome(yolo_outcome=yolo, llm_outcome=out, llm_status=status)
            final = project_result(profile, decision, {**normalized[0], "views": normalized})
            final_command = {"command": "FINAL_RESULT", "request_id": request.request_id,
                             "outcome": final["outcome"],
                             "decision_source": final.get("decision_source", "provider"),
                             "verification": final.get("verification", {})}
            for rt in runtimes:
                try:
                    rt.send(final_command)
                except Exception as exc:
                    on_log(f"[vision] FINAL_RESULT send failed: {exc}")
            on_event({"event":"result", **final})
            # Stop inference immediately after the verdict.  The resident
            # camera/RTSP pipeline remains alive, so the browser can continue
            # displaying the holding frame without spending YOLO cycles.
            for rt in runtimes:
                try:
                    rt.send({"command": "STOP_ADJUDICATION", "request_id": request.request_id})
                except Exception as exc:
                    on_log(f"[vision] STOP_ADJUDICATION send failed: {exc}")
            hold = float(profile.get("lifecycle", {}).get("post_result_hold_seconds", 0))
            if hold > 0:
                end = time.monotonic() + hold
                while time.monotonic() < end:
                    if is_cancelled(): raise RuntimeError("cancelled")
                    remaining = max(0, end-time.monotonic()); on_event({"event":"phase", "phase":"holding", "remaining_ms":int(remaining*1000)})
                    time.sleep(min(0.25, remaining))
            on_event({"event":"complete", "phase":"complete"}); round_completed = True; return final
        finally:
            # Only unlink paths that crossed the validation boundary above.
            # In particular, never trust an arbitrary snapshot path supplied
            # by a failed or malicious runtime event.
            for path in cleanup_paths:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            # If a round failed or was cancelled, explicitly return each
            # resident runtime to idle.  Normal rounds already sent STOP
            # before holding; no process is restarted between requests.
            if round_started and not round_completed:
                for rt in runtimes:
                    try:
                        rt.send({"command": "CANCEL", "request_id": request.request_id})
                    except Exception:
                        pass
