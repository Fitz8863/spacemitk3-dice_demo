"""Loading and validation for game visual-adjudication profiles."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit


class ProfileError(ValueError):
    """Raised when a vision profile or component configuration is invalid."""


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME_CONFIG = PROJECT_ROOT / "vision" / "yolov8_adjudicator" / "config.json"


def resolve_project_path(value: str, project_root: Path = PROJECT_ROOT) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ProfileError("path must be a non-empty repository-relative string")
    raw = Path(value)
    if raw.is_absolute() or any(part in {"", ".", ".."} for part in raw.parts):
        raise ProfileError("path must stay inside the project")
    root = project_root.resolve()
    candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ProfileError("path escapes the project") from exc
    return candidate


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProfileError(f"unable to read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProfileError(f"{label} must be a JSON object")
    return payload


def resolve_runtime_config_path(
    component: Mapping[str, Any] | str | Path,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Resolve the C++ runtime configuration owned by the vision package.

    New component manifests declare ``runtime.config``.  A missing value is
    intentionally compatible with older deployments and falls back to the
    standard package location; arbitrary absolute paths and traversal are
    still rejected by :func:`resolve_project_path`.
    """
    if isinstance(component, Mapping):
        runtime = component.get("runtime", {})
        value = runtime.get("config") if isinstance(runtime, Mapping) else None
    else:
        value = component
    if value is None:
        candidate = DEFAULT_RUNTIME_CONFIG
        root = project_root.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ProfileError("default runtime config must stay inside the project") from exc
        return candidate
    return resolve_project_path(str(value), project_root)


def load_runtime_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate the hardware runtime configuration."""
    payload = _read_object(Path(path), "vision runtime config")
    if "rtsp" in payload and not isinstance(payload["rtsp"], dict):
        raise ProfileError("runtime config rtsp must be an object")
    video = payload.get("video", {})
    if not isinstance(video, dict):
        raise ProfileError("runtime config video must be an object")
    if "webrtc_base_url" in video:
        _validate_base_url(video["webrtc_base_url"], "runtime video.webrtc_base_url")
    return payload


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileError(f"{field} must be a non-empty string")
    return value.strip()


def _validate_video_path(path: Any) -> str:
    path = _required_string(path, "video.path")
    parsed = urlsplit(path)
    if not path.startswith("/") or parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise ProfileError("video.path must be a URL path")
    if ".." in Path(path).parts or any(ord(ch) < 32 for ch in path):
        raise ProfileError("video.path contains unsafe characters")
    return path


def _validate_camera(value: Any, field: str) -> str:
    """Validate a profile-owned camera selector without accepting traversal."""
    camera = _required_string(value, field)
    if any(ord(ch) < 32 for ch in camera) or ".." in Path(camera).parts:
        raise ProfileError(f"{field} contains unsafe characters")
    return camera


def _validate_base_url(value: Any, field: str = "video.webrtc_base_url") -> str:
    value = _required_string(value, field)
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        raise ProfileError(f"{field} must be an absolute HTTP(S) URL")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ProfileError(f"{field} must not contain a path or query")
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def compose_video_url(base_url: str, path: str) -> str:
    base = _validate_base_url(base_url)
    game_path = _validate_video_path(path)
    return base.rstrip("/") + "/" + game_path.lstrip("/")


def validate_profile(profile: dict[str, Any]) -> dict[str, Any]:
    if profile.get("schema_version") != 1:
        raise ProfileError("schema_version must be 1")
    game_id = profile.get("game_id")
    if not isinstance(game_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", game_id):
        raise ProfileError("game_id must be a non-empty identifier")
    vision = profile.get("vision")
    if not isinstance(vision, dict):
        raise ProfileError("vision must be an object")
    model = _required_string(vision.get("model"), "vision.model")
    resolve_project_path(model)
    if not isinstance(vision.get("class_map"), dict) or not vision["class_map"]:
        raise ProfileError("vision.class_map must be a non-empty object")
    if not isinstance(vision.get("participants"), list) or not vision["participants"]:
        raise ProfileError("vision.participants must be a non-empty array")
    assignment = vision.get("participant_assignment", "")
    grouping = vision.get("grouping", "")
    if assignment not in {"", "x_midpoint", "divider_regions"}:
        raise ProfileError("vision.participant_assignment must be x_midpoint or divider_regions")
    if grouping not in {"", "divider_regions", "x_midpoint"}:
        raise ProfileError("vision.grouping must be divider_regions or x_midpoint")
    if grouping == "divider_regions":
        divider = vision.get("divider", {})
        if not isinstance(divider, dict):
            raise ProfileError("vision.divider must be an object")
        orientation = divider.get("orientation", "vertical")
        if orientation not in {"vertical", "horizontal"}:
            raise ProfileError("vision.divider.orientation must be vertical or horizontal")
        position = divider.get("position", 0.5)
        if not isinstance(position, (int, float)) or isinstance(position, bool) or not math.isfinite(position) or not 0 < position < 1:
            raise ProfileError("vision.divider.position must be between 0 and 1")
    if not isinstance(vision.get("stable_frames"), int) or vision["stable_frames"] <= 0:
        raise ProfileError("vision.stable_frames must be a positive integer")

    llm = profile.get("llm")
    if not isinstance(llm, dict):
        raise ProfileError("llm must be an object")
    _required_string(llm.get("system_prompt"), "llm.system_prompt")
    _required_string(llm.get("user_prompt_template"), "llm.user_prompt_template")
    for prompt_field in ("diagnosis_system_prompt", "diagnosis_user_prompt_template"):
        if prompt_field in llm:
            _required_string(llm.get(prompt_field), f"llm.{prompt_field}")
    outcomes = llm.get("allowed_outcomes")
    if (
        not isinstance(outcomes, list)
        or not outcomes
        or not all(isinstance(item, str) and item.strip() for item in outcomes)
        or len(outcomes) != len(set(outcomes))
    ):
        raise ProfileError("llm.allowed_outcomes must be a non-empty unique array")
    if llm.get("context_mode") != "single_turn_no_history":
        raise ProfileError("llm.context_mode must be single_turn_no_history")
    llm_timeout = llm.get("timeout_seconds", 3)
    if (
        not isinstance(llm_timeout, (int, float))
        or isinstance(llm_timeout, bool)
        or not math.isfinite(llm_timeout)
        or llm_timeout <= 0
    ):
        raise ProfileError("llm.timeout_seconds must be a positive number")
    llm["timeout_seconds"] = float(llm_timeout)

    video = profile.get("video")
    if not isinstance(video, dict):
        raise ProfileError("video must be an object")
    if "webrtc_base_url" in video:
        _validate_base_url(video["webrtc_base_url"], "video.webrtc_base_url")
    _validate_video_path(video.get("path"))
    if video.get("enabled", True) not in {True, False}:
        raise ProfileError("video.enabled must be boolean")

    timeouts = profile.get("timeouts", {"adjudication_seconds": 120})
    if not isinstance(timeouts, dict):
        raise ProfileError("timeouts must be an object")
    adjudication_timeout = timeouts.get("adjudication_seconds")
    if (
        not isinstance(adjudication_timeout, (int, float))
        or isinstance(adjudication_timeout, bool)
        or not math.isfinite(adjudication_timeout)
        or adjudication_timeout <= 0
    ):
        raise ProfileError("timeouts.adjudication_seconds must be a positive number")
    if "diagnosis_llm_seconds" in timeouts:
        raise ProfileError(
            "timeouts.diagnosis_llm_seconds was removed; use llm.timeout_seconds"
        )
    normalized_timeouts = {"adjudication_seconds": float(adjudication_timeout)}
    for field, default in (("yolo_detection_seconds", adjudication_timeout),):
        value = timeouts.get(field, default)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ProfileError(f"timeouts.{field} must be a positive number")
        normalized_timeouts[field] = float(value)
    profile["timeouts"] = normalized_timeouts
    if "diagnosis_allowed_reason_codes" in llm:
        reasons = llm["diagnosis_allowed_reason_codes"]
        if not isinstance(reasons, list) or not reasons or not all(isinstance(item, str) and item.strip() for item in reasons):
            raise ProfileError("llm.diagnosis_allowed_reason_codes must be a non-empty string array")

    lifecycle = profile.get("lifecycle", {})
    if not isinstance(lifecycle, dict):
        raise ProfileError("lifecycle must be an object")
    hold = lifecycle.get("post_result_hold_seconds", 0)
    if not isinstance(hold, (int, float)) or isinstance(hold, bool) or not math.isfinite(hold) or hold < 0 or hold > 300:
        raise ProfileError("lifecycle.post_result_hold_seconds must be between 0 and 300 seconds")

    multi = profile.get("multi_view", {"enabled": False, "min_views": 1})
    if not isinstance(multi, dict) or not isinstance(multi.get("enabled", False), bool):
        raise ProfileError("multi_view must be an object with boolean enabled")
    if not isinstance(multi.get("min_views", 1), int) or multi.get("min_views", 1) < 1:
        raise ProfileError("multi_view.min_views must be a positive integer")
    views = multi.get("views", [])
    if views is None:
        views = []
    if not isinstance(views, list):
        raise ProfileError("multi_view.views must be an array")
    seen_view_ids: set[str] = set()
    for index, view in enumerate(views):
        field = f"multi_view.views[{index}]"
        if not isinstance(view, dict):
            raise ProfileError(f"{field} must be an object")
        view_id = _required_string(view.get("id"), f"{field}.id")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", view_id) or view_id in seen_view_ids:
            raise ProfileError(f"{field}.id must be unique and identifier-like")
        seen_view_ids.add(view_id)
        _validate_camera(view.get("camera"), f"{field}.camera")
        view_video = view.get("video")
        if not isinstance(view_video, dict):
            raise ProfileError(f"{field}.video must be an object")
        _validate_video_path(view_video.get("path"))
        if view_video.get("enabled", True) not in {True, False}:
            raise ProfileError(f"{field}.video.enabled must be boolean")
    if bool(multi.get("enabled", False)) and len(views) < int(multi.get("min_views", 1)):
        raise ProfileError("multi_view.views must contain at least min_views entries when enabled")
    if multi.get("yolo_fusion", "majority_vote") not in {"majority_vote"}:
        raise ProfileError("multi_view.yolo_fusion must be majority_vote")
    if multi.get("llm_images", "all_stable_views") not in {"all_stable_views"}:
        raise ProfileError("multi_view.llm_images must be all_stable_views")
    return profile


def load_profile(path: Path) -> dict[str, Any]:
    return validate_profile(_read_object(Path(path), "vision profile"))


def load_component_config(package_dir: Path) -> dict[str, Any]:
    payload = _read_object(Path(package_dir) / "config.json", "vision component config")
    if payload.get("schema_version") != 1:
        raise ProfileError("component config schema_version must be 1")
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("mode", "per_request") not in {"per_request", "resident"}:
        raise ProfileError("runtime.mode must be per_request or resident")
    if "binary" in runtime:
        resolve_project_path(runtime["binary"])
    if "working_dir" in runtime:
        resolve_project_path(runtime["working_dir"])
    resolve_runtime_config_path(payload)
    video = payload.get("video", {})
    if not isinstance(video, dict):
        raise ProfileError("component config video must be an object")
    if "webrtc_base_url" in video:
        _validate_base_url(video["webrtc_base_url"], "video.webrtc_base_url")
    return payload
