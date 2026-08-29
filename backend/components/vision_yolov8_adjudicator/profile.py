"""Loading and validation for game visual-adjudication profiles."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


class ProfileError(ValueError):
    """Raised when a vision profile or component configuration is invalid."""


PROJECT_ROOT = Path(__file__).resolve().parents[3]


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


def _validate_base_url(value: Any) -> str:
    value = _required_string(value, "mediamtx.webrtc_base_url")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        raise ProfileError("mediamtx.webrtc_base_url must be an absolute HTTP(S) URL")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ProfileError("mediamtx.webrtc_base_url must not contain a path or query")
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
    if not isinstance(vision.get("stable_frames"), int) or vision["stable_frames"] <= 0:
        raise ProfileError("vision.stable_frames must be a positive integer")

    llm = profile.get("llm")
    if not isinstance(llm, dict):
        raise ProfileError("llm must be an object")
    _required_string(llm.get("system_prompt"), "llm.system_prompt")
    _required_string(llm.get("user_prompt_template"), "llm.user_prompt_template")
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

    video = profile.get("video")
    if not isinstance(video, dict):
        raise ProfileError("video must be an object")
    _validate_video_path(video.get("path"))
    if video.get("enabled", True) not in {True, False}:
        raise ProfileError("video.enabled must be boolean")

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
    mediamtx = payload.get("mediamtx")
    if not isinstance(mediamtx, dict):
        raise ProfileError("mediamtx must be an object")
    _validate_base_url(mediamtx.get("webrtc_base_url"))
    return payload
