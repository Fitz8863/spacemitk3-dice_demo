from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.vision import VisionAdjudicationRequest  # noqa: E402
from components.vision_yolov8_adjudicator.profile import (  # noqa: E402
    ProfileError,
    compose_video_url,
    load_component_config,
    load_profile,
)
from games.dice import pipeline as dice_pipeline  # noqa: E402


def test_profile_loads_dice_and_composes_mediamtx_url():
    profile = load_profile(ROOT / "backend/games/dice/vision_profile.json")
    config = json.loads(
        (ROOT / "backend/components/vision_yolov8_adjudicator/config.json").read_text()
    )
    assert profile["game_id"] == "dice"
    assert profile["llm"]["context_mode"] == "single_turn_no_history"
    assert profile["video"]["path"] == "/dice/"
    assert compose_video_url(config["mediamtx"]["webrtc_base_url"], profile["video"]["path"]) == (
        "http://100.118.229.28:8889/dice/"
    )


def test_profile_rejects_full_url_in_game_path(tmp_path: Path):
    valid = {
        "schema_version": 1,
        "game_id": "bad",
        "vision": {"model": "vision/model.onnx", "class_map": {"0": "x"}, "participants": ["A"], "stable_frames": 1},
        "llm": {"system_prompt": "judge", "user_prompt_template": "judge", "allowed_outcomes": ["A"], "context_mode": "single_turn_no_history"},
    }
    valid["video"] = {"path": "https://x/"}
    path = tmp_path / "vision_profile.json"
    path.write_text(json.dumps(valid))
    with pytest.raises(ProfileError, match="video.path"):
        load_profile(path)


def test_compose_video_url_rejects_path_traversal():
    with pytest.raises(ProfileError, match="video.path"):
        compose_video_url("http://localhost:8889", "/../secret")


def test_adjudication_request_is_immutable_and_contains_profile():
    request = VisionAdjudicationRequest(
        game_id="dice", profile={"game_id": "dice"}, request_id="abc", timeout_seconds=2.0
    )
    assert request.game_id == "dice"
    assert request.profile["game_id"] == "dice"
    with pytest.raises(Exception):
        request.game_id = "rps"  # type: ignore[misc]


def _minimal_profile():
    return {
        "schema_version": 1,
        "game_id": "bad",
        "vision": {"model": "vision/model.onnx", "class_map": {"0": "x"}, "participants": ["A"], "stable_frames": 1},
        "llm": {"system_prompt": "judge", "user_prompt_template": "judge", "allowed_outcomes": ["A"], "context_mode": "single_turn_no_history"},
        "video": {"path": "/bad/"},
    }


def test_profile_rejects_absolute_model_path(tmp_path: Path):
    profile = _minimal_profile()
    profile["vision"]["model"] = "/tmp/model.onnx"
    path = tmp_path / "vision_profile.json"
    path.write_text(json.dumps(profile))
    with pytest.raises(ProfileError, match="path"):
        load_profile(path)


def test_profile_rejects_non_string_allowed_outcomes(tmp_path: Path):
    profile = _minimal_profile()
    profile["llm"]["allowed_outcomes"] = ["A", 1]
    path = tmp_path / "vision_profile.json"
    path.write_text(json.dumps(profile))
    with pytest.raises(ProfileError, match="allowed_outcomes"):
        load_profile(path)


def test_component_config_requires_valid_schema(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps({"schema_version": 2}))
    with pytest.raises(ProfileError, match="schema_version"):
        load_component_config(tmp_path)


def test_dice_pipeline_requires_loaded_profile():
    with pytest.raises(ValueError, match="vision profile"):
        dice_pipeline.run(lambda _: None, lambda: False, 1.0, components=object(), manifest={"providers": {}}, on_event=lambda _: None)
