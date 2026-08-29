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
from components.vision_yolov8_adjudicator.rules import (  # noqa: E402
    RuleError,
    evaluate_rule,
    finalize_outcome,
    fuse_yolo_outcomes,
    project_result,
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


def test_majority_vote_requires_strict_majority():
    assert fuse_yolo_outcomes(["LEFT", "LEFT", "RIGHT"]) == "LEFT"
    assert fuse_yolo_outcomes(["LEFT", "RIGHT"]) is None
    assert fuse_yolo_outcomes([]) is None


def test_numeric_compare_sums_participant_values():
    rule = {"kind": "numeric_compare", "aggregation": "sum", "higher_wins": True, "tie_value": "TIE"}
    observations = [{"participants": {"LEFT": [6, 4], "RIGHT": [3, 5]}}]
    assert evaluate_rule(rule, observations) == "LEFT"


def test_numeric_compare_rejects_wrong_detection_count():
    rule = {
        "kind": "numeric_compare",
        "aggregation": "sum",
        "higher_wins": True,
        "tie_value": "TIE",
        "expected_count": 2,
    }
    observations = [{"participants": {"LEFT": [6], "RIGHT": [3, 5]}}]
    with pytest.raises(RuleError, match="expected_count"):
        evaluate_rule(rule, observations)


def test_categorical_relation_compares_two_participants():
    rule = {
        "kind": "categorical_relation",
        "relations": {"rock": "scissors", "scissors": "paper", "paper": "rock"},
        "tie_value": "TIE",
    }
    observations = [{"participants": {"LEFT": "rock", "RIGHT": "scissors"}}]
    assert evaluate_rule(rule, observations) == "LEFT"
    assert evaluate_rule(rule, [{"participants": {"LEFT": "rock", "RIGHT": "rock"}}]) == "TIE"


def test_categorical_relation_rejects_unknown_category():
    rule = {"kind": "categorical_relation", "relations": {"rock": "scissors"}}
    with pytest.raises(RuleError, match="unknown"):
        evaluate_rule(rule, [{"participants": {"LEFT": "lizard", "RIGHT": "scissors"}}])


def test_llm_success_overrides_yolo_mismatch():
    result = finalize_outcome(yolo_outcome="LEFT", llm_outcome="RIGHT", llm_status="success")
    assert result["outcome"]["value"] == "RIGHT"
    assert result["decision_source"] == "llm_override"
    assert result["adjudicated"] is True


def test_llm_timeout_falls_back_to_yolo():
    result = finalize_outcome(yolo_outcome="LEFT", llm_outcome=None, llm_status="timeout")
    assert result["outcome"]["value"] == "LEFT"
    assert result["decision_source"] == "yolo_timeout_fallback"


def test_llm_other_failure_is_an_error():
    with pytest.raises(RuleError, match="LLM"):
        finalize_outcome(yolo_outcome="LEFT", llm_outcome=None, llm_status="failure")


def test_project_result_adds_generic_and_dice_compatibility_fields():
    profile = {
        "game_id": "dice",
        "llm": {"allowed_outcomes": ["LEFT", "RIGHT", "TIE"]},
    }
    decision = finalize_outcome(yolo_outcome="LEFT", llm_outcome="LEFT", llm_status="success")
    result = project_result(
        profile,
        decision,
        {"rule": "numeric_compare", "participants": {"LEFT": [6, 4], "RIGHT": [3, 5]}},
    )
    assert result["profile_id"] == "dice"
    assert result["provider_id"] == "vision_yolov8_adjudicator"
    assert result["outcome"]["value"] == "LEFT"
    assert result["left_values"] == [6, 4]
    assert result["right_sum"] == 8
