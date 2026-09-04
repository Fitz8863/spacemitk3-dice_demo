from __future__ import annotations

import json
import os
import sys
import threading
import time
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
    load_runtime_config,
    resolve_runtime_config_path,
)
from components.vision_yolov8_adjudicator.rules import (  # noqa: E402
    RuleError,
    diagnose_detection_failure,
    evaluate_rule,
    finalize_outcome,
    fuse_yolo_outcomes,
    project_result,
)
from components.vision_yolov8_adjudicator.process import (  # noqa: E402
    SnapshotError,
    build_rtsp_args,
    verify_snapshot,
)
from components.vision_yolov8_adjudicator.provider import (  # noqa: E402
    VisionYolov8Adjudicator,
    normalize_observation,
)
from games.dice import pipeline as dice_pipeline  # noqa: E402


def test_profile_loads_dice_and_composes_mediamtx_url():
    manifest = json.loads((ROOT / "backend/games/dice/manifest.json").read_text())
    profile = manifest["vision_profile"]
    assert profile["game_id"] == "dice"
    assert profile["llm"]["context_mode"] == "single_turn_no_history"
    assert profile["video"]["path"] == "/dice/det"
    assert profile["vision"]["divider_detection"] is True
    component = load_component_config(ROOT / "backend" / "components" / "vision_yolov8_adjudicator")
    runtime = load_runtime_config(resolve_runtime_config_path(component))
    assert compose_video_url(runtime["video"]["webrtc_base_url"], profile["video"]["path"]) == (
        "http://100.118.229.28:8889/dice/det"
    )


def test_component_points_to_single_runtime_config_and_loads_hardware_defaults():
    component_dir = ROOT / "backend" / "components" / "vision_yolov8_adjudicator"
    component = load_component_config(component_dir)
    runtime_path = resolve_runtime_config_path(component)
    runtime = load_runtime_config(runtime_path)
    assert runtime_path == ROOT / "vision" / "yolov8_adjudicator" / "config.json"
    assert runtime["camera"] == "/dev/video1"
    assert runtime["rtsp"]["port"] == 8554
    assert runtime["video"]["webrtc_base_url"] == "http://100.118.229.28:8889"
    assert "rtsp" not in component
    assert "video" not in component


def test_profile_allows_component_owned_webrtc_base_url_and_validates_timeout(tmp_path: Path):
    profile = _minimal_profile()
    profile["video"].pop("webrtc_base_url")
    profile["timeouts"] = {"adjudication_seconds": 15}
    path = tmp_path / "vision_profile.json"
    path.write_text(json.dumps(profile))
    assert load_profile(path)["timeouts"]["adjudication_seconds"] == 15

    profile["video"]["webrtc_base_url"] = "http://localhost:8889"
    profile["video"]["webrtc_base_url"] = "http://localhost:8889/dice"
    path.write_text(json.dumps(profile))
    with pytest.raises(ProfileError, match="webrtc_base_url"):
        load_profile(path)


def test_profile_rejects_full_url_in_game_path(tmp_path: Path):
    valid = {
        "schema_version": 1,
        "game_id": "bad",
        "vision": {"model": "vision/model.onnx", "class_map": {"0": "x"}, "participants": ["A"], "stable_frames": 1},
        "llm": {"system_prompt": "judge", "user_prompt_template": "judge", "allowed_outcomes": ["A"], "context_mode": "single_turn_no_history"},
    }
    valid["video"] = {"path": "https://x/", "webrtc_base_url": "http://localhost:8889"}
    path = tmp_path / "vision_profile.json"
    path.write_text(json.dumps(valid))
    with pytest.raises(ProfileError, match="video.path"):
        load_profile(path)


def test_profile_accepts_optional_multiview_camera_and_video_paths(tmp_path: Path):
    profile = _minimal_profile()
    profile["multi_view"] = {
        "enabled": True,
        "min_views": 2,
        "yolo_fusion": "majority_vote",
        "llm_images": "all_stable_views",
        "views": [
            {"id": "front", "camera": "/dev/video1", "video": {"path": "/dice-front/"}},
            {"id": "side", "camera": "/dev/video2", "video": {"path": "/dice-side/"}},
        ],
    }
    path = tmp_path / "vision_profile.json"
    path.write_text(json.dumps(profile))
    loaded = load_profile(path)
    assert [view["id"] for view in loaded["multi_view"]["views"]] == ["front", "side"]
    assert loaded["multi_view"]["views"][1]["video"]["path"] == "/dice-side/"


def test_compose_video_url_rejects_path_traversal():
    with pytest.raises(ProfileError, match="video.path"):
        compose_video_url("http://localhost:8889", "/../secret")


def test_video_event_uses_profile_webrtc_base_url():
    profile = {"video": {"enabled": True, "path": "/rps/", "webrtc_base_url": "http://example.test:8889"}}
    event = VisionYolov8Adjudicator._video_event(profile, "default", {"event": "video"})
    assert event == {"event": "video", "url": "http://example.test:8889/rps/", "view_id": "default"}


def test_video_event_uses_component_webrtc_base_when_profile_has_only_path():
    profile = {"video": {"enabled": True, "path": "/rps/"}}
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "components.vision_yolov8_adjudicator.provider.load_component_config",
            lambda _path: {"video": {"webrtc_base_url": "http://component.test:8889"}},
        )
        event = VisionYolov8Adjudicator._video_event(profile, "default", {"event": "video"})
    assert event == {"event": "video", "url": "http://component.test:8889/rps/", "view_id": "default"}


def test_provider_prefers_game_adjudication_timeout_over_request_fallback():
    profile = {"timeouts": {"adjudication_seconds": 7}}
    assert VisionYolov8Adjudicator._adjudication_timeout(profile, 120) == 7


def test_profile_accepts_yolo_and_unified_llm_timeouts(tmp_path: Path):
    profile = _minimal_profile()
    profile["llm"]["timeout_seconds"] = 3
    profile["timeouts"] = {
        "yolo_detection_seconds": 8,
        "adjudication_seconds": 120,
    }
    profile["llm"]["diagnosis_system_prompt"] = "Diagnose only."
    profile["llm"]["diagnosis_user_prompt_template"] = "Summary: {detector_summary}"
    path = tmp_path / "vision_profile.json"
    path.write_text(json.dumps(profile))
    loaded = load_profile(path)
    assert loaded["timeouts"]["yolo_detection_seconds"] == 8.0
    assert loaded["timeouts"] == {"yolo_detection_seconds": 8.0, "adjudication_seconds": 120.0}
    assert loaded["llm"]["timeout_seconds"] == 3


def test_profile_rejects_invalid_llm_timeout(tmp_path: Path):
    profile = _minimal_profile()
    profile["llm"]["timeout_seconds"] = 0
    path = tmp_path / "vision_profile.json"
    path.write_text(json.dumps(profile))
    with pytest.raises(ProfileError, match=r"llm\.timeout_seconds"):
        load_profile(path)


def test_profile_rejects_removed_diagnosis_timeout(tmp_path: Path):
    profile = _minimal_profile()
    profile["timeouts"]["diagnosis_llm_seconds"] = 3
    path = tmp_path / "vision_profile.json"
    path.write_text(json.dumps(profile))
    with pytest.raises(ProfileError, match="use llm.timeout_seconds"):
        load_profile(path)


def test_local_diagnosis_reports_incomplete_dice_from_yolo_evidence():
    profile = {
        "game_id": "dice",
        "vision": {"expected_count": 5, "participants": ["LEFT", "RIGHT"]},
    }
    evidence = {"participants": {"LEFT": [1, 2, 3, 4], "RIGHT": [6, 6, 6, 6, 6]}}
    diagnosis = diagnose_detection_failure(profile, evidence)
    assert diagnosis["reason_code"] == "INCOMPLETE_OBJECTS"
    assert "LEFT" in diagnosis["message"]
    assert diagnosis["detected_counts"] == {"LEFT": 4, "RIGHT": 5}
    assert diagnosis["retry"] is True


def test_provider_yolo_timeout_calls_diagnosis_and_returns_retry_result(tmp_path: Path):
    image = tmp_path / "diagnostic.jpg"
    image.write_bytes(b"jpeg")

    class Runtime:
        def start(self, *args, **kwargs):
            self.events_data = iter([{
                "event": "diagnostic_snapshot",
                "stable": False,
                "snapshot": {"path": str(image)},
                "participants": {"LEFT": [1, 2, 3, 4], "RIGHT": [6, 6, 6, 6, 6]},
                "detections": [],
            }])

        def send(self, command):
            self.commands = getattr(self, "commands", []) + [dict(command)]

        def events(self):
            return self.events_data

        def stop(self):
            self.stopped = True

    captured = {}

    class Verifier:
        def diagnose(self, **kwargs):
            captured.update(kwargs)
            return type("R", (), {
                "status": "success",
                "reason_code": "OVERLAPPING_OBJECTS",
                "message": "左侧骰子可能叠放。",
                "retry": True,
                "error": None,
            })()

    profile = {
        "game_id": "dice",
        "vision": {"expected_count": 5, "participants": ["LEFT", "RIGHT"]},
        "llm": {
            "enabled": True,
            "timeout_seconds": 0.37,
            "system_prompt": "judge",
            "user_prompt_template": "judge",
            "diagnosis_system_prompt": "diagnose",
            "diagnosis_user_prompt_template": "Detector summary: {detector_summary}",
            "allowed_outcomes": ["LEFT", "RIGHT", "TIE"],
            "diagnosis_allowed_reason_codes": ["OVERLAPPING_OBJECTS", "UNKNOWN"],
        },
        "timeouts": {"yolo_detection_seconds": 0.01, "adjudication_seconds": 120},
        "lifecycle": {"post_result_hold_seconds": 0},
    }
    events = []
    runtime = Runtime()
    logs = []
    result = VisionYolov8Adjudicator(
        runtime_factory=lambda _view_id: runtime, verifier=Verifier()
    ).adjudicate(
        VisionAdjudicationRequest("dice", profile, "diagnose-round", 120),
        on_log=lambda _: None,
        on_event=events.append,
        is_cancelled=lambda: False,
    )
    assert result["adjudicated"] is False
    assert result["diagnosed"] is True
    assert result["retry_required"] is True
    assert result["diagnosis"]["source"] == "llm"
    assert result["diagnosis"]["reason_code"] == "OVERLAPPING_OBJECTS"
    assert captured["timeout_seconds"] == pytest.approx(0.37)
    assert any(event.get("event") == "diagnosis" for event in events)
    assert any(command["command"] in {"STOP_ADJUDICATION", "CANCEL"} for command in runtime.commands)


def test_provider_diagnosis_llm_timeout_uses_yolo_evidence_fallback(tmp_path: Path):
    image = tmp_path / "diagnostic.jpg"
    image.write_bytes(b"jpeg")

    class Runtime:
        def start(self, *args, **kwargs):
            self.events_data = iter([{
                "event": "diagnostic_snapshot",
                "stable": False,
                "snapshot": {"path": str(image)},
                "participants": {"LEFT": [], "RIGHT": []},
                "detections": [],
            }])
        def send(self, command):
            self.commands = getattr(self, "commands", []) + [dict(command)]
        def events(self): return self.events_data
        def stop(self): pass

    class Verifier:
        def diagnose(self, **kwargs):
            return type("R", (), {"status": "timeout", "reason_code": None, "message": None, "retry": True, "error": "timeout"})()

    profile = {
        "game_id": "dice",
        "vision": {"expected_count": 5, "participants": ["LEFT", "RIGHT"]},
        "llm": {
            "enabled": True,
            "timeout_seconds": 0.41,
            "system_prompt": "judge",
            "user_prompt_template": "judge",
            "diagnosis_system_prompt": "diagnose",
            "diagnosis_user_prompt_template": "Detector summary: {detector_summary}",
            "allowed_outcomes": ["LEFT", "RIGHT", "TIE"],
            "diagnosis_allowed_reason_codes": ["NO_OBJECTS_DETECTED", "UNKNOWN"],
        },
        "timeouts": {"yolo_detection_seconds": 0.01, "adjudication_seconds": 120},
        "lifecycle": {"post_result_hold_seconds": 0},
    }
    result = VisionYolov8Adjudicator(
        runtime_factory=lambda _view_id: Runtime(), verifier=Verifier()
    ).adjudicate(
        VisionAdjudicationRequest("dice", profile, "fallback-round", 120),
        on_log=lambda _: None,
        on_event=lambda _: None,
        is_cancelled=lambda: False,
    )
    assert result["diagnosis"]["source"] == "yolo_fallback"
    assert result["diagnosis"]["llm_status"] == "timeout"
    assert result["diagnosis"]["reason_code"] == "NO_OBJECTS_DETECTED"


def test_provider_skips_diagnosis_llm_after_total_budget_expires(tmp_path: Path):
    image = tmp_path / "diagnostic.jpg"
    image.write_bytes(b"jpeg")

    class Verifier:
        calls = 0

        def diagnose(self, **kwargs):
            self.calls += 1
            raise AssertionError("diagnosis LLM must not start after the total deadline")

    verifier = Verifier()
    profile = {
        "game_id": "dice",
        "vision": {"expected_count": 5, "participants": ["LEFT", "RIGHT"]},
        "llm": {
            "enabled": True,
            "timeout_seconds": 3,
            "diagnosis_allowed_reason_codes": ["NO_OBJECTS_DETECTED", "UNKNOWN"],
        },
    }
    result = VisionYolov8Adjudicator(verifier=verifier)._diagnose_failure(
        VisionAdjudicationRequest("dice", profile, "expired-diagnosis", 1),
        profile,
        [{
            "view_id": "default",
            "snapshot": {"path": str(image)},
            "participants": {"LEFT": [], "RIGHT": []},
        }],
        {},
        set(),
        lambda _: None,
        lambda _: None,
        time.monotonic() - 1,
    )
    assert verifier.calls == 0
    assert result["diagnosis"]["source"] == "yolo_fallback"
    assert result["diagnosis"]["llm_status"] == "timeout"


def test_dice_pipeline_preserves_diagnosis_without_projecting_winner():
    class Components:
        def require(self, *args, **kwargs):
            class Adjudicator:
                def adjudicate(self, request, **kwargs):
                    return {
                        "adjudicated": False,
                        "diagnosed": True,
                        "retry_required": True,
                        "diagnosis": {"reason_code": "NO_OBJECTS_DETECTED", "message": "未检测到骰子。"},
                    }
            return Adjudicator()

    manifest = {
        "participants": {"player": "LEFT", "agent": "RIGHT"},
        "providers": {"vision_adjudicator": "vision_yolov8_adjudicator"},
        "vision_profile": {"game_id": "dice"},
    }
    result = dice_pipeline.run(
        lambda _: None,
        lambda: False,
        1.0,
        components=Components(),
        manifest=manifest,
        on_event=lambda _: None,
    )
    assert result["diagnosed"] is True


class _SlotComponents:
    """Registry double: hands out a capturing adjudicator plus a fake LLM."""

    def __init__(self, llm=None, llm_error: Exception | None = None):
        self.llm = llm
        self.llm_error = llm_error
        self.llm_resolved: list[str] = []
        self.adjudicator = self._Adjudicator()

    class _Adjudicator:
        def __init__(self):
            self.request = None

        def adjudicate(self, request, **kwargs):
            self.request = request
            return {
                "adjudicated": False,
                "diagnosed": True,
                "retry_required": True,
                "diagnosis": {"reason_code": "NO_OBJECTS_DETECTED", "message": "未检测到骰子。"},
            }

    def require(self, provider_id, expected_type=None, expected_role=None):
        if expected_type == "llm":
            self.llm_resolved.append(provider_id)
            if self.llm_error is not None:
                raise self.llm_error
            return self.llm
        return self.adjudicator


_SLOT_MANIFEST = {
    "participants": {"player": "LEFT", "agent": "RIGHT"},
    "providers": {"vision_adjudicator": "vision_yolov8_adjudicator"},
    "vision_profile": {"game_id": "dice"},
}


def test_dice_pipeline_resolves_llm_slot_onto_request():
    llm = object()
    components = _SlotComponents(llm=llm)
    dice_pipeline.run(
        lambda _: None, lambda: False, 1.0,
        components=components,
        manifest={**_SLOT_MANIFEST, "providers": {**_SLOT_MANIFEST["providers"], "llm": "llm_openai_compat"}},
        on_event=lambda _: None,
    )
    assert components.llm_resolved == ["llm_openai_compat"]
    assert components.adjudicator.request.llm_provider is llm


def test_dice_pipeline_missing_llm_slot_leaves_request_without_provider():
    components = _SlotComponents(llm=object())
    dice_pipeline.run(
        lambda _: None, lambda: False, 1.0,
        components=components, manifest=_SLOT_MANIFEST, on_event=lambda _: None,
    )
    assert components.llm_resolved == []
    assert components.adjudicator.request.llm_provider is None


def test_dice_pipeline_broken_llm_slot_degrades_to_yolo_only():
    from core.errors import ComponentNotFoundError

    components = _SlotComponents(llm_error=ComponentNotFoundError("llm_broken"))
    logs: list[str] = []
    result = dice_pipeline.run(
        logs.append, lambda: False, 1.0,
        components=components,
        manifest={**_SLOT_MANIFEST, "providers": {**_SLOT_MANIFEST["providers"], "llm": "llm_broken"}},
        on_event=lambda _: None,
    )
    # The round survives: verification is disabled, detector result stands.
    assert result["diagnosed"] is True
    assert components.adjudicator.request.llm_provider is None
    assert any("llm_broken" in line and "YOLO-only" in line for line in logs)


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
        "video": {"path": "/bad/", "webrtc_base_url": "http://localhost:8889"},
        "timeouts": {"adjudication_seconds": 15},
    }


def test_profile_rejects_absolute_model_path(tmp_path: Path):
    profile = _minimal_profile()
    profile["vision"]["model"] = "/tmp/model.onnx"
    path = tmp_path / "vision_profile.json"
    path.write_text(json.dumps(profile))
    with pytest.raises(ProfileError, match="path"):
        load_profile(path)


def test_profile_validates_pre_adjudication_wait_seconds(tmp_path: Path):
    profile = _minimal_profile()
    path = tmp_path / "vision_profile.json"
    for bad_value in (-1, 301, "3", True):
        profile["lifecycle"] = {"pre_adjudication_wait_seconds": bad_value}
        path.write_text(json.dumps(profile))
        with pytest.raises(ProfileError, match="pre_adjudication_wait_seconds"):
            load_profile(path)

    profile["lifecycle"] = {"pre_adjudication_wait_seconds": 3}
    path.write_text(json.dumps(profile))
    assert load_profile(path)["lifecycle"]["pre_adjudication_wait_seconds"] == 3

    profile["lifecycle"] = {}
    path.write_text(json.dumps(profile))
    assert load_profile(path)["lifecycle"] == {}


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


def test_component_config_does_not_require_mediamtx(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps({
        "schema_version": 1,
        "runtime": {"mode": "resident", "prewarm_camera": True},
    }))
    assert load_component_config(tmp_path)["runtime"]["mode"] == "resident"


def test_runtime_config_exposes_mediamtx_base_and_component_has_no_duplicate_video():
    component_dir = ROOT / "backend" / "components" / "vision_yolov8_adjudicator"
    config = load_component_config(component_dir)
    runtime = load_runtime_config(resolve_runtime_config_path(config))
    assert runtime["video"]["webrtc_base_url"] == "http://100.118.229.28:8889"
    assert "video" not in config
    assert "rtsp" not in config


def test_provider_health_no_longer_reports_llm_state():
    """LLM configuration moved to the llm component; vision health is silent about it."""
    health = VisionYolov8Adjudicator().health()
    assert health["ok"] is True
    assert "llm_configured" not in health


def test_rtsp_args_emit_one_profile_owned_path():
    args = build_rtsp_args(
        {"rtsp": {"enabled": True, "host": "127.0.0.1", "port": 8554, "path": "/default"}},
        {"path": "/dice/"},
    )
    assert args.count("--rtsp-path") == 1
    assert args[args.index("--rtsp-path") + 1] == "/dice"


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


def test_llm_success_overrides_yolo_mismatch_only_when_corroborated():
    # A corroborated dissent may override the detector.
    result = finalize_outcome(
        yolo_outcome="LEFT", llm_outcome="RIGHT", llm_status="success",
        reask={"outcome": "RIGHT", "status": "success"},
    )
    assert result["outcome"]["value"] == "RIGHT"
    assert result["decision_source"] == "llm_override"
    assert result["adjudicated"] is True
    assert result["verification"]["reask_outcome"] == "RIGHT"

    # An uncorroborated dissent (no re-ask, timeout, or a third outcome)
    # never outranks the detector.
    for reask in (None, {"outcome": None, "status": "timeout"}, {"outcome": "TIE", "status": "success"}):
        result = finalize_outcome(
            yolo_outcome="LEFT", llm_outcome="RIGHT", llm_status="success", reask=reask,
        )
        assert result["outcome"]["value"] == "LEFT"
        assert result["decision_source"] == "yolo_reask_fallback"


def test_reask_confirms_yolo_after_llm_fluke():
    result = finalize_outcome(
        yolo_outcome="LEFT", llm_outcome="RIGHT", llm_status="success",
        reask={"outcome": "LEFT", "status": "success"},
    )
    assert result["outcome"]["value"] == "LEFT"
    assert result["decision_source"] == "yolo_reask_confirmed"
    assert result["verification"]["status"] == "reask_confirmed"


def test_tie_cannot_be_overridden_into_a_winner():
    # 14:14 is an arithmetic tie: however stable the verifier's dissent is,
    # it can never promote the outcome into a winner.
    result = finalize_outcome(
        yolo_outcome="TIE", llm_outcome="LEFT", llm_status="success",
        reask={"outcome": "LEFT", "status": "success"},
    )
    assert result["outcome"]["value"] == "TIE"
    assert result["decision_source"] == "tie_upheld"
    assert result["verification"]["status"] == "tie_upheld"
    assert result["verification"]["llm_outcome"] == "LEFT"

    # A custom tie_value gets the same protection.
    result = finalize_outcome(
        yolo_outcome="draw", llm_outcome="LEFT", llm_status="success",
        reask={"outcome": "LEFT", "status": "success"}, tie_value="draw",
    )
    assert result["outcome"]["value"] == "draw"
    assert result["decision_source"] == "tie_upheld"


def test_tie_with_flipping_llm_confirms_tie_via_reask():
    result = finalize_outcome(
        yolo_outcome="TIE", llm_outcome="LEFT", llm_status="success",
        reask={"outcome": "TIE", "status": "success"},
    )
    assert result["outcome"]["value"] == "TIE"
    assert result["decision_source"] == "yolo_reask_confirmed"


def test_llm_timeout_falls_back_to_yolo():
    result = finalize_outcome(yolo_outcome="LEFT", llm_outcome=None, llm_status="timeout")
    assert result["outcome"]["value"] == "LEFT"
    assert result["decision_source"] == "yolo_timeout_fallback"


def test_llm_failure_falls_back_to_yolo():
    result = finalize_outcome(yolo_outcome="LEFT", llm_outcome=None, llm_status="failure")
    assert result["outcome"]["value"] == "LEFT"
    assert result["decision_source"] == "yolo_failure_fallback"
    assert result["verification"]["status"] == "failure_fallback"
    assert result["verification"]["llm_called"] is True


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


def test_project_result_preserves_frontend_result_contract():
    profile = {"game_id": "dice", "llm": {"allowed_outcomes": ["LEFT", "RIGHT", "TIE"]}}
    decision = finalize_outcome(
        yolo_outcome="LEFT", llm_outcome="RIGHT", llm_status="success",
        reask={"outcome": "RIGHT", "status": "success"},
    )
    result = project_result(
        profile,
        decision,
        {"participants": {"LEFT": [6, 4], "RIGHT": [3, 5]}},
    )
    assert result["first_dice"] == [6, 4]
    assert result["second_dice"] == [3, 5]
    assert result["first_sum"] == 10
    assert result["second_sum"] == 8
    assert result["winner"] == "RIGHT"
    assert result["source"] == "llm_override"
    assert result["llm_winner"] == "RIGHT"


def test_provider_runs_one_round_and_holds_result(tmp_path: Path):
    image = tmp_path / "stable.jpg"; image.write_bytes(b"jpeg")
    class Runtime:
        def __init__(self, view_id="default"):
            self.commands = []; self.view_id = view_id
        def start(self, profile, view_id, prewarm=True): self.events_data = iter([
            {"event":"started","phase":"starting"}, {"event":"ready","phase":"idle"},
            {"event":"video","url":"http://x/dice/"},
            {"event":"progress","phase":"detecting","stable_count":1,"stable_frames":2},
            {"event":"observation","stable":True,"yolo_outcome":"LEFT","snapshot":{"path":str(image)},"participants":{"LEFT":[6],"RIGHT":[1]}},
        ])
        def send(self, command): self.commands.append(command)
        def events(self): return self.events_data
        def stop(self): pass
    runtimes=[]
    def factory(view_id="default"):
        r=Runtime(view_id); runtimes.append(r); return r
    class Verifier:
        def __init__(self): self.calls=0; self.timeout_seconds=None
        def verify(self, **kwargs):
            self.calls += 1; self.timeout_seconds = kwargs["timeout_seconds"]
            return type("R", (), {"status":"success","outcome":"LEFT","error":None})()
    profile={"game_id":"dice","vision":{"stable_frames":1},"llm":{"enabled":True,"timeout_seconds":0.29,"system_prompt":"s","user_prompt_template":"u","allowed_outcomes":["LEFT","RIGHT","TIE"]},"video":{"path":"/dice/","webrtc_base_url":"http://100.118.229.28:8889"},"multi_view":{"enabled":False,"min_views":1},"lifecycle":{"post_result_hold_seconds":0},"timeouts":{"adjudication_seconds":15}}
    events=[]; verifier=Verifier()
    result=VisionYolov8Adjudicator(runtime_factory=factory, verifier=verifier).adjudicate(VisionAdjudicationRequest("dice",profile,"r1",2),on_log=lambda x:None,on_event=events.append,is_cancelled=lambda:False)
    assert result["decision_source"] == "consensus"; assert verifier.calls == 1
    assert verifier.timeout_seconds == pytest.approx(0.29)
    assert any(r.commands and r.commands[0]["command"] == "START_ADJUDICATION" for r in runtimes)
    video_events = [event for event in events if event.get("event") == "video"]
    assert video_events == [{"event": "video", "url": "http://100.118.229.28:8889/dice/", "view_id": "default"}]
    progress_events = [event for event in events if event.get("event") == "progress"]
    assert progress_events == [{"event":"progress","phase":"detecting","stable_count":1,"stable_frames":2,"view_id":"default"}]


def test_post_result_hold_is_not_consumed_by_adjudication_deadline(tmp_path: Path):
    image = tmp_path / "stable.jpg"
    image.write_bytes(b"jpeg")

    class Runtime:
        def start(self, *args, **kwargs):
            self.events_data = iter([{
                "event": "observation",
                "stable": True,
                "yolo_outcome": "LEFT",
                "snapshot": {"path": str(image)},
            }])

        def send(self, command):
            pass

        def events(self):
            return self.events_data

        def stop(self):
            pass

    profile = {
        "game_id": "x",
        "vision": {"stable_frames": 1},
        "llm": {"enabled": False, "allowed_outcomes": ["LEFT", "RIGHT"]},
        "lifecycle": {"post_result_hold_seconds": 0.12},
        "timeouts": {"adjudication_seconds": 0.01},
    }
    events = []
    event_times = []

    def record_event(event):
        events.append(event)
        event_times.append((event.get("event"), time.monotonic()))

    VisionYolov8Adjudicator(runtime_factory=lambda _view_id: Runtime()).adjudicate(
        VisionAdjudicationRequest("x", profile, "r", 1),
        on_log=lambda _line: None,
        on_event=record_event,
        is_cancelled=lambda: False,
    )

    result_at = next(timestamp for event, timestamp in event_times if event == "result")
    complete_at = next(timestamp for event, timestamp in event_times if event == "complete")
    assert complete_at - result_at >= 0.10
    assert any(event.get("phase") == "holding" for event in events)
    assert events[-1] == {"event": "complete", "phase": "complete"}


def test_pre_adjudication_wait_delays_start_and_preserves_deadline(tmp_path: Path):
    image = tmp_path / "stable.jpg"
    image.write_bytes(b"jpeg")

    class Runtime:
        def __init__(self):
            self.commands = []
            self.command_times = []

        def start(self, *args, **kwargs):
            self.events_data = iter([{
                "event": "observation",
                "stable": True,
                "yolo_outcome": "LEFT",
                "snapshot": {"path": str(image)},
            }])

        def send(self, command):
            self.commands.append(dict(command))
            self.command_times.append(time.monotonic())

        def events(self):
            return self.events_data

        def stop(self):
            pass

    profile = {
        "game_id": "x",
        "vision": {"stable_frames": 1},
        "llm": {"enabled": False, "allowed_outcomes": ["LEFT", "RIGHT"]},
        "lifecycle": {"pre_adjudication_wait_seconds": 0.12, "post_result_hold_seconds": 0},
        "timeouts": {"adjudication_seconds": 0.01},
    }
    events = []
    runtime = Runtime()
    started_at = time.monotonic()

    result = VisionYolov8Adjudicator(runtime_factory=lambda _view_id: runtime).adjudicate(
        VisionAdjudicationRequest("x", profile, "r", 1),
        on_log=lambda _line: None,
        on_event=events.append,
        is_cancelled=lambda: False,
    )

    # The settling window publishes pre_wait phase events before detecting.
    pre_wait_indexes = [
        index for index, event in enumerate(events) if event.get("phase") == "pre_wait"
    ]
    detecting_index = next(
        index for index, event in enumerate(events) if event.get("phase") == "detecting"
    )
    assert pre_wait_indexes
    assert max(pre_wait_indexes) < detecting_index
    # START_ADJUDICATION is only sent after the wait has elapsed.
    assert runtime.commands[0]["command"] == "START_ADJUDICATION"
    assert runtime.command_times[0] - started_at >= 0.10
    # A 0.01s adjudication budget still completes successfully: the pre-wait
    # must not consume the deadline, which starts after START_ADJUDICATION.
    assert result["adjudicated"] is True
    assert result["outcome"]["value"] == "LEFT"
    assert events[-1] == {"event": "complete", "phase": "complete"}


def test_pre_adjudication_wait_cancellation_sends_no_commands():
    class Runtime:
        def __init__(self):
            self.commands = []
            self.stop_calls = 0

        def start(self, *args, **kwargs):
            pass

        def send(self, command):
            self.commands.append(dict(command))

        def events(self):
            return iter([])

        def stop(self):
            self.stop_calls += 1

    base = {
        "game_id": "x",
        "vision": {"stable_frames": 1},
        "llm": {"enabled": False, "allowed_outcomes": ["LEFT", "RIGHT"]},
        "lifecycle": {"pre_adjudication_wait_seconds": 30},
    }

    resident = dict(base, runtime={"mode": "resident", "prewarm_camera": True})
    resident_runtime = Runtime()
    with pytest.raises(RuntimeError, match="cancelled"):
        VisionYolov8Adjudicator(runtime_factory=lambda _vid: resident_runtime).adjudicate(
            VisionAdjudicationRequest("x", resident, "cancelled", 1),
            on_log=lambda _line: None,
            on_event=lambda _event: None,
            is_cancelled=lambda: True,
        )
    # The runtime never entered detecting: a resident process stays warm and
    # needs neither a control command nor a stop.
    assert resident_runtime.commands == []
    assert resident_runtime.stop_calls == 0

    per_request = dict(base, runtime={"mode": "per_request"})
    per_request_runtime = Runtime()
    with pytest.raises(RuntimeError, match="cancelled"):
        VisionYolov8Adjudicator(runtime_factory=lambda _vid: per_request_runtime).adjudicate(
            VisionAdjudicationRequest("x", per_request, "cancelled", 1),
            on_log=lambda _line: None,
            on_event=lambda _event: None,
            is_cancelled=lambda: True,
        )
    assert per_request_runtime.commands == []
    assert per_request_runtime.stop_calls == 1


def test_provider_multiview_sends_single_llm_request(tmp_path: Path):
    (tmp_path / "a.jpg").write_bytes(b"a"); (tmp_path / "b.jpg").write_bytes(b"b")
    class Runtime:
        def __init__(self, view_id): self.view_id=view_id; self.commands=[]
        def start(self,*a,**k): self.events_data=iter([{"event":"ready","phase":"idle"},{"event":"observation","stable":True,"yolo_outcome":"LEFT","snapshot":{"path":str(tmp_path / "a.jpg")}},{"event":"observation","stable":True,"yolo_outcome":"LEFT","snapshot":{"path":str(tmp_path / "b.jpg")}}])
        def send(self,c): self.commands.append(c)
        def events(self): return self.events_data
        def stop(self): pass
    class V:
        def __init__(self): self.calls=0
        def verify(self, **kw): self.calls+=1; return type("R",(),{"status":"success","outcome":"LEFT","error":None})()
    profile={"game_id":"x","vision":{"stable_frames":1},"llm":{"enabled":True,"system_prompt":"s","user_prompt_template":"u","allowed_outcomes":["LEFT","RIGHT"]},"multi_view":{"enabled":True,"min_views":2,"views":[{"id":"a"},{"id":"b"}]},"lifecycle":{"post_result_hold_seconds":0}}
    v=V(); p=VisionYolov8Adjudicator(runtime_factory=lambda vid:Runtime(vid),verifier=v)
    out=p.adjudicate(VisionAdjudicationRequest("x",profile,"r",2),on_log=lambda x:None,on_event=lambda e:None,is_cancelled=lambda:False)
    assert out["outcome"]["value"] == "LEFT" and v.calls == 1


def test_provider_reasks_llm_on_disagreement(tmp_path: Path):
    (tmp_path / "a.jpg").write_bytes(b"a")
    class Runtime:
        def start(self, *a, **k):
            self.events_data = iter([{"event": "observation", "stable": True, "yolo_outcome": "TIE", "snapshot": {"path": str(tmp_path / "a.jpg")}}])
        def send(self, c): pass
        def events(self): return self.events_data
        def stop(self): pass
    class V:
        def __init__(self, answers): self.calls = 0; self.answers = answers
        def verify(self, **kw):
            self.calls += 1
            outcome = self.answers[min(self.calls, len(self.answers)) - 1]
            return type("R", (), {"status": "success", "outcome": outcome, "error": None})()
    profile = {"game_id": "x", "vision": {"stable_frames": 1}, "rule": {"kind": "numeric_compare", "aggregation": "sum", "tie_value": "TIE"}, "llm": {"enabled": True, "timeout_seconds": 3, "system_prompt": "s", "user_prompt_template": "u", "allowed_outcomes": ["LEFT", "RIGHT", "TIE"]}, "lifecycle": {"post_result_hold_seconds": 0}, "timeouts": {"adjudication_seconds": 15}}

    # The re-ask flips back to the detector: the first dissent was a fluke.
    v = V(["LEFT", "TIE"])
    (tmp_path / "a.jpg").write_bytes(b"a")  # each round's cleanup unlinks it
    out = VisionYolov8Adjudicator(runtime_factory=lambda vid: Runtime(), verifier=v).adjudicate(
        VisionAdjudicationRequest("x", profile, "r", 2),
        on_log=lambda x: None, on_event=lambda e: None, is_cancelled=lambda: False,
    )
    assert v.calls == 2
    assert out["decision_source"] == "yolo_reask_confirmed"
    assert out["outcome"]["value"] == "TIE"

    # The dissent is stable, but a tie is arithmetic fact and stays a tie.
    v = V(["LEFT", "LEFT"])
    (tmp_path / "a.jpg").write_bytes(b"a")
    out = VisionYolov8Adjudicator(runtime_factory=lambda vid: Runtime(), verifier=v).adjudicate(
        VisionAdjudicationRequest("x", profile, "r", 2),
        on_log=lambda x: None, on_event=lambda e: None, is_cancelled=lambda: False,
    )
    assert v.calls == 2
    assert out["decision_source"] == "tie_upheld"
    assert out["outcome"]["value"] == "TIE"

    # Away from the tie, a stable dissent still overrides (documented semantics).
    class RuntimeLeft:
        def start(self, *a, **k):
            self.events_data = iter([{"event": "observation", "stable": True, "yolo_outcome": "RIGHT", "snapshot": {"path": str(tmp_path / "a.jpg")}}])
        def send(self, c): pass
        def events(self): return self.events_data
        def stop(self): pass
    v = V(["LEFT", "LEFT"])
    (tmp_path / "a.jpg").write_bytes(b"a")
    out = VisionYolov8Adjudicator(runtime_factory=lambda vid: RuntimeLeft(), verifier=v).adjudicate(
        VisionAdjudicationRequest("x", profile, "r", 2),
        on_log=lambda x: None, on_event=lambda e: None, is_cancelled=lambda: False,
    )
    assert v.calls == 2
    assert out["decision_source"] == "llm_override"
    assert out["outcome"]["value"] == "LEFT"


def test_provider_cleans_runtime_snapshots_after_llm(tmp_path: Path):
    image = tmp_path / "stable.jpg"; image.write_bytes(b"jpeg")
    class Runtime:
        def start(self, *args, **kwargs):
            self.events_data = iter([{"event":"observation", "stable":True, "yolo_outcome":"LEFT", "snapshot":{"path":str(image)}, "participants":{"LEFT":[1],"RIGHT":[2]}}])
        def send(self, command): pass
        def events(self): return self.events_data
        def stop(self): pass
    class V:
        def verify(self, **kwargs):
            assert image.exists()
            return type("R", (), {"status":"success", "outcome":"LEFT", "error":None})()
    profile={"game_id":"x","vision":{"stable_frames":1},"llm":{"enabled":True,"system_prompt":"s","user_prompt_template":"u","allowed_outcomes":["LEFT","RIGHT"]},"lifecycle":{"post_result_hold_seconds":0}}
    VisionYolov8Adjudicator(runtime_factory=lambda vid: Runtime(), verifier=V()).adjudicate(VisionAdjudicationRequest("x",profile,"r",2),on_log=lambda x:None,on_event=lambda e:None,is_cancelled=lambda:False)
    assert not image.exists()


def test_provider_applies_vision_expected_count_to_rule():
    class Runtime:
        def __init__(self):
            self.commands = []

        def start(self, *args, **kwargs):
            self.events_data = iter([
                {"event": "observation", "stable": True,
                 "snapshot": {"path": "/tmp/no.jpg"},
                 "participants": {"LEFT": [1], "RIGHT": [2, 3]}},
            ])

        def send(self, command):
            self.commands.append(dict(command))

        def events(self):
            return self.events_data

        def stop(self):
            pass

    runtime = Runtime()
    profile = {
        "game_id": "x",
        "vision": {"stable_frames": 1, "expected_count": 2},
        "rule": {"kind": "numeric_compare", "aggregation": "sum", "higher_wins": True, "tie_value": "TIE"},
        "llm": {"enabled": False, "allowed_outcomes": ["LEFT", "RIGHT", "TIE"]},
        "lifecycle": {"post_result_hold_seconds": 0},
    }
    result = VisionYolov8Adjudicator(runtime_factory=lambda vid: runtime).adjudicate(
        VisionAdjudicationRequest("x", profile, "r", 2),
        on_log=lambda _: None, on_event=lambda _: None, is_cancelled=lambda: False,
    )

    assert result["diagnosed"] is True
    assert result["diagnosis"]["reason_code"] == "INCOMPLETE_OBJECTS"
    assert result["diagnosis"]["detected_counts"] == {"LEFT": 1, "RIGHT": 2}
    assert [command["command"] for command in runtime.commands] == [
        "START_ADJUDICATION", "STOP_ADJUDICATION",
    ]


def test_snapshot_is_read_and_removed_after_verification(tmp_path: Path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    image = task_dir / "stable.jpg"
    image.write_bytes(b"jpeg-bytes")
    seen = {}

    class StubVerifier:
        def verify(self, **kwargs):
            seen["bytes"] = Path(kwargs["image_path"]).read_bytes()
            return type("Result", (), {"status": "success", "outcome": "LEFT"})()

    result = verify_snapshot(
        {"snapshot": {"path": str(image)}}, task_dir=task_dir,
        verifier=StubVerifier(), system_prompt="Judge", user_prompt="JSON",
        allowed_outcomes=["LEFT"], timeout_seconds=1,
    )
    assert result.outcome == "LEFT"
    assert seen["bytes"] == b"jpeg-bytes"
    assert not image.exists()


def test_snapshot_rejects_path_outside_task_directory(tmp_path: Path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    image = tmp_path / "outside.jpg"
    image.write_bytes(b"jpeg-bytes")
    with pytest.raises(SnapshotError):
        verify_snapshot(
            {"snapshot": {"path": str(image)}}, task_dir=task_dir,
            verifier=object(), system_prompt="Judge", user_prompt="JSON",
            allowed_outcomes=["LEFT"], timeout_seconds=1,
        )


def test_runtime_process_uses_dedicated_control_and_event_fds(tmp_path: Path):
    """A noisy runtime must not be able to corrupt structured events."""
    script = tmp_path / "fake_runtime.py"
    script.write_text(
        """#!/usr/bin/env python3
import argparse, json, os, sys
p=argparse.ArgumentParser(); p.add_argument('--config'); p.add_argument('--control-fd',type=int); p.add_argument('--event-fd',type=int); p.add_argument('--snapshot-dir',default='/tmp'); p.add_argument('--view-id',default='default'); p.add_argument('--prewarm',action='store_true'); p.add_argument('--no-display',action='store_true'); p.add_argument('--rtsp',action='store_true'); p.add_argument('--rtsp-host'); p.add_argument('--rtsp-port'); p.add_argument('--rtsp-path'); a=p.parse_args()
def emit(e): os.write(a.event_fd, (json.dumps(e)+'\\n').encode())
print('diagnostic line that is not JSON', flush=True)
emit({'event':'started','phase':'starting'}); emit({'event':'ready','phase':'idle'}); emit({'event':'video','url':'rtsp://private/cam'})
buf=b''
while True:
    chunk=os.read(a.control_fd, 4096)
    if not chunk: break
    buf += chunk
    while b'\\n' in buf:
        line, buf = buf.split(b'\\n',1)
        cmd=json.loads(line)
        if cmd.get('command') == 'START_ADJUDICATION':
            path=os.path.join(a.snapshot_dir, 'stable.jpg'); open(path,'wb').write(b'jpeg')
            emit({'event':'observation','stable':True,'yolo_outcome':'LEFT','snapshot':{'path':path}})
        elif cmd.get('command') == 'STOP_ADJUDICATION': emit({'event':'phase','phase':'idle'})
        elif cmd.get('command') == 'CANCEL': emit({'event':'cancelled'}); raise SystemExit
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    from components.vision_yolov8_adjudicator.process import YoloRuntimeProcess
    runtime = YoloRuntimeProcess(binary=script)
    runtime.start({}, "front", prewarm=True, snapshot_dir=tmp_path / "snapshots")
    try:
        initial = [next(runtime.events()) for _ in range(3)]
        assert [item["event"] for item in initial] == ["started", "ready", "video"]
        runtime.send({"command": "START_ADJUDICATION"})
        observation = next(item for item in runtime.events() if item.get("event") == "observation")
        assert Path(observation["snapshot"]["path"]).is_file()
        assert str(tmp_path / "snapshots") in observation["snapshot"]["path"]
    finally:
        runtime.stop()


def test_runtime_process_passes_explicit_runtime_config_for_profile_binary(tmp_path: Path):
    script = tmp_path / "config_runtime.py"
    script.write_text(
        """#!/usr/bin/env python3
import argparse, json, os, sys
p=argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--control-fd',type=int); p.add_argument('--event-fd',type=int); p.add_argument('--view-id',default='default'); p.add_argument('--no-display',action='store_true'); p.add_argument('--prewarm',action='store_true'); p.add_argument('--rtsp',action='store_true'); p.add_argument('--rtsp-host'); p.add_argument('--rtsp-port'); p.add_argument('--rtsp-path'); a=p.parse_args()
os.write(a.event_fd, (json.dumps({'event':'started','config':a.config})+'\\n').encode())
os.close(a.control_fd)
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    from components.vision_yolov8_adjudicator.process import YoloRuntimeProcess
    runtime = YoloRuntimeProcess()
    runtime.start(
        {"runtime": {"binary": str(script), "working_dir": str(tmp_path)}},
        "default",
        prewarm=True,
    )
    try:
        event = next(runtime.events())
        assert event["config"].endswith("vision/yolov8_adjudicator/config.json")
    finally:
        runtime.stop()


def test_runtime_process_forwards_diagnostics_and_reports_exit(tmp_path: Path):
    """An early camera/model exit must be visible instead of becoming a vague timeout."""
    script = tmp_path / "exiting_runtime.py"
    script.write_text(
        """#!/usr/bin/env python3
import argparse, json, os, sys
p=argparse.ArgumentParser(); p.add_argument('--config'); p.add_argument('--control-fd',type=int); p.add_argument('--event-fd',type=int); p.add_argument('--view-id',default='default'); p.add_argument('--no-display',action='store_true'); p.add_argument('--prewarm',action='store_true'); p.add_argument('--rtsp',action='store_true'); p.add_argument('--rtsp-host'); p.add_argument('--rtsp-port'); p.add_argument('--rtsp-path'); a=p.parse_args()
print('camera open failed: /dev/video1', flush=True)
os.write(a.event_fd, (json.dumps({'event':'ready','view_id':a.view_id})+'\\n').encode())
raise SystemExit(7)
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    from components.vision_yolov8_adjudicator.process import YoloRuntimeProcess
    logs = []
    runtime = YoloRuntimeProcess(binary=script)
    runtime.start({}, "front", prewarm=True, on_log=logs.append)
    try:
        events = list(runtime.events())
        assert events[-1] == {"event": "runtime_exit", "returncode": 7}
        assert any("camera open failed" in line for line in logs)
    finally:
        runtime.stop()


def test_runtime_stop_unblocks_reader_waiting_for_events(tmp_path: Path):
    """Stopping a resident runtime must release a blocked events reader."""
    script = tmp_path / "blocked_runtime.py"
    script.write_text(
        """#!/usr/bin/env python3
import argparse, os, time
p=argparse.ArgumentParser(); p.add_argument('--control-fd',type=int); p.add_argument('--event-fd',type=int); p.add_argument('--view-id',default='default'); p.add_argument('--no-display',action='store_true'); p.add_argument('--prewarm',action='store_true'); p.add_argument('--rtsp',action='store_true'); p.add_argument('--rtsp-host'); p.add_argument('--rtsp-port'); p.add_argument('--rtsp-path'); p.add_argument('--snapshot-dir')
a=p.parse_args()
os.close(a.control_fd)
time.sleep(60)
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    from components.vision_yolov8_adjudicator.process import YoloRuntimeProcess

    runtime = YoloRuntimeProcess(binary=script)
    runtime.start({}, "front", prewarm=True)
    reader = threading.Thread(target=lambda: list(runtime.events()), daemon=True)
    reader.start()
    time.sleep(0.1)

    started = time.monotonic()
    runtime.stop()
    elapsed = time.monotonic() - started

    reader.join(timeout=1)
    assert elapsed < 1.5
    assert not reader.is_alive()


def test_provider_sends_final_result_and_stops_resident_runtime(tmp_path: Path):
    image = tmp_path / "stable.jpg"; image.write_bytes(b"jpeg")
    class Runtime:
        def __init__(self): self.commands=[]
        def start(self,*a,**k): self.events_data=iter([{"event":"observation","stable":True,"yolo_outcome":"LEFT","snapshot":{"path":str(image)}}])
        def send(self,c): self.commands.append(c)
        def events(self): return self.events_data
    class Verifier:
        def verify(self, **kwargs): return type("R",(),{"status":"success","outcome":"LEFT","error":None})()
    runtime = Runtime()
    profile={"game_id":"x","vision":{"stable_frames":1},"llm":{"enabled":True,"system_prompt":"s","user_prompt_template":"u","allowed_outcomes":["LEFT"]},"lifecycle":{"post_result_hold_seconds":0}}
    VisionYolov8Adjudicator(runtime_factory=lambda vid: runtime, verifier=Verifier()).adjudicate(VisionAdjudicationRequest("x",profile,"r",2),on_log=lambda x:None,on_event=lambda e:None,is_cancelled=lambda:False)
    assert [c["command"] for c in runtime.commands] == ["START_ADJUDICATION", "FINAL_RESULT", "STOP_ADJUDICATION"]
    assert runtime.commands[1]["outcome"] == {"kind":"winner","value":"LEFT"}
    assert runtime.commands[1]["decision_source"] == "consensus"


def test_provider_reuses_resident_runtime_for_two_rounds_without_stale_observation(tmp_path: Path):
    """A resident camera process must serve consecutive rounds in place."""
    image = tmp_path / "stable.jpg"
    image.write_bytes(b"round-1")

    class Runtime:
        def __init__(self):
            self.commands = []
            self.starts = 0
            self.rounds = 0
            self._events = []

        def start(self, *args, **kwargs):
            self.starts += 1

        def send(self, command):
            self.commands.append(dict(command))
            if command.get("command") == "START_ADJUDICATION":
                self.rounds += 1
                image.write_bytes(f"round-{self.rounds}".encode())
                self._events = iter([{
                    "event": "observation",
                    "stable": True,
                    "yolo_outcome": "LEFT" if self.rounds == 1 else "RIGHT",
                    "snapshot": {"path": str(image)},
                }])

        def events(self):
            return self._events

    class Verifier:
        def __init__(self):
            self.seen = []

        def verify(self, **kwargs):
            paths = kwargs.get("image_paths") or [kwargs["image_path"]]
            self.seen.append(Path(paths[0]).read_bytes())
            return type("R", (), {"status": "success", "outcome": "LEFT", "error": None})()

    runtime = Runtime()
    verifier = Verifier()
    profile = {
        "game_id": "x",
        "vision": {"stable_frames": 1},
        "llm": {
            "enabled": True,
            "system_prompt": "s",
            "user_prompt_template": "u",
            "allowed_outcomes": ["LEFT", "RIGHT"],
        },
        "lifecycle": {"post_result_hold_seconds": 0},
    }
    provider = VisionYolov8Adjudicator(runtime_factory=lambda vid: runtime, verifier=verifier)

    first = provider.adjudicate(
        VisionAdjudicationRequest("x", profile, "round-1", 2),
        on_log=lambda _: None,
        on_event=lambda _: None,
        is_cancelled=lambda: False,
    )
    second = provider.adjudicate(
        VisionAdjudicationRequest("x", profile, "round-2", 2),
        on_log=lambda _: None,
        on_event=lambda _: None,
        is_cancelled=lambda: False,
    )

    assert first["verification"]["yolo_outcome"] == "LEFT"
    assert second["verification"]["yolo_outcome"] == "RIGHT"
    assert runtime.starts == 1
    assert [command["command"] for command in runtime.commands] == [
        "START_ADJUDICATION", "FINAL_RESULT", "STOP_ADJUDICATION",
        "START_ADJUDICATION", "FINAL_RESULT", "STOP_ADJUDICATION",
    ]
    # Round 2's dissent (RIGHT yolo vs LEFT llm) triggers one corroborating
    # re-ask against the same image.
    assert verifier.seen == [b"round-1", b"round-2", b"round-2"]


def test_provider_ignores_stale_idle_event_before_current_round_detection(tmp_path: Path):
    image = tmp_path / "diagnostic.jpg"
    image.write_bytes(b"jpeg")

    class Runtime:
        def start(self, *args, **kwargs):
            self.round = 0

        def send(self, command):
            self.commands = getattr(self, "commands", []) + [dict(command)]
            if command.get("command") == "START_ADJUDICATION":
                self.round += 1

        def events(self):
            # Resident processes can leave the prior round's idle transition
            # queued ahead of the current START phase.
            return iter([
                {"event": "phase", "phase": "idle"},
                {"event": "phase", "phase": "detecting"},
                {"event": "observation", "stable": True, "yolo_outcome": "LEFT",
                 "snapshot": {"path": str(image)}},
            ])

        def stop(self):
            pass

    class Verifier:
        def verify(self, **kwargs):
            return type("R", (), {"status": "success", "outcome": "LEFT"})()

    profile = {
        "game_id": "x",
        "vision": {"stable_frames": 1},
        "llm": {"enabled": True, "system_prompt": "s", "user_prompt_template": "u", "allowed_outcomes": ["LEFT"]},
        "runtime": {"mode": "resident", "prewarm_camera": True},
        "lifecycle": {"post_result_hold_seconds": 0},
    }
    logs = []
    result = VisionYolov8Adjudicator(runtime_factory=lambda _view_id: Runtime(), verifier=Verifier()).adjudicate(
        VisionAdjudicationRequest("x", profile, "stale-idle", 2),
        on_log=logs.append, on_event=lambda _: None, is_cancelled=lambda: False,
    )
    assert not logs, logs
    assert result["outcome"]["value"] == "LEFT"


def test_provider_ignores_stale_cancelled_event_before_current_round_detection(tmp_path: Path):
    """A stale resident CANCEL event must not abort the next round."""
    image = tmp_path / "stable.jpg"
    image.write_bytes(b"jpeg")

    class Runtime:
        def __init__(self):
            self.commands = []
            self.starts = 0

        def start(self, *args, **kwargs):
            self.starts += 1

        def send(self, command):
            self.commands.append(dict(command))

        def events(self):
            # The previous round's CANCEL can remain queued before the next
            # round's START phase reaches the provider-side reader.
            return iter([
                {"event": "cancelled"},
                {"event": "phase", "phase": "detecting"},
                {"event": "progress", "phase": "detecting", "stable_count": 1, "stable_frames": 1},
                {"event": "observation", "stable": True, "yolo_outcome": "LEFT",
                 "snapshot": {"path": str(image)}},
            ])

        def stop(self):
            pass

    class Verifier:
        def verify(self, **kwargs):
            return type("R", (), {"status": "success", "outcome": "LEFT"})()

    profile = {
        "game_id": "x",
        "vision": {"stable_frames": 1},
        "llm": {"enabled": False, "allowed_outcomes": ["LEFT", "RIGHT"]},
        "runtime": {"mode": "resident", "prewarm_camera": True},
        "lifecycle": {"post_result_hold_seconds": 0},
    }
    runtime = Runtime()
    events = []
    result = VisionYolov8Adjudicator(runtime_factory=lambda _view_id: runtime, verifier=Verifier()).adjudicate(
        VisionAdjudicationRequest("x", profile, "stale-cancelled", 2),
        on_log=lambda _: None, on_event=events.append, is_cancelled=lambda: False,
    )

    assert result["outcome"]["value"] == "LEFT"
    assert runtime.starts == 1
    assert any(event.get("stable_count") == 1 for event in events)


def test_provider_reports_incomplete_stable_observation_without_cancelling_resident_runtime(tmp_path: Path):
    """A stable but incomplete dice frame is a retry diagnosis, not a CANCEL error."""
    image = tmp_path / "stable.jpg"
    image.write_bytes(b"jpeg")

    class Runtime:
        def __init__(self):
            self.commands = []

        def start(self, *args, **kwargs):
            pass

        def send(self, command):
            self.commands.append(dict(command))

        def events(self):
            return iter([
                {"event": "phase", "phase": "detecting"},
                {"event": "observation", "stable": True, "yolo_outcome": "LEFT",
                 "snapshot": {"path": str(image)},
                 "participants": {"LEFT": [1, 2, 3, 4], "RIGHT": [6, 6, 6, 6, 6]},
                 "detections": [1] * 9},
            ])

        def stop(self):
            pass

    profile = {
        "game_id": "dice",
        "vision": {"stable_frames": 1, "expected_count": 5, "participants": ["LEFT", "RIGHT"]},
        "rule": {"kind": "numeric_compare", "aggregation": "sum", "higher_wins": True, "tie_value": "TIE"},
        "llm": {"enabled": False, "allowed_outcomes": ["LEFT", "RIGHT", "TIE"]},
        "runtime": {"mode": "resident", "prewarm_camera": True},
        "lifecycle": {"post_result_hold_seconds": 0},
    }
    runtime = Runtime()
    result = VisionYolov8Adjudicator(runtime_factory=lambda _view_id: runtime).adjudicate(
        VisionAdjudicationRequest("dice", profile, "incomplete-stable", 2),
        on_log=lambda _: None, on_event=lambda _: None, is_cancelled=lambda: False,
    )

    assert result["adjudicated"] is False
    assert result["diagnosed"] is True
    assert result["diagnosis"]["reason_code"] == "INCOMPLETE_OBJECTS"
    assert result["diagnosis"]["detected_counts"] == {"LEFT": 4, "RIGHT": 5}
    assert [command["command"] for command in runtime.commands] == [
        "START_ADJUDICATION", "STOP_ADJUDICATION",
    ]


def test_provider_drains_runtime_events_after_yolo_timeout_before_diagnosis(tmp_path: Path):
    image = tmp_path / "diagnostic.jpg"
    image.write_bytes(b"jpeg")

    class Runtime:
        def start(self, *args, **kwargs):
            pass

        def send(self, command):
            self.commands = getattr(self, "commands", []) + [dict(command)]

        def events(self):
            # Simulate the real pipe reader being scheduled just after the
            # provider's detection deadline expires.
            time.sleep(0.08)
            yield {
                "event": "diagnostic_snapshot",
                "stable": False,
                "snapshot": {"path": str(image)},
                "participants": {"LEFT": [1, 2, 3, 4], "RIGHT": [6, 6, 6, 6, 6]},
            }
            yield {"event": "phase", "phase": "detecting"}
            time.sleep(0.08)
            yield {"event": "phase", "phase": "idle"}

        def stop(self):
            pass

    class Verifier:
        def diagnose(self, **kwargs):
            assert kwargs.get("image_paths") or kwargs.get("image_path")
            return type("R", (), {
                "status": "success",
                "reason_code": "OVERLAPPING_OBJECTS",
                "message": "目标可能叠放。",
                "retry": True,
            })()

    profile = {
        "game_id": "x",
        "vision": {"expected_count": 5, "participants": ["LEFT", "RIGHT"]},
        "llm": {
            "enabled": True,
            "system_prompt": "judge",
            "user_prompt_template": "judge",
            "diagnosis_system_prompt": "diagnose",
            "diagnosis_user_prompt_template": "Detector summary: {detector_summary}",
            "allowed_outcomes": ["LEFT", "RIGHT", "TIE"],
            "diagnosis_allowed_reason_codes": ["OVERLAPPING_OBJECTS", "UNKNOWN"],
        },
        "timeouts": {"yolo_detection_seconds": 0.01, "adjudication_seconds": 2},
        "lifecycle": {"post_result_hold_seconds": 0},
    }
    result = VisionYolov8Adjudicator(
        runtime_factory=lambda _view_id: Runtime(), verifier=Verifier()
    ).adjudicate(
        VisionAdjudicationRequest("x", profile, "drain-timeout", 2),
        on_log=lambda _: None, on_event=lambda _: None, is_cancelled=lambda: False,
    )
    assert result["diagnosis"]["source"] == "llm"
    assert result["diagnosis"]["reason_code"] == "OVERLAPPING_OBJECTS"


def test_provider_cancel_keeps_resident_runtime_warm():
    """Cancellation returns a resident worker to idle instead of restarting it."""
    class Runtime:
        def __init__(self):
            self.commands = []
            self.stop_calls = 0

        def start(self, *args, **kwargs):
            pass

        def send(self, command):
            self.commands.append(dict(command))

        def events(self):
            return iter(())

        def stop(self):
            self.stop_calls += 1

    runtime = Runtime()
    profile = {
        "game_id": "x",
        "vision": {"stable_frames": 1},
        "llm": {"enabled": False, "allowed_outcomes": ["LEFT", "RIGHT"]},
        "lifecycle": {"post_result_hold_seconds": 0},
    }
    provider = VisionYolov8Adjudicator(runtime_factory=lambda vid: runtime)
    with pytest.raises(RuntimeError, match="cancelled"):
        provider.adjudicate(
            VisionAdjudicationRequest("x", profile, "cancelled", 1),
            on_log=lambda _: None,
            on_event=lambda _: None,
            is_cancelled=lambda: True,
        )
    assert runtime.stop_calls == 0
    assert [command["command"] for command in runtime.commands] == [
        "START_ADJUDICATION", "CANCEL"
    ]


def test_resident_round_emits_video_before_waiting_for_observation(tmp_path: Path):
    """Every resident round must publish its profile video URL immediately."""
    import threading
    import time

    class Runtime:
        def start(self, *args, **kwargs):
            pass

        def send(self, command):
            pass

        def events(self):
            # Simulate a cached runtime whose one-shot startup video event was
            # already consumed by a previous round.
            time.sleep(0.25)
            return iter([{
                "event": "observation",
                "stable": True,
                "yolo_outcome": "LEFT",
                "snapshot": {"path": str(image)},
            }])

    image = tmp_path / "stable.jpg"
    image.write_bytes(b"resident-round")
    profile = {
        "game_id": "x",
        "vision": {"stable_frames": 1},
        "video": {"path": "/dice/", "webrtc_base_url": "http://100.118.229.28:8889"},
        "llm": {"enabled": False, "allowed_outcomes": ["LEFT", "RIGHT"]},
        "lifecycle": {"post_result_hold_seconds": 0},
        "runtime": {"mode": "resident", "prewarm_camera": True},
    }
    events = []
    detecting = threading.Event()
    video_ready = threading.Event()

    def on_event(event):
        events.append(dict(event))
        if event.get("phase") == "detecting":
            detecting.set()
        if event.get("event") == "video":
            video_ready.set()

    provider = VisionYolov8Adjudicator(runtime_factory=lambda vid: Runtime())
    worker = threading.Thread(
        target=lambda: provider.adjudicate(
            VisionAdjudicationRequest("x", profile, "resident-video", 2),
            on_log=lambda _: None,
            on_event=on_event,
            is_cancelled=lambda: False,
        )
    )
    worker.start()
    assert detecting.wait(1)
    assert video_ready.wait(0.5)
    assert [event["event"] for event in events[:2]] == ["video", "phase"]
    worker.join(2)
    assert not worker.is_alive()


def test_provider_shutdown_stops_and_cleans_resident_runtimes(tmp_path: Path):
    class Runtime:
        def __init__(self):
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1

    runtime = Runtime()
    snapshot_dir = tmp_path / "resident"
    snapshot_dir.mkdir()
    provider = VisionYolov8Adjudicator(runtime_factory=lambda vid: runtime)
    provider._runtime_cache["default"] = runtime
    provider._runtime_snapshot_dirs["default"] = snapshot_dir
    provider._runtime_signatures["default"] = "signature"

    provider.shutdown()

    assert runtime.stop_calls == 1
    assert provider._runtime_cache == {}
    assert provider._runtime_snapshot_dirs == {}
    assert provider._runtime_signatures == {}
    assert not snapshot_dir.exists()


def test_normalize_generic_detections_maps_profile_classes_to_participants():
    profile = {
        "vision": {
            "class_map": {"0": "1", "1": "rock"},
            "participants": ["LEFT", "RIGHT"],
            "participant_assignment": "x_midpoint",
        },
        "rule": {"kind": "categorical_relation"},
    }
    observation = {
        "width": 100,
        "detections": [
            {"class_id": 0, "label": "class_0", "bbox": [10, 0, 30, 20]},
            {"class_id": 1, "label": "class_1", "bbox": [70, 0, 90, 20]},
        ],
    }
    normalized = normalize_observation(profile, observation)
    assert normalized["participants"] == {"LEFT": ["1"], "RIGHT": ["rock"]}


def test_normalize_divider_regions_uses_profile_divider_not_frame_midpoint():
    profile = {
        "vision": {
            "class_map": {"0": "1", "1": "2"},
            "participants": ["LEFT", "RIGHT"],
            "grouping": "divider_regions",
            "divider": {"orientation": "vertical", "position": 0.65},
        },
        "rule": {"kind": "numeric_compare"},
    }
    observation = {
        "width": 100,
        "detections": [
            {"class_id": 0, "bbox": [55, 0, 60, 20]},
            {"class_id": 1, "bbox": [70, 0, 80, 20]},
        ],
    }
    normalized = normalize_observation(profile, observation)
    assert normalized["participants"] == {"LEFT": [1], "RIGHT": [2]}


def test_normalize_numeric_classes_produces_numeric_rule_values():
    profile = {
        "vision": {"class_map": {"0": "6"}, "participants": ["LEFT", "RIGHT"], "participant_assignment": "x_midpoint"},
        "rule": {"kind": "numeric_compare"},
    }
    observation = {"width": 100, "detections": [{"class_id": 0, "bbox": [10, 0, 30, 20]}]}
    assert normalize_observation(profile, observation)["participants"] == {"LEFT": [6], "RIGHT": []}


def test_normalize_generic_detections_skips_unknown_class_ids():
    profile = {"vision": {"class_map": {"0": "1"}, "participants": ["LEFT", "RIGHT"], "participant_assignment": "x_midpoint"}}
    observation = {
        "width": 100,
        "detections": [{"class_id": 9, "label": "class_9", "bbox": [10, 0, 30, 20]}],
    }
    assert "participants" not in normalize_observation(profile, observation)


def test_multiview_missing_yolo_vote_uses_profile_rule_for_all_views(tmp_path: Path):
    image = tmp_path / "stable.jpg"; image.write_bytes(b"jpeg")

    class Runtime:
        def __init__(self, view_id): self.view_id = view_id
        def start(self, *args, **kwargs):
            participants = {"LEFT": [6, 6], "RIGHT": [1, 1]}
            self.events_data = iter([{"event": "observation", "stable": True,
                                      "yolo_outcome": "LEFT" if self.view_id == "front" else None,
                                      "snapshot": {"path": str(image)}, "participants": participants}])
        def send(self, command): pass
        def events(self): return self.events_data
        def stop(self): pass

    profile = {
        "game_id": "dice",
        "vision": {"participants": ["LEFT", "RIGHT"], "stable_frames": 1},
        "rule": {"kind": "numeric_compare", "aggregation": "sum", "higher_wins": True, "tie_value": "TIE"},
        "llm": {"enabled": False, "allowed_outcomes": ["LEFT", "RIGHT", "TIE"]},
        "multi_view": {"enabled": True, "min_views": 2, "views": [{"id": "front"}, {"id": "side"}]},
        "lifecycle": {"post_result_hold_seconds": 0},
    }
    result = VisionYolov8Adjudicator(runtime_factory=lambda vid: Runtime(vid)).adjudicate(
        VisionAdjudicationRequest("dice", profile, "mixed-votes", 2),
        on_log=lambda _: None, on_event=lambda _: None, is_cancelled=lambda: False,
    )
    assert result["outcome"]["value"] == "LEFT"


def test_provider_marks_disabled_llm_as_yolo_only_not_timeout(tmp_path: Path):
    """Disabling verification must not claim a timeout or a call occurred."""
    class Runtime:
        def start(self, *args, **kwargs):
            self.events_data = iter([{
                "event": "observation",
                "stable": True,
                "yolo_outcome": "LEFT",
            }])
        def send(self, command):
            pass
        def events(self):
            return self.events_data

    profile = {
        "game_id": "x",
        "vision": {"participants": ["LEFT", "RIGHT"], "stable_frames": 1},
        "llm": {"enabled": False, "allowed_outcomes": ["LEFT", "RIGHT"]},
        "lifecycle": {"post_result_hold_seconds": 0},
    }
    result = VisionYolov8Adjudicator(runtime_factory=lambda vid: Runtime()).adjudicate(
        VisionAdjudicationRequest("x", profile, "llm-disabled", 2),
        on_log=lambda _: None,
        on_event=lambda _: None,
        is_cancelled=lambda: False,
    )
    assert result["outcome"]["value"] == "LEFT"
    assert result["decision_source"] == "yolo_only"
    assert result["verification"] == {
        "status": "disabled",
        "yolo_outcome": "LEFT",
        "llm_outcome": None,
        "llm_called": False,
        "reask_outcome": None,
        "reask_status": "not_needed",
    }


def test_round_llm_arrives_on_the_request_not_the_constructor(tmp_path: Path):
    """Production LLM engines ride the request; the constructor seam is fallback."""
    image = tmp_path / "stable.jpg"
    image.write_bytes(b"jpeg")

    class Runtime:
        def start(self, *args, **kwargs):
            self.events_data = iter([{
                "event": "observation",
                "stable": True,
                "yolo_outcome": "LEFT",
                "snapshot": {"path": str(image)},
            }])
        def send(self, command):
            pass
        def events(self):
            return self.events_data

    class RequestLlm:
        def __init__(self):
            self.verify_calls = 0
        def verify(self, **kwargs):
            self.verify_calls += 1
            return type("R", (), {"status": "success", "outcome": "LEFT", "error": None})()

    class ConstructorVerifier:
        def verify(self, **kwargs):
            raise AssertionError("request provider must take precedence over the constructor seam")

    request_llm = RequestLlm()
    profile = {
        "game_id": "x",
        "vision": {"participants": ["LEFT", "RIGHT"], "stable_frames": 1},
        "llm": {"enabled": True, "allowed_outcomes": ["LEFT", "RIGHT"]},
        "lifecycle": {"post_result_hold_seconds": 0},
    }
    result = VisionYolov8Adjudicator(
        runtime_factory=lambda vid: Runtime(), verifier=ConstructorVerifier()
    ).adjudicate(
        VisionAdjudicationRequest("x", profile, "request-llm", 2, llm_provider=request_llm),
        on_log=lambda _: None, on_event=lambda _: None, is_cancelled=lambda: False,
    )
    assert request_llm.verify_calls == 1
    assert result["outcome"]["value"] == "LEFT"


def test_missing_llm_provider_disables_verification_not_the_round(tmp_path: Path):
    """Profile enabled + deployment resolved no LLM → yolo_only, never a failure."""
    image = tmp_path / "stable.jpg"
    image.write_bytes(b"jpeg")

    class Runtime:
        def start(self, *args, **kwargs):
            self.events_data = iter([{
                "event": "observation",
                "stable": True,
                "yolo_outcome": "LEFT",
                "snapshot": {"path": str(image)},
            }])
        def send(self, command):
            pass
        def events(self):
            return self.events_data

    profile = {
        "game_id": "x",
        "vision": {"participants": ["LEFT", "RIGHT"], "stable_frames": 1},
        "llm": {"enabled": True, "allowed_outcomes": ["LEFT", "RIGHT"]},
        "lifecycle": {"post_result_hold_seconds": 0},
    }
    logs: list[str] = []
    result = VisionYolov8Adjudicator(runtime_factory=lambda vid: Runtime()).adjudicate(
        VisionAdjudicationRequest("x", profile, "no-llm", 2),
        on_log=logs.append, on_event=lambda _: None, is_cancelled=lambda: False,
    )
    assert result["outcome"]["value"] == "LEFT"
    assert result["decision_source"] == "yolo_only"
    assert result["verification"]["status"] == "disabled"
    assert any("no LLM provider attached" in line for line in logs)
