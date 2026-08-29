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
from components.vision_yolov8_adjudicator.llm import (  # noqa: E402
    OpenAICompatibleVisionVerifier,
)
from components.vision_yolov8_adjudicator.process import (  # noqa: E402
    SnapshotError,
    verify_snapshot,
)
from components.vision_yolov8_adjudicator.provider import VisionYolov8Adjudicator  # noqa: E402
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


def test_llm_request_is_single_turn_with_image(tmp_path: Path):
    image = tmp_path / "stable.jpg"
    image.write_bytes(b"jpeg-bytes")
    captured = {}

    def fake_post(url, payload, headers, timeout):
        captured.update(payload)
        return {"choices": [{"message": {"content": '{"winner":"LEFT"}'}}]}

    verifier = OpenAICompatibleVisionVerifier(post=fake_post)
    result = verifier.verify(
        image_path=image,
        system_prompt="Judge the image.",
        user_prompt="Return JSON.",
        allowed_outcomes=["LEFT", "RIGHT", "TIE"],
        timeout_seconds=3,
    )
    assert result.outcome == "LEFT"
    assert result.status == "success"
    assert len(captured["messages"]) == 2
    assert captured["messages"][0] == {"role": "system", "content": "Judge the image."}
    user_content = captured["messages"][1]["content"]
    assert user_content[0] == {"type": "text", "text": "Return JSON."}
    assert user_content[1]["type"] == "image_url"
    assert user_content[1]["image_url"]["url"] == "data:image/jpeg;base64,anBlZy1ieXRlcw=="


def test_llm_timeout_is_distinguished_from_failure(tmp_path: Path):
    image = tmp_path / "stable.png"
    image.write_bytes(b"png-bytes")

    def timeout_post(url, payload, headers, timeout):
        raise TimeoutError("deadline exceeded")

    result = OpenAICompatibleVisionVerifier(post=timeout_post).verify(
        image_path=image,
        system_prompt="Judge.",
        user_prompt="Return JSON.",
        allowed_outcomes=["LEFT"],
        timeout_seconds=0.1,
    )
    assert result.status == "timeout"
    assert result.outcome is None


def test_provider_runs_one_round_and_holds_result(tmp_path: Path):
    image = tmp_path / "stable.jpg"; image.write_bytes(b"jpeg")
    class Runtime:
        def __init__(self, view_id="default"):
            self.commands = []; self.view_id = view_id
        def start(self, profile, view_id, prewarm=True): self.events_data = iter([
            {"event":"started","phase":"starting"}, {"event":"ready","phase":"idle"},
            {"event":"video","url":"http://x/dice/"},
            {"event":"observation","stable":True,"yolo_outcome":"LEFT","snapshot":{"path":str(image)},"participants":{"LEFT":[6],"RIGHT":[1]}},
        ])
        def send(self, command): self.commands.append(command)
        def events(self): return self.events_data
        def stop(self): pass
    runtimes=[]
    def factory(view_id="default"):
        r=Runtime(view_id); runtimes.append(r); return r
    class Verifier:
        def __init__(self): self.calls=0
        def verify(self, **kwargs):
            self.calls += 1; return type("R", (), {"status":"success","outcome":"LEFT","error":None})()
    profile={"game_id":"dice","vision":{"stable_frames":1},"llm":{"enabled":True,"system_prompt":"s","user_prompt_template":"u","allowed_outcomes":["LEFT","RIGHT","TIE"]},"multi_view":{"enabled":False,"min_views":1},"lifecycle":{"post_result_hold_seconds":0}}
    events=[]; verifier=Verifier()
    result=VisionYolov8Adjudicator(runtime_factory=factory, verifier=verifier).adjudicate(VisionAdjudicationRequest("dice",profile,"r1",2),on_log=lambda x:None,on_event=events.append,is_cancelled=lambda:False)
    assert result["decision_source"] == "consensus"; assert verifier.calls == 1
    assert any(r.commands and r.commands[0]["command"] == "START_ADJUDICATION" for r in runtimes)


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
        def start(self, *args, **kwargs):
            self.events_data = iter([{"event":"observation", "stable":True, "snapshot":{"path":"/tmp/no.jpg"}, "participants":{"LEFT":[1],"RIGHT":[2,3]}}])
        def send(self, command): pass
        def events(self): return self.events_data
        def stop(self): pass
    profile={"game_id":"x","vision":{"stable_frames":1,"expected_count":2},"rule":{"kind":"numeric_compare","aggregation":"sum","higher_wins":True,"tie_value":"TIE"},"llm":{"enabled":False,"allowed_outcomes":["LEFT","RIGHT","TIE"]},"lifecycle":{"post_result_hold_seconds":0}}
    with pytest.raises(RuleError, match="expected_count"):
        VisionYolov8Adjudicator(runtime_factory=lambda vid: Runtime()).adjudicate(VisionAdjudicationRequest("x",profile,"r",2),on_log=lambda x:None,on_event=lambda e:None,is_cancelled=lambda:False)


@pytest.mark.parametrize("content", ["not-json", '{"winner":"UNKNOWN"}', '{"winner":1}'])
def test_llm_invalid_or_unknown_response_is_failure(tmp_path: Path, content: str):
    image = tmp_path / "stable.jpg"
    image.write_bytes(b"jpeg-bytes")

    def fake_post(url, payload, headers, timeout):
        return {"choices": [{"message": {"content": content}}]}

    result = OpenAICompatibleVisionVerifier(post=fake_post).verify(
        image_path=image,
        system_prompt="Judge.",
        user_prompt="Return JSON.",
        allowed_outcomes=["LEFT", "RIGHT", "TIE"],
        timeout_seconds=1,
    )
    assert result.status == "failure"
    assert result.outcome is None


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
