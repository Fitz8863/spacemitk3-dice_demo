from __future__ import annotations

import json
import os
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
from components.vision_yolov8_adjudicator.provider import (  # noqa: E402
    VisionYolov8Adjudicator,
    normalize_observation,
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


def test_project_result_preserves_frontend_result_contract():
    profile = {"game_id": "dice", "llm": {"allowed_outcomes": ["LEFT", "RIGHT", "TIE"]}}
    decision = finalize_outcome(yolo_outcome="LEFT", llm_outcome="RIGHT", llm_status="success")
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
    profile={"game_id":"dice","vision":{"stable_frames":1},"llm":{"enabled":True,"system_prompt":"s","user_prompt_template":"u","allowed_outcomes":["LEFT","RIGHT","TIE"]},"video":{"path":"/dice/"},"multi_view":{"enabled":False,"min_views":1},"lifecycle":{"post_result_hold_seconds":0}}
    events=[]; verifier=Verifier()
    result=VisionYolov8Adjudicator(runtime_factory=factory, verifier=verifier).adjudicate(VisionAdjudicationRequest("dice",profile,"r1",2),on_log=lambda x:None,on_event=events.append,is_cancelled=lambda:False)
    assert result["decision_source"] == "consensus"; assert verifier.calls == 1
    assert any(r.commands and r.commands[0]["command"] == "START_ADJUDICATION" for r in runtimes)
    video_events = [event for event in events if event.get("event") == "video"]
    assert video_events == [{"event": "video", "url": "http://100.118.229.28:8889/dice/", "view_id": "default"}]


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


def test_runtime_process_uses_dedicated_control_and_event_fds(tmp_path: Path):
    """A noisy runtime must not be able to corrupt structured events."""
    script = tmp_path / "fake_runtime.py"
    script.write_text(
        """#!/usr/bin/env python3
import argparse, json, os, sys
p=argparse.ArgumentParser(); p.add_argument('--control-fd',type=int); p.add_argument('--event-fd',type=int); p.add_argument('--snapshot-dir',default='/tmp'); p.add_argument('--view-id',default='default'); p.add_argument('--prewarm',action='store_true'); p.add_argument('--no-display',action='store_true'); p.add_argument('--rtsp',action='store_true'); p.add_argument('--rtsp-host'); p.add_argument('--rtsp-port'); p.add_argument('--rtsp-path'); a=p.parse_args()
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


def test_runtime_process_forwards_diagnostics_and_reports_exit(tmp_path: Path):
    """An early camera/model exit must be visible instead of becoming a vague timeout."""
    script = tmp_path / "exiting_runtime.py"
    script.write_text(
        """#!/usr/bin/env python3
import argparse, json, os, sys
p=argparse.ArgumentParser(); p.add_argument('--control-fd',type=int); p.add_argument('--event-fd',type=int); p.add_argument('--view-id',default='default'); p.add_argument('--no-display',action='store_true'); p.add_argument('--prewarm',action='store_true'); p.add_argument('--rtsp',action='store_true'); p.add_argument('--rtsp-host'); p.add_argument('--rtsp-port'); p.add_argument('--rtsp-path'); a=p.parse_args()
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
    assert verifier.seen == [b"round-1", b"round-2"]


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
    }
