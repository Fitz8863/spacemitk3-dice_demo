"""多游戏视觉架构：同一套裁决器服务多个游戏时的隔离契约。

本文件针对的缺陷是「跨游戏参数串味」：resident runtime 的缓存键是 ``view_id``
（各游戏都是 ``"default"``），而重启判据 ``_runtime_signature`` 曾经只覆盖
model/stable_frames/confidence/camera/runtime —— 于是两个游戏只要这几项相同，
就会**复用同一个已缓存的 runtime 进程**，拿到上一个游戏的分界线门控、每侧数量
与 RTSP 路径。这些用例就是那条契约的守卫。
"""
from __future__ import annotations

import json as _json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.vision import VisionAdjudicationRequest  # noqa: E402
from components.vision_yolov8_adjudicator.provider import (  # noqa: E402
    VisionYolov8Adjudicator,
)


def _profile(
    game_id: str,
    *,
    divider_detection: bool = True,
    grouping: str = "divider_regions",
    expected_count: int | None = None,
    video_path: str = "/dice/det",
) -> dict:
    """A resident profile; the shared fields stay identical across games.

    Only the fields a game genuinely owns differ, which is exactly the shape
    that used to collide: same model / stable frames / camera / confidence.
    """
    vision: dict = {
        "model": "vision/yolov8_adjudicator/models/best.q.onnx",
        "stable_frames": 30,
        "participants": ["LEFT", "RIGHT"],
        "divider_detection": divider_detection,
        "grouping": grouping,
    }
    if expected_count is not None:
        vision["expected_count"] = expected_count
    return {
        "schema_version": 1,
        "game_id": game_id,
        "runtime": {"mode": "resident", "prewarm_camera": True},
        "vision": vision,
        "llm": {"enabled": False, "allowed_outcomes": ["LEFT", "RIGHT"]},
        "lifecycle": {"post_result_hold_seconds": 0},
        "video": {"enabled": True, "path": video_path},
        "timeouts": {"adjudication_seconds": 10},
    }


# ---- the restart verdict itself -------------------------------------------

SIGNATURE_CASES = [
    # (字段说明, 游戏 A 的覆盖, 游戏 B 的覆盖)
    ("game_id", {"game_id": "dice"}, {"game_id": "rps"}),
    ("divider_detection", {"divider_detection": True}, {"divider_detection": False}),
    ("grouping", {"grouping": "divider_regions"}, {"grouping": "x_midpoint"}),
    ("expected_count", {"expected_count": 5}, {"expected_count": None}),
    ("video_path", {"video_path": "/dice/det"}, {"video_path": "/rps/"}),
]


@pytest.mark.parametrize("label,overrides_a,overrides_b", SIGNATURE_CASES)
def test_runtime_signature_separates_each_game_owned_field(label, overrides_a, overrides_b):
    """任一「游戏自有」字段不同，签名就必须不同——否则会复用错进程。"""
    profile_a = _profile(**{"game_id": "dice", **overrides_a})
    profile_b = _profile(**{"game_id": "rps", **overrides_b})
    signature_a = VisionYolov8Adjudicator._runtime_signature(profile_a, "default")
    signature_b = VisionYolov8Adjudicator._runtime_signature(profile_b, "default")
    assert signature_a != signature_b, f"{label} 没有参与签名"


def test_runtime_signature_is_stable_for_the_same_profile():
    """同一份 profile 必须给出同一签名，否则每回合都会白重建 runtime。"""
    profile = _profile("dice")
    first = VisionYolov8Adjudicator._runtime_signature(profile, "default")
    second = VisionYolov8Adjudicator._runtime_signature(_profile("dice"), "default")
    assert first == second


# ---- end to end: a second game must not inherit the first game's runtime ---

class _ObservingRuntime:
    """Runtime double that serves one stable observation and records lifecycle."""

    def __init__(self, view_id: str, snapshot_path: str) -> None:
        self.view_id = view_id
        self.commands: list[dict] = []
        self.start_calls = 0
        self.stop_calls = 0
        self.start_kwargs: dict = {}
        self._snapshot_path = snapshot_path
        self._events: list[dict] = []

    def start(self, *args, **kwargs):
        self.start_calls += 1
        self.start_kwargs = dict(kwargs)
        self._events = [{
            "event": "observation",
            "stable": True,
            "yolo_outcome": "LEFT",
            "snapshot": {"path": self._snapshot_path},
        }]

    def send(self, command):
        self.commands.append(dict(command))

    def events(self):
        return iter(self._events)

    def stop(self):
        self.stop_calls += 1


def _adjudicate(provider, profile, request_id, tmp_path, index):
    snapshot = tmp_path / f"stable-{index}.jpg"
    snapshot.write_bytes(b"jpeg")
    return provider.adjudicate(
        VisionAdjudicationRequest(profile["game_id"], profile, request_id, 10),
        on_log=lambda _line: None,
        on_event=lambda _event: None,
        is_cancelled=lambda: False,
    )


def test_second_game_rebuilds_the_runtime_instead_of_reusing_it(tmp_path: Path):
    """游戏 B 的裁决参数与 A 不同时，必须拆掉 A 的进程再建 B 的。

    它们在 model / stable_frames / camera / confidence 上完全一致——正是过去
    会导致签名相同、直接复用旧进程的形状。
    """
    created: list[_ObservingRuntime] = []

    def factory(view_id="default"):
        runtime = _ObservingRuntime(
            view_id, str(tmp_path / f"snap-{len(created)}.jpg")
        )
        (tmp_path / f"snap-{len(created)}.jpg").write_bytes(b"jpeg")
        created.append(runtime)
        return runtime

    provider = VisionYolov8Adjudicator(runtime_factory=factory)
    profile_a = _profile("dice", divider_detection=True, video_path="/dice/det")
    profile_b = _profile("rps", divider_detection=False, video_path="/rps/")

    result_a = _adjudicate(provider, profile_a, "round-a", tmp_path, 0)
    assert result_a["outcome"]["value"] == "LEFT"
    assert len(created) == 1

    result_b = _adjudicate(provider, profile_b, "round-b", tmp_path, 1)
    assert result_b["outcome"]["value"] == "LEFT"

    # The second game must get its own process with its own parameters.
    assert len(created) == 2, "第二个游戏复用了上一个游戏的 runtime"
    assert created[0].stop_calls == 1, "被替换掉的 runtime 没有释放"
    assert created[1].start_calls == 1
    # B received B's profile (its own divider gate and RTSP path).
    assert created[1]._snapshot_path != created[0]._snapshot_path


def test_same_game_reuses_its_runtime_across_rounds(tmp_path: Path):
    """"再来一局"必须复用进程——重建会让每回合都付摄像头+模型加载。"""
    created: list[_ObservingRuntime] = []

    def factory(view_id="default"):
        runtime = _ObservingRuntime(
            view_id, str(tmp_path / f"reuse-{len(created)}.jpg")
        )
        (tmp_path / f"reuse-{len(created)}.jpg").write_bytes(b"jpeg")
        created.append(runtime)
        return runtime

    provider = VisionYolov8Adjudicator(runtime_factory=factory)
    profile = _profile("dice")

    _adjudicate(provider, profile, "round-1", tmp_path, 0)
    _adjudicate(provider, profile, "round-2", tmp_path, 1)

    assert len(created) == 1, "同一游戏跨回合不应重建 runtime"
    assert created[0].stop_calls == 0


# ---- editing the shared runtime config must reach the runtime --------------

def test_runtime_signature_changes_when_the_runtime_config_changes(tmp_path: Path, monkeypatch):
    """改 vision/yolov8_adjudicator/config.json 的 conf/zoom 后下一回合应重建。

    该文件只通过 ``--config`` 影响 runtime；此前改它必须重启后端，与
    manifest 热加载的语义不一致。签名纳入该文件的 mtime+size 后，
    保存即生效。
    """
    import components.vision_yolov8_adjudicator.provider as vision_provider

    config_file = tmp_path / "runtime-config.json"
    config_file.write_text('{"conf": 0.45}\n', encoding="utf-8")
    monkeypatch.setattr(
        vision_provider,
        "resolve_runtime_config_path",
        lambda _component, **_kwargs: config_file,
    )

    profile = _profile("dice")
    before = VisionYolov8Adjudicator._runtime_signature(profile, "default")

    # Same path, unchanged file: the verdict must stay stable.
    assert VisionYolov8Adjudicator._runtime_signature(profile, "default") == before

    # New size (and mtime) => different verdict.
    config_file.write_text('{"conf": 0.30, "zoom": 150}\n', encoding="utf-8")
    after = VisionYolov8Adjudicator._runtime_signature(profile, "default")
    assert after != before, "runtime config 变化没有触发重建"


def test_runtime_signature_survives_an_unreadable_runtime_config(monkeypatch):
    """读不到 runtime config 时降级为空指纹，绝不因此拒绝启动。"""
    import components.vision_yolov8_adjudicator.provider as vision_provider

    def _boom(_component, **_kwargs):
        raise OSError("no such file")

    monkeypatch.setattr(vision_provider, "resolve_runtime_config_path", _boom)
    profile = _profile("dice")
    # Still produces a usable signature instead of raising.
    signature = VisionYolov8Adjudicator._runtime_signature(profile, "default")
    assert "dice" in signature


# ---- the pipeline itself is game-agnostic ---------------------------------

class _FakeAdjudicator:
    """Adjudicator double that returns one physical result per game."""

    def __init__(self, physical):
        self.physical = physical
        self.requests = []

    def adjudicate(self, request, **_kwargs):
        self.requests.append(request)
        return dict(self.physical)


class _FakeRegistry:
    def __init__(self, providers):
        self._providers = providers

    def require(self, provider_id, *, expected_type=None, expected_role=None):
        from core.errors import ComponentNotFoundError

        provider = self._providers.get(provider_id)
        if provider is None:
            raise ComponentNotFoundError(provider_id)
        return provider


def test_shared_pipeline_serves_a_non_dice_game():
    """通用 pipeline 必须能跑非 dice 游戏——这正是「加游戏不用改 core」的证据。"""
    from core.vision_pipeline import run_vision_game

    adjudicator = _FakeAdjudicator({
        "winner": "LEFT", "left_values": [2], "right_values": [1],
    })
    components = _FakeRegistry({"vision_yolov8_adjudicator": adjudicator})
    manifest = {
        "providers": {"vision_adjudicator": "vision_yolov8_adjudicator"},
        "participants": {"player": "LEFT", "agent": "RIGHT"},
        "vision_profile": _profile("rps"),
    }
    seen: list[str] = []

    result = run_vision_game(
        lambda line: seen.append(line),
        lambda: False,
        5.0,
        components=components,
        manifest=manifest,
        on_event=lambda _event: None,
        game_id="rps",
        projector=lambda physical, participants: {
            "winner": physical["winner"], "projected_for": "rps",
        },
    )
    assert result == {"winner": "LEFT", "projected_for": "rps"}
    # The request carries the game's own id and profile.
    assert adjudicator.requests[0].game_id == "rps"


def test_shared_pipeline_rejects_a_profile_for_another_game():
    """profile 的 game_id 与调用游戏不符时必须拒绝，不能拿骰子的参数跑猜拳。"""
    from core.vision_pipeline import run_vision_game

    components = _FakeRegistry({"vision_yolov8_adjudicator": _FakeAdjudicator({})})
    manifest = {
        "providers": {"vision_adjudicator": "vision_yolov8_adjudicator"},
        "participants": {"player": "LEFT", "agent": "RIGHT"},
        "vision_profile": _profile("dice"),
    }
    with pytest.raises(ValueError, match="vision profile"):
        run_vision_game(
            lambda _line: None, lambda: False, 5.0,
            components=components, manifest=manifest,
            on_event=lambda _event: None, game_id="rps",
            projector=lambda physical, participants: physical,
        )


def test_shared_pipeline_passes_a_diagnosed_result_straight_through():
    """诊断结果是终态重试结论，不能被游戏的结果投影层改写。"""
    from core.vision_pipeline import run_vision_game

    diagnosis = {"diagnosed": True, "diagnosis": {"reason_code": "NO_OBJECTS_DETECTED"}}
    components = _FakeRegistry(
        {"vision_yolov8_adjudicator": _FakeAdjudicator(diagnosis)}
    )
    manifest = {
        "providers": {"vision_adjudicator": "vision_yolov8_adjudicator"},
        "participants": {"player": "LEFT", "agent": "RIGHT"},
        "vision_profile": _profile("dice"),
    }

    def _projector(_physical, _participants):
        raise AssertionError("诊断结果不应进入投影层")

    result = run_vision_game(
        lambda _line: None, lambda: False, 5.0,
        components=components, manifest=manifest,
        on_event=lambda _event: None, game_id="dice", projector=_projector,
    )
    assert result == diagnosis


# ---- categorical projection: the second kind of vision game ---------------

def _categorical_result():
    return {
        "winner": "LEFT",
        "left_choice": "rock",
        "right_choice": "scissors",
        "outcome": {"value": "LEFT"},
    }


def test_categorical_projection_maps_choices_onto_roles():
    """猜拳这类「每侧一个类别」的游戏：投影出 player/agent 的选择，且不要求数字。"""
    from core.participants import project_categorical_result

    projected = project_categorical_result(
        _categorical_result(), {"player": "LEFT", "agent": "RIGHT"}
    )
    assert projected["winner_role"] == "PLAYER"
    assert projected["player_choice"] == "rock"
    assert projected["agent_choice"] == "scissors"
    # Physical fields stay untouched for compatibility.
    assert projected["left_choice"] == "rock"
    assert projected["right_choice"] == "scissors"
    # Crucially: no numeric evidence was required.
    assert "left_values" not in projected


def test_categorical_projection_follows_the_participant_mapping():
    """摆位对调后，选择必须跟着换边——这是机械臂联调会碰到的路径。"""
    from core.participants import project_categorical_result

    projected = project_categorical_result(
        _categorical_result(), {"player": "RIGHT", "agent": "LEFT"}
    )
    assert projected["winner_role"] == "AGENT"
    assert projected["player_choice"] == "scissors"
    assert projected["agent_choice"] == "rock"


def test_categorical_projection_maps_a_tie():
    from core.participants import project_categorical_result

    result = {
        "winner": "TIE", "left_choice": "paper", "right_choice": "paper",
    }
    projected = project_categorical_result(
        result, {"player": "LEFT", "agent": "RIGHT"}
    )
    assert projected["winner_role"] == "TIE"
    assert projected["player_choice"] == projected["agent_choice"] == "paper"


def test_categorical_projection_supports_a_custom_choice_field():
    from core.participants import project_categorical_result

    result = {"winner": "RIGHT", "left_gesture": "rock", "right_gesture": "paper"}
    projected = project_categorical_result(
        result, {"player": "LEFT", "agent": "RIGHT"}, choice_field="gesture"
    )
    assert projected["player_choice"] == "rock"
    assert projected["agent_choice"] == "paper"


@pytest.mark.parametrize("bad", [None, "", "   ", 3])
def test_categorical_projection_rejects_missing_or_non_string_choices(bad):
    from core.participants import project_categorical_result

    result = {"winner": "LEFT", "left_choice": bad, "right_choice": "scissors"}
    with pytest.raises(ValueError, match="left_choice"):
        project_categorical_result(result, {"player": "LEFT", "agent": "RIGHT"})


def test_categorical_projection_rejects_an_inconsistent_verdict():
    """结果自相矛盾（winner 与 outcome.value 不一致）必须拒绝。"""
    from core.participants import project_categorical_result

    result = _categorical_result()
    result["outcome"] = {"value": "RIGHT"}
    with pytest.raises(ValueError, match="outcome.value"):
        project_categorical_result(result, {"player": "LEFT", "agent": "RIGHT"})


# ---- health must describe the deployed game, not a hardcoded one -----------

def test_primary_vision_game_prefers_the_enabled_vision_game(monkeypatch):
    """dice 停用、另一个视觉游戏启用时，健康元数据必须取自那个游戏。"""
    import server

    class _Registry:
        def all(self):
            return [
                {"id": "dice", "enabled": False, "vision_profile": {"game_id": "dice"}},
                {"id": "rps", "enabled": True, "vision_profile": {"game_id": "rps"}},
            ]

    monkeypatch.setattr(server, "get_games", lambda: _Registry())
    assert server._primary_vision_game_id() == "rps"


def test_primary_vision_game_skips_games_without_a_vision_profile(monkeypatch):
    import server

    class _Registry:
        def all(self):
            return [
                {"id": "board_only", "enabled": True},
                {"id": "rps", "enabled": True, "vision_profile": {"game_id": "rps"}},
            ]

    monkeypatch.setattr(server, "get_games", lambda: _Registry())
    assert server._primary_vision_game_id() == "rps"


def test_primary_vision_game_falls_back_to_dice(monkeypatch):
    """一个视觉游戏都没有时保持历史值，不退化成空串。"""
    import server

    class _Empty:
        def all(self):
            return []

    monkeypatch.setattr(server, "get_games", lambda: _Empty())
    assert server._primary_vision_game_id() == "dice"


def test_primary_vision_game_uses_a_disabled_game_when_that_is_all_there_is(monkeypatch):
    """全是停用游戏时，报出那个游戏总好过报一个不存在的 dice。"""
    import server

    class _Registry:
        def all(self):
            return [{"id": "rps", "enabled": False, "vision_profile": {"game_id": "rps"}}]

    monkeypatch.setattr(server, "get_games", lambda: _Registry())
    assert server._primary_vision_game_id() == "rps"


# ---- detection threshold belongs to the game, not the shared runtime config

def test_dice_declares_its_own_confidence_threshold():
    """检测阈值是游戏参数：骰子（小方块）与手势需要的阈值不同。

    此前 `conf: 0.45` 只写在共享的 `vision/yolov8_adjudicator/config.json` 里，
    是唯一一个「游戏自有却不在 manifest」的裁决参数——多游戏架构下会被所有
    游戏继承。现在它归 dice 自己的 `vision_profile.vision.confidence`。
    """
    import json as _json

    manifest = _json.loads(
        (ROOT / "backend/games/dice/manifest.json").read_text(encoding="utf-8")
    )
    confidence = manifest["vision_profile"]["vision"].get("confidence")
    assert isinstance(confidence, (int, float)) and 0 < confidence < 1

    runtime_config = _json.loads(
        (ROOT / "vision/yolov8_adjudicator/config.json").read_text(encoding="utf-8")
    )
    # The shared runtime config is hardware/deployment only now; a game-owned
    # threshold must not reappear there, or it silently applies to every game.
    assert "conf" not in runtime_config


def test_confidence_reaches_the_runtime_command_line():
    """manifest 的 confidence 必须真的转发成 --conf，否则只是装饰。

    转发逻辑内联在 ``YoloRuntimeProcess.start()`` 里，没有可单独调用的辅助
    函数，因此这里断言源码里的转发契约（仓库既有测试也用这种源码断言方式）。
    """
    source = (ROOT / "backend/components/vision_yolov8_adjudicator/process.py").read_text(
        encoding="utf-8"
    )
    # Reads the manifest field (confidence first, legacy conf spelling second)...
    assert 'vision.get("confidence", vision.get("conf"))' in source
    # ...and forwards it as the runtime's --conf override.
    assert '"--conf"' in source


# ---- per-game hardware runtime config --------------------------------------

def _write_runtime_config(path: Path, **overrides) -> Path:
    payload = {
        "camera": "/dev/video1",
        "width": 1280,
        "height": 720,
        "fps": 25,
        "intra_threads": 2,
        "ep_affinity": "14;15",
        "focus": -1,
        "zoom": 150,
        "rtsp": {"enabled": True, "host": "127.0.0.1", "port": 8554},
        "video": {"webrtc_base_url": "http://127.0.0.1:8889"},
    }
    payload.update(overrides)
    path.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_a_game_can_point_at_its_own_hardware_config(tmp_path: Path):
    """游戏 manifest 可以声明自己的硬件配置文件，且解析优先于共享默认。"""
    from components.vision_yolov8_adjudicator.profile import resolve_runtime_config_path

    own = _write_runtime_config(tmp_path / "rps-runtime.json", camera="/dev/video3")
    component = {"runtime": {"config": "vision/yolov8_adjudicator/config.json"}}

    shared = resolve_runtime_config_path(component)
    assert shared.name == "config.json"

    # A relative path is resolved against the repository root, so the test
    # reaches the file through a path relative to ROOT.
    rel = own.relative_to(ROOT) if own.is_relative_to(ROOT) else None
    if rel is None:
        # tmp_path is outside the repo: assert the rejection instead.
        with pytest.raises(Exception):
            resolve_runtime_config_path(component, profile={"runtime_config": str(own)})
        return
    resolved = resolve_runtime_config_path(component, profile={"runtime_config": str(rel)})
    assert resolved == own.resolve()


def test_per_game_runtime_config_must_stay_inside_the_project():
    """绝对路径与 .. 越界必须被拒绝——否则游戏能把运行时指向任意文件。"""
    from components.vision_yolov8_adjudicator.profile import resolve_runtime_config_path

    component = {"runtime": {"config": "vision/yolov8_adjudicator/config.json"}}
    for bad in ("/etc/passwd", "../../etc/passwd", "backend/../../outside.json"):
        with pytest.raises(Exception):
            resolve_runtime_config_path(component, profile={"runtime_config": bad})


def test_profile_validation_accepts_and_shape_checks_runtime_config():
    """runtime_config 是可选字段；声明了就必须是仓库内相对路径。"""
    from components.vision_yolov8_adjudicator.profile import ProfileError, validate_profile

    def valid_profile():
        """A profile that passes full validation (the shared helper is minimal)."""
        p = _profile("dice")
        p["vision"]["class_map"] = {"0": "1"}
        p["vision"]["participants"] = ["LEFT", "RIGHT"]
        p["llm"] = {
            "enabled": False,
            "context_mode": "single_turn_no_history",
            "system_prompt": "judge",
            "user_prompt_template": "judge",
            "allowed_outcomes": ["LEFT", "RIGHT"],
        }
        p["video"] = {"enabled": True, "path": "/dice/det"}
        return p

    # Optional: a game with no hardware differences simply omits it.
    validate_profile(valid_profile())

    # Declared: accepted when it is a repository-relative path.
    declared = valid_profile()
    declared["runtime_config"] = "vision/yolov8_adjudicator/config.json"
    validate_profile(declared)

    # Rejected shapes: empty, non-string, absolute, and traversal.
    for bad in ("", "   ", 5, "/abs/path.json", "../escape.json"):
        broken = valid_profile()
        broken["runtime_config"] = bad
        with pytest.raises(ProfileError):
            validate_profile(broken)


def test_declared_per_game_config_is_never_silently_dropped(tmp_path: Path):
    """声明了自己配置文件的游戏，解析失败必须硬报错。

    否则 `--config` 根本不会被传，C++ 会去读工作目录下的 config.json ——
    也就是悄悄用了别的游戏的摄像头与 RTSP 设置。
    """
    import components.vision_yolov8_adjudicator.process as process

    captured = {}

    class _FakePopen:
        def __init__(self, cmd, **kwargs):
            captured["cmd"] = list(cmd)
            self.stdout = None
            self.pid = 1
            self.returncode = None

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    original = process.subprocess.Popen
    process.subprocess.Popen = _FakePopen
    try:
        runtime = process.YoloRuntimeProcess(binary="/bin/true", working_dir=str(ROOT))
        profile = _profile("dice")
        profile["video"] = {"enabled": True, "path": "/dice/det"}
        profile["runtime_config"] = "backend/games/dice/does-not-exist.json"
        with pytest.raises(Exception):
            runtime.start(profile, "default", prewarm=True)
    finally:
        process.subprocess.Popen = original


def test_declared_per_game_config_reaches_the_command_line(tmp_path: Path):
    """声明成功时，命令行里的 --config 必须指向那个文件。"""
    import components.vision_yolov8_adjudicator.process as process

    own_dir = ROOT / "backend" / "games" / "dice"
    own = own_dir / "_tmp_probe_runtime.json"
    own.write_text(_json.dumps({"camera": "/dev/video7"}), encoding="utf-8")
    captured = {}

    class _FakePopen:
        def __init__(self, cmd, **kwargs):
            captured["cmd"] = list(cmd)
            self.stdout = None
            self.pid = 1
            self.returncode = None

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    original = process.subprocess.Popen
    process.subprocess.Popen = _FakePopen
    try:
        runtime = process.YoloRuntimeProcess(binary="/bin/true", working_dir=str(ROOT))
        profile = _profile("dice")
        profile["video"] = {"enabled": True, "path": "/dice/det"}
        profile["runtime_config"] = "backend/games/dice/_tmp_probe_runtime.json"
        runtime.start(profile, "default", prewarm=True)
        cmd = captured["cmd"]
        assert "--config" in cmd
        assert cmd[cmd.index("--config") + 1] == str(own.resolve())
    finally:
        process.subprocess.Popen = original
        own.unlink(missing_ok=True)
