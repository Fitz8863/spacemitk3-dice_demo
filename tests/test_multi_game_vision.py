"""多游戏视觉架构：同一套裁决器服务多个游戏时的隔离契约。

本文件针对的缺陷是「跨游戏参数串味」：resident runtime 的缓存键是 ``view_id``
（各游戏都是 ``"default"``），而重启判据 ``_runtime_signature`` 曾经只覆盖
model/stable_frames/confidence/camera/runtime —— 于是两个游戏只要这几项相同，
就会**复用同一个已缓存的 runtime 进程**，拿到上一个游戏的分界线门控、每侧数量
与 RTSP 路径。这些用例就是那条契约的守卫。
"""
from __future__ import annotations

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
