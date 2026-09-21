"""Contract tests for the YOLOv10 vision package (component wiring + observe)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from components.vision_yolov10_objdetect.provider import (  # noqa: E402
    VisionYolov10Objdetect,
    resolve_runtime_binary,
    COMPONENT_DIR as V10_COMPONENT_DIR,
)
from components.vision_yolov10_objdetect.process import YoloRuntimeProcess  # noqa: E402
from core.vision import VisionAdjudicationRequest  # noqa: E402
from core import games as games_module  # noqa: E402


def test_v10_binary_resolves_to_the_v10_package():
    """The v10 provider's default binary is the v10 runtime, not the v8 one."""
    binary = resolve_runtime_binary(component_dir=V10_COMPONENT_DIR)
    assert binary.name == "yolov10_camera"
    assert "yolov10_objdetect" in str(binary)
    assert "yolov8_objdetect" not in str(binary)


def test_v10_health_reports_the_v10_binary():
    payload = VisionYolov10Objdetect().health()
    assert payload["id"] == "vision_yolov10_objdetect"
    assert "yolov10_objdetect" in payload["binary"]
    assert payload["binary"].endswith("yolov10_camera")


def test_v10_runtime_defaults_come_from_the_v10_config():
    adapter = YoloRuntimeProcess()
    assert adapter.binary.endswith("yolov10_camera")
    assert "yolov10_objdetect" in adapter.binary
    if adapter.working_dir:
        assert adapter.working_dir.endswith("yolov10_objdetect")


def _stable_observation(label: str, class_id: int) -> dict:
    return {
        "event": "observation",
        "view_id": "default",
        "stable": True,
        "width": 1080,
        "height": 1920,
        "detections": [
            {"class_id": class_id, "label": label, "confidence": 0.9,
             "bbox": [10, 20, 30, 40]}
        ],
    }


class _FakeRuntime:
    def __init__(self, events):
        self._events = list(events)
        self.commands: list[dict] = []
        self.stop_calls = 0

    def start(self, *args, **kwargs):
        return None

    def send(self, command):
        self.commands.append(dict(command))

    def events(self):
        return iter(self._events)

    def stop(self):
        self.stop_calls += 1


class TestObserveContract:
    """observe() is the single-view evidence entry the rps pipeline uses."""

    def _profile(self) -> dict:
        return {
            "game_id": "rps",
            "vision": {"stable_frames": 1},
            "lifecycle": {"pre_adjudication_wait_seconds": 0},
            "timeouts": {"yolo_detection_seconds": 2, "adjudication_seconds": 5},
            "multi_view": {"enabled": True, "min_views": 1},
        }

    def test_observe_returns_folded_detections_and_stops_inference(self):
        runtime = _FakeRuntime([_stable_observation("Paper", 0)])
        provider = VisionYolov10Objdetect(runtime_factory=lambda vid: runtime)
        request = VisionAdjudicationRequest("rps", self._profile(), "round-1", 5)
        seen: list[dict] = []

        result = provider.observe(
            request,
            on_log=lambda _line: None,
            on_event=seen.append,
            is_cancelled=lambda: False,
            timeout_seconds=5,
        )

        # The observation contract carries the folded label the game maps on.
        observation = result["observations"]["default"]
        assert observation["detections"][0]["label"] == "Paper"
        # Inference stops as soon as the evidence exists; the resident
        # camera/RTSP pipeline stays warm (no stop() call).
        assert any(c["command"] == "STOP_ADJUDICATION" for c in runtime.commands)
        assert runtime.stop_calls == 0
        # The choreography tail (verifying/result/holding/complete) belongs to
        # the caller; observe itself only announces the phases it ran.
        assert not any(e.get("event") == "result" for e in seen)

    def test_observe_timeout_returns_the_diagnosis_contract(self):
        runtime = _FakeRuntime([])  # no events at all -> collection times out
        provider = VisionYolov10Objdetect(runtime_factory=lambda vid: runtime)
        request = VisionAdjudicationRequest("rps", self._profile(), "round-2", 5)

        result = provider.observe(
            request,
            on_log=lambda _line: None,
            on_event=lambda _event: None,
            is_cancelled=lambda: False,
            timeout_seconds=0.2,
        )

        assert result.get("diagnosed") is True
        assert result["diagnosis"]["reason_code"] == "NO_OBJECTS_DETECTED"

    def test_observe_cancel_sends_cancel_to_the_runtime(self):
        runtime = _FakeRuntime([])
        provider = VisionYolov10Objdetect(runtime_factory=lambda vid: runtime)
        request = VisionAdjudicationRequest("rps", self._profile(), "round-3", 5)

        with pytest.raises(RuntimeError, match="cancelled"):
            provider.observe(
                request,
                on_log=lambda _line: None,
                on_event=lambda _event: None,
                is_cancelled=lambda: True,
                timeout_seconds=1,
            )

        assert any(c["command"] == "CANCEL" for c in runtime.commands)


class TestRpsRuntimeConfigContract:
    """The shipped rps adjudicator_config pins the v10 fold/stability keys."""

    def test_rps_config_declares_the_v10_vocabulary(self):
        config_path = ROOT / "backend" / "games" / "rps" / "adjudicator_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        # The package has no built-in vocabulary: everything must be declared.
        assert config["model"] == "models/yolov10n_gestures.q.onnx"
        assert "no_gesture" in config["classes"]
        assert len(config["classes"]) == 34
        assert config["rps_mode"] is True
        labels = set(config["rps_map"])
        assert labels == {"Rock", "Paper", "Scissors"}
        # Every folded source class must exist in the declared vocabulary.
        for sources in config["rps_map"].values():
            for source in sources:
                assert source in config["classes"]
        # Fold happens before stability: 15 folded-multiset frames.
        assert config["stable_frames"] == 15
        # Rotation + ROI are the rps hardware specifics from the demo.
        assert config["rotate"] == {"enabled": True, "direction": "cw", "angle": 90}
        assert config["roi"]["enabled"] is True
        # rtsp.path stays undeclared: manifest video.path overrides via CLI.
        assert "path" not in config["rtsp"]
        # The v10 model file is checked in.
        model_path = config_path.parent / config["model"]
        assert model_path.is_file(), model_path


class TestGamesProfileRouting:
    """games.py validates a game's vision_profile with its declared provider."""

    def _write_game(self, root: Path, game_id: str, provider_id: str) -> None:
        game_dir = root / game_id
        game_dir.mkdir(parents=True)
        (game_dir / "manifest.json").write_text(json.dumps({
            "id": game_id,
            "name": game_id,
            "enabled": True,
            "providers": {"vision_adjudicator": provider_id},
            "state_machine": {
                "schema_version": 1,
                "initial": "rules",
                "states": {
                    "rules": {
                        "on_enter": [
                            {"action": "speech", "mode": "tts_local", "text": "欢迎"}
                        ]
                    }
                },
            },
            "vision_profile": {
                "schema_version": 1,
                "game_id": game_id,
                "runtime_config": "backend/games/dice/adjudicator_config.json",
                "vision": {"class_map": {"0": "1"}, "participants": ["LEFT", "RIGHT"]},
                "video": {"enabled": False, "path": "/demo/det"},
                "llm": {
                    "enabled": False,
                    "context_mode": "single_turn_no_history",
                    "timeout_seconds": 5,
                    "system_prompt": "s",
                    "user_prompt_template": "u",
                    "allowed_outcomes": ["LEFT", "RIGHT", "TIE"],
                },
                "rule": {"kind": "categorical_relation", "relations": {"a": "b"}},
            },
        }, ensure_ascii=False), encoding="utf-8")

    def test_game_declaring_v10_routes_validation_to_the_v10_module(self, tmp_path):
        self._write_game(tmp_path, "rpsdemo", "vision_yolov10_objdetect")
        with patch.object(games_module, "GAMES_ROOT", Path(tmp_path)):
            registry = games_module.load_games()
        manifest = registry.get("rpsdemo")
        assert manifest["providers"]["vision_adjudicator"] == "vision_yolov10_objdetect"
        assert manifest["vision_profile"]["game_id"] == "rpsdemo"

    def test_game_declaring_packageless_vision_provider_falls_back_to_schema(self, tmp_path):
        """In-memory providers (test doubles) carry no package; the shared
        v8 schema validates them instead of refusing to load the game."""
        self._write_game(tmp_path, "rpsdemo", "vision_dummy")
        with patch.object(games_module, "GAMES_ROOT", Path(tmp_path)):
            registry = games_module.load_games()
        manifest = registry.get("rpsdemo")
        assert manifest["providers"]["vision_adjudicator"] == "vision_dummy"
        assert manifest["vision_profile"]["game_id"] == "rpsdemo"
