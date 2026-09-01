from __future__ import annotations

import io
import json
import os
import tempfile
import sys
import threading
import time
import unittest
import wave
from unittest.mock import patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.components import Component, ComponentRegistry, _validate_manifest, build_registry  # noqa: E402
from core.games import load_games, resolve_provider_id  # noqa: E402
from core.jobs import ComponentJob  # noqa: E402
from core.tts import TtsProvider  # noqa: E402
from core.tts_config import TtsConfigError, load_component_config, resolve_config_path  # noqa: E402
from core.vision import VisionAdjudicatorProvider, VisionLocalizerProvider  # noqa: E402
from components.vision_yolov8_adjudicator.provider import _consume_legacy_log_line  # noqa: E402
from components.tts_moss_nano.daemon import TTS_CONFIG as MOSS_CONFIG, _wave_bytes  # noqa: E402
from components.tts_moss_nano.provider import MOSS_ROOT, MOSS_VOICE, TtsMossNano  # noqa: E402
from components.tts_qwen3.provider import TTS_ROOT as QWEN_ROOT, TTS_SPEAKER_FILE  # noqa: E402
from games.dice import pipeline as dice_pipeline  # noqa: E402


class DummyTts(TtsProvider):
    id = "tts_dummy"

    def synthesize(self, payload):
        self.validate(payload)
        return b"RIFF" + b"\0" * 4 + b"WAVE" + b"\0" * 40, {"Content-Type": "audio/wav"}


class InvalidVision(Component):
    id = "vision_invalid"
    type = "vision"
    role = "adjudicator"


def physical_dice_result(winner="LEFT"):
    return {
        "verified": True,
        "winner": winner,
        "outcome": {"kind": "winner", "value": winner},
        "left_values": [6, 5, 4, 3, 2],
        "right_values": [1, 1, 1, 1, 1],
        "left_sum": 20,
        "right_sum": 5,
        "first_dice": [6, 5, 4, 3, 2],
        "second_dice": [1, 1, 1, 1, 1],
        "first_sum": 20,
        "second_sum": 5,
    }


class DummyAdjudicator(VisionAdjudicatorProvider):
    id = "vision_dummy_adjudicator"

    def adjudicate(self, *, on_log, on_event, is_cancelled, timeout_seconds):
        return physical_dice_result()


class RequestAwareDummyAdjudicator(VisionAdjudicatorProvider):
    id = "vision_request_aware_adjudicator"

    def adjudicate(self, request, *, on_log, on_event, is_cancelled, timeout_seconds):
        return physical_dice_result()


class DummyLocalizer(VisionLocalizerProvider):
    id = "vision_dummy_localizer"

    def locate(self, request, *, on_log, on_event, is_cancelled, timeout_seconds):
        return {"objects": [], "coordinate_frame": request.get("coordinate_frame", "pixel")}


class ComponentTests(unittest.TestCase):
    def test_game_loader_prefers_inline_vision_profile(self):
        import core.games as games_module

        profile = json.loads((ROOT / "backend/games/dice/manifest.json").read_text())["vision_profile"]
        profile["video"]["webrtc_base_url"] = "http://inline.example:8889"
        with tempfile.TemporaryDirectory() as temp_dir:
            game_dir = Path(temp_dir) / "dice"
            game_dir.mkdir()
            (game_dir / "manifest.json").write_text(json.dumps({
                "id": "dice",
                "name": "Dice",
                "enabled": True,
                "participants": {"player": "LEFT", "agent": "RIGHT"},
                "providers": {},
                "texts": {},
                "vision_profile": profile,
            }), encoding="utf-8")
            with patch.object(games_module, "GAMES_ROOT", Path(temp_dir)):
                registry = load_games()
        loaded = registry.get("dice")
        self.assertEqual(
            loaded["vision_profile"]["video"]["webrtc_base_url"],
            "http://inline.example:8889",
        )
        self.assertEqual(
            loaded["participants"],
            {"player": "LEFT", "agent": "RIGHT"},
        )

    def test_game_loader_rejects_ambiguous_participant_layout(self):
        import core.games as games_module

        with tempfile.TemporaryDirectory() as temp_dir:
            game_dir = Path(temp_dir) / "dice"
            game_dir.mkdir()
            (game_dir / "manifest.json").write_text(json.dumps({
                "id": "dice",
                "name": "Dice",
                "enabled": True,
                "participants": {"player": "LEFT", "agent": "LEFT"},
                "providers": {},
                "texts": {},
            }), encoding="utf-8")
            with patch.object(games_module, "GAMES_ROOT", Path(temp_dir)):
                registry = load_games()
        self.assertEqual(registry.all(), [])

    def test_registry_loads_packaged_providers(self):
        registry = build_registry()
        self.assertEqual(
            registry.provider_ids("vision"),
            ["vision_yolov8_adjudicator"],
        )
        self.assertEqual(
            registry.provider_ids("vision", "adjudicator"),
            ["vision_yolov8_adjudicator"],
        )
        self.assertEqual(registry.provider_ids("vision", "localizer"), [])
        self.assertEqual(registry.provider_ids("tts"), ["tts_moss_nano", "tts_qwen3"])
        self.assertEqual(registry.get_manifest("tts_qwen3")["entry"], "provider.py:TtsQwen3")
        self.assertEqual(registry.get_manifest("tts_moss_nano")["entry"], "provider.py:TtsMossNano")
        self.assertEqual(
            registry.get("vision_yolo").id,
            "vision_yolov8_adjudicator",
        )
        self.assertEqual(registry.get_manifest("vision_yolo")["role"], "adjudicator")
        self.assertEqual(
            registry.get_manifest("vision_yolov8_adjudicator")["entry"],
            "provider.py:VisionYolov8Adjudicator",
        )

    def test_tts_components_have_independent_configs(self):
        moss = load_component_config(ROOT / "backend" / "components" / "tts_moss_nano")
        qwen = load_component_config(ROOT / "backend" / "components" / "tts_qwen3")
        self.assertEqual(moss["voice"]["name"], "Junhao")
        self.assertIn(moss["voice"]["mode"], {"builtin", "clone"})
        if moss["voice"]["mode"] == "clone":
            reference_audio = ROOT / "tts" / "moss-tts-nano" / moss["voice"]["reference_audio"]
            self.assertTrue(reference_audio.is_file(), reference_audio)
        self.assertEqual(qwen["voice"]["speaker_file"], "anke.spk.bin")
        self.assertNotEqual(moss, qwen)

    def test_tts_config_paths_are_repository_relative(self):
        self.assertEqual(MOSS_CONFIG["runtime"]["root"], "tts/moss-tts-nano")
        self.assertEqual(Path(MOSS_ROOT), (ROOT / "tts" / "moss-tts-nano").resolve())
        self.assertEqual(QWEN_ROOT, (ROOT / "tts" / "qwen3-tts").resolve())
        self.assertEqual(MOSS_VOICE, "Junhao")
        self.assertEqual(TTS_SPEAKER_FILE, "anke.spk.bin")

    def test_relative_reference_audio_resolves_from_runtime_root(self):
        runtime_root = ROOT / "tts" / "moss-tts-nano"
        resolved = resolve_config_path("voice/reference.wav", base_dir=runtime_root)
        self.assertEqual(resolved, (runtime_root / "voice" / "reference.wav").resolve())

    def test_tts_config_loader_rejects_missing_and_invalid_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            component = Path(temp_dir)
            with self.assertRaises(TtsConfigError):
                load_component_config(component)
            (component / "config.json").write_text("[]", encoding="utf-8")
            with self.assertRaises(TtsConfigError):
                load_component_config(component)

    def test_moss_health_declares_voice_clone_without_network(self):
        provider = TtsMossNano()
        with patch(
            "components.tts_moss_nano.provider.urllib.request.urlopen",
            side_effect=OSError("offline"),
        ):
            health = provider.health()
        self.assertTrue(health["supports_voice_clone"])
        self.assertEqual(health["voice_mode"], MOSS_CONFIG["voice"]["mode"])
        if health["voice_mode"] == "clone":
            self.assertTrue(health["reference_audio"])
        else:
            self.assertIsNone(health["reference_audio"])

    def test_default_tts_stream_wraps_single_wav(self):
        provider = DummyTts()
        frames = []
        provider.stream({"text": "hello"}, frames.append)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0][:4], b"RIFF")

    def test_moss_chunk_encoder_returns_browser_playable_wav(self):
        import numpy as np

        audio = _wave_bytes(np.zeros((480, 2), dtype=np.float32), 48000)
        with wave.open(io.BytesIO(audio), "rb") as wav_file:
            self.assertEqual(wav_file.getframerate(), 48000)
            self.assertEqual(wav_file.getnchannels(), 2)
            self.assertEqual(wav_file.getsampwidth(), 2)
            self.assertEqual(wav_file.getnframes(), 480)

    def test_moss_provider_rejects_unsupported_speed_before_network(self):
        provider = TtsMossNano()
        with self.assertRaisesRegex(Exception, "speed=1.0"):
            provider.stream({"text": "hello", "speed": 1.1}, lambda _frame: None)

    def test_registry_rejects_vision_provider_without_vision_interface(self):
        registry = ComponentRegistry()
        with self.assertRaisesRegex(ValueError, "VisionAdjudicatorProvider"):
            registry.register(InvalidVision())

    def test_registry_does_not_allow_localizer_in_adjudicator_slot(self):
        registry = ComponentRegistry()
        registry.register(DummyLocalizer(), {
            "id": "vision_dummy_localizer",
            "type": "vision",
            "role": "localizer",
            "entry": "provider.py:DummyLocalizer",
        })
        with self.assertRaisesRegex(Exception, "role mismatch: expected adjudicator"):
            registry.require(
                "vision_dummy_localizer",
                expected_type="vision",
                expected_role="adjudicator",
            )

    def test_registry_logs_legacy_alias_once_and_resolves_canonical(self):
        logs = []
        registry = ComponentRegistry(migration_logger=logs.append)
        registry.register(DummyAdjudicator(), {
            "id": "vision_dummy_adjudicator",
            "type": "vision",
            "role": "adjudicator",
            "entry": "provider.py:DummyAdjudicator",
        })
        # Install a test-only alias to exercise the same registry seam without
        # requiring the production provider to be renamed.
        from core import components as component_module
        with patch.dict(component_module.COMPONENT_ID_ALIASES, {"vision_old": "vision_dummy_adjudicator"}):
            assert registry.get("vision_old").id == "vision_dummy_adjudicator"
            assert registry.require("vision_old", expected_type="vision").id == "vision_dummy_adjudicator"
        assert logs == ["[components] migration alias vision_old -> vision_dummy_adjudicator"]

    def test_manifest_entry_cannot_escape_provider_package(self):
        manifest_path = ROOT / "backend" / "components" / "vision_bad" / "manifest.json"
        with self.assertRaisesRegex(ValueError, "inside the provider package"):
            _validate_manifest(manifest_path, {
                "id": "vision_bad",
                "type": "vision",
                "role": "adjudicator",
                "entry": "../outside.py:Provider",
            })

    def test_semantic_adjudicator_slot_supports_legacy_aliases(self):
        canonical = {"providers": {"vision_adjudicator": "vision_new"}}
        legacy = {"providers": {"vision": "vision_old"}}
        self.assertEqual(
            resolve_provider_id(canonical, "vision_adjudicator", "vision_fallback"),
            "vision_new",
        )
        self.assertEqual(
            resolve_provider_id(legacy, "vision_adjudicator", "vision_fallback"),
            "vision_old",
        )

    def test_dice_pipeline_invokes_adjudicator_interface(self):
        registry = ComponentRegistry()
        registry.register(DummyAdjudicator(), {
            "id": "vision_dummy_adjudicator",
            "type": "vision",
            "role": "adjudicator",
            "entry": "provider.py:DummyAdjudicator",
        })
        result = dice_pipeline.run(
            lambda _line: None,
            lambda: False,
            1.0,
            components=registry,
            manifest={
                "participants": {"player": "LEFT", "agent": "RIGHT"},
                "providers": {"vision_adjudicator": "vision_dummy_adjudicator"},
                "vision_profile": json.loads(
                    (ROOT / "backend/games/dice/manifest.json").read_text()
                )["vision_profile"],
            },
            on_event=lambda _event: None,
        )
        self.assertEqual(result["winner"], "LEFT")
        self.assertEqual(result["winner_role"], "PLAYER")
        self.assertEqual(result["player_score"], 20)
        self.assertEqual(result["agent_score"], 5)

    def test_dice_pipeline_projects_request_aware_adjudicator_result(self):
        registry = ComponentRegistry()
        registry.register(RequestAwareDummyAdjudicator(), {
            "id": "vision_request_aware_adjudicator",
            "type": "vision",
            "role": "adjudicator",
            "entry": "provider.py:RequestAwareDummyAdjudicator",
        })
        result = dice_pipeline.run(
            lambda _line: None,
            lambda: False,
            1.0,
            components=registry,
            manifest={
                "participants": {"player": "RIGHT", "agent": "LEFT"},
                "providers": {
                    "vision_adjudicator": "vision_request_aware_adjudicator"
                },
                "vision_profile": json.loads(
                    (ROOT / "backend/games/dice/manifest.json").read_text()
                )["vision_profile"],
            },
            on_event=lambda _event: None,
        )
        self.assertEqual(result["winner"], "LEFT")
        self.assertEqual(result["winner_role"], "AGENT")
        self.assertEqual(result["player_score"], 5)
        self.assertEqual(result["agent_score"], 20)

    def test_untagged_json_stdout_remains_diagnostic_log(self):
        logs = []
        events = []
        parsed = _consume_legacy_log_line(
            '{"event":"result","verified":true}', logs.append, events.append
        )
        self.assertIsNone(parsed)
        self.assertEqual(events, [])
        self.assertEqual(logs, ['{"event":"result","verified":true}'])


class JobTests(unittest.TestCase):
    def test_adjudicated_result_enters_holding_until_explicit_complete(self):
        release = threading.Event()

        def run(_on_log, _cancelled, on_event):
            on_event({"event": "result", "adjudicated": True, "winner": "LEFT"})
            on_event({"event": "phase", "phase": "holding", "remaining_ms": 100})
            release.wait(timeout=1)
            on_event({"event": "complete", "phase": "complete"})
            return {"adjudicated": True, "winner": "LEFT"}

        job = ComponentJob(run)
        job.start()
        deadline = time.time() + 1
        while time.time() < deadline:
            snapshot = job.snapshot()
            if snapshot["phase"] == "holding":
                break
            time.sleep(0.01)
        snapshot = job.snapshot()
        self.assertEqual(snapshot["status"], "running")
        self.assertEqual(snapshot["phase"], "holding")
        self.assertIsNone(snapshot["result"])
        release.set()
        job.thread.join(timeout=2)
        snapshot = job.snapshot()
        self.assertEqual(snapshot["status"], "success")
        self.assertEqual(snapshot["phase"], "complete")

    def test_adjudicated_return_without_complete_stays_holding(self):
        def run(_on_log, _cancelled, _on_event):
            return {"adjudicated": True, "winner": "LEFT"}

        job = ComponentJob(run)
        job.start()
        job.thread.join(timeout=1)
        snapshot = job.snapshot()
        self.assertEqual(snapshot["status"], "running")
        self.assertEqual(snapshot["phase"], "holding")
        self.assertIsNone(snapshot["finished_at"])

    def test_diagnosed_adjudication_failure_is_terminal_error_with_result(self):
        def run(_on_log, _cancelled, _on_event):
            return {
                "adjudicated": False,
                "diagnosed": True,
                "retry_required": True,
                "diagnosis": {
                    "reason_code": "INCOMPLETE_OBJECTS",
                    "message": "左侧只检测到 4 个目标，请重新开始。",
                },
            }

        job = ComponentJob(run)
        job.start()
        job.thread.join(timeout=1)
        snapshot = job.snapshot()
        self.assertEqual(snapshot["status"], "error")
        self.assertEqual(snapshot["phase"], "error")
        self.assertIn("左侧只检测到 4 个目标", snapshot["error"])
        self.assertTrue(snapshot["result"]["diagnosed"])

    def test_cancel_during_holding_is_terminal_and_cannot_become_success(self):
        entered = threading.Event()
        release = threading.Event()

        def run(_on_log, _cancelled, on_event):
            on_event({"event": "result", "adjudicated": True, "winner": "LEFT"})
            on_event({"event": "phase", "phase": "holding", "remaining_ms": 1000})
            entered.set()
            release.wait(timeout=1)
            on_event({"event": "complete", "phase": "complete"})
            return {"adjudicated": True, "winner": "LEFT"}

        job = ComponentJob(run)
        job.start()
        self.assertTrue(entered.wait(timeout=1))
        self.assertEqual(job.snapshot()["phase"], "holding")
        job.cancel()
        release.set()
        job.thread.join(timeout=2)
        snapshot = job.snapshot()
        self.assertEqual(snapshot["status"], "error")
        self.assertEqual(snapshot["phase"], "error")
        self.assertTrue(snapshot["cancelled"])
        self.assertIsNone(snapshot["result"])

    def test_verified_legacy_result_is_promoted_to_structured_event(self):
        def run(_log, _cancelled, _event):
            return {"verified": True, "winner": "LEFT"}

        job = ComponentJob(run)
        job.start()
        job.thread.join(timeout=2)
        snapshot = job.snapshot()
        self.assertEqual(snapshot["status"], "success")
        self.assertEqual(snapshot["events"][-1]["event"], "result")
        self.assertEqual(snapshot["events"][-1]["winner"], "LEFT")

    def test_legacy_phase_change_wakes_revision_waiter(self):
        release = threading.Event()
        finish = threading.Event()

        def run(on_log, _cancelled, _event):
            release.wait(timeout=1)
            on_log("provider detecting")
            finish.wait(timeout=1)
            return {"verified": False}

        job = ComponentJob(
            run, phase_of=lambda line: "detecting" if "detecting" in line else None
        )
        job.start()
        initial = job.snapshot()
        release.set()
        updated = job.wait_for_update(initial["revision"], timeout=1)
        finish.set()
        job.thread.join(timeout=2)
        self.assertGreater(updated["revision"], initial["revision"])
        self.assertEqual(updated["phase"], "detecting")
        self.assertIn("provider detecting", updated["logs"])

    def test_cancelled_job_cannot_later_become_success(self):
        entered = threading.Event()
        release = threading.Event()

        def run(_log, _cancelled, _event):
            entered.set()
            release.wait(timeout=1)
            return {"verified": True, "winner": "RIGHT"}

        job = ComponentJob(run)
        job.start()
        self.assertTrue(entered.wait(timeout=1))
        job.cancel()
        release.set()
        job.thread.join(timeout=2)
        snapshot = job.snapshot()
        self.assertEqual(snapshot["status"], "error")
        self.assertTrue(snapshot["cancelled"])
        self.assertIsNone(snapshot["result"])


if __name__ == "__main__":
    unittest.main()
