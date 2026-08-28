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
from core.games import resolve_provider_id  # noqa: E402
from core.jobs import ComponentJob  # noqa: E402
from core.tts import TtsProvider  # noqa: E402
from core.tts_config import TtsConfigError, load_component_config, resolve_config_path  # noqa: E402
from core.vision import VisionAdjudicatorProvider, VisionLocalizerProvider  # noqa: E402
from components.vision_yolo.provider import _consume_legacy_log_line  # noqa: E402
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


class DummyAdjudicator(VisionAdjudicatorProvider):
    id = "vision_dummy_adjudicator"

    def adjudicate(self, *, on_log, on_event, is_cancelled, timeout_seconds):
        return {"verified": True, "winner": "LEFT"}


class DummyLocalizer(VisionLocalizerProvider):
    id = "vision_dummy_localizer"

    def locate(self, request, *, on_log, on_event, is_cancelled, timeout_seconds):
        return {"objects": [], "coordinate_frame": request.get("coordinate_frame", "pixel")}


class ComponentTests(unittest.TestCase):
    def test_registry_loads_packaged_providers(self):
        registry = build_registry()
        self.assertEqual(registry.provider_ids("vision"), ["vision_yolo"])
        self.assertEqual(registry.provider_ids("vision", "adjudicator"), ["vision_yolo"])
        self.assertEqual(registry.provider_ids("vision", "localizer"), [])
        self.assertEqual(registry.provider_ids("tts"), ["tts_moss_nano", "tts_qwen3"])
        self.assertEqual(registry.get_manifest("tts_qwen3")["entry"], "provider.py:TtsQwen3")
        self.assertEqual(registry.get_manifest("tts_moss_nano")["entry"], "provider.py:TtsMossNano")
        self.assertEqual(registry.get_manifest("vision_yolo")["role"], "adjudicator")

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
        with patch.dict("os.environ", {
            "DICE_VISION_ADJUDICATOR_PROVIDER": "",
            "DICE_VISION_PROVIDER": "",
        }):
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
        with patch.dict("os.environ", {
            "DICE_VISION_ADJUDICATOR_PROVIDER": "",
            "DICE_VISION_PROVIDER": "",
        }):
            result = dice_pipeline.run(
                lambda _line: None,
                lambda: False,
                1.0,
                components=registry,
                manifest={"providers": {"vision_adjudicator": "vision_dummy_adjudicator"}},
                on_event=lambda _event: None,
            )
        self.assertEqual(result["winner"], "LEFT")

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
