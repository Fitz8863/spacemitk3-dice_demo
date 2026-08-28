from __future__ import annotations

import sys
import threading
import time
import unittest
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
from core.vision import VisionAdjudicatorProvider, VisionLocalizerProvider  # noqa: E402
from components.vision_yolo.provider import _consume_legacy_log_line  # noqa: E402
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
        self.assertEqual(registry.provider_ids("tts"), ["tts_qwen3"])
        self.assertEqual(registry.get_manifest("tts_qwen3")["entry"], "provider.py:TtsQwen3")
        self.assertEqual(registry.get_manifest("vision_yolo")["role"], "adjudicator")

    def test_default_tts_stream_wraps_single_wav(self):
        provider = DummyTts()
        frames = []
        provider.stream({"text": "hello"}, frames.append)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0][:4], b"RIFF")

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
