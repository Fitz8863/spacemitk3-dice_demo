from __future__ import annotations

import json
import os
import sys
import threading
import time
import unittest
from http.client import HTTPConnection
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import server  # noqa: E402
from core.components import ComponentRegistry  # noqa: E402
from core.jobs import ComponentJob  # noqa: E402
from core.tts import TtsProvider  # noqa: E402
from core.vision import VisionAdjudicatorProvider  # noqa: E402


WAV = b"RIFF" + b"\0" * 4 + b"WAVE" + b"\0" * 40


class DummyVisionAdjudicator(VisionAdjudicatorProvider):
    id = "vision_dummy"
    type = "vision"
    role = "adjudicator"

    def health(self):
        return {"id": self.id, "type": self.type, "ok": True, "ready": True}

    def adjudicate(self, *, on_log, on_event, is_cancelled, timeout_seconds):
        return {"verified": True, "winner": "LEFT"}


class DummyTts(TtsProvider):
    id = "tts_dummy"

    def health(self):
        return {"id": self.id, "type": self.type, "ok": True, "engine": "dummy"}

    def synthesize(self, payload):
        self.validate(payload)
        return WAV, {"Content-Type": "audio/wav"}


class ServerApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_components = server.COMPONENTS
        cls.original_tts_provider = os.environ.get("DICE_TTS_PROVIDER")
        cls.original_adjudicator_provider = os.environ.get("DICE_VISION_ADJUDICATOR_PROVIDER")
        cls.original_vision_provider = os.environ.get("DICE_VISION_PROVIDER")
        registry = ComponentRegistry()
        registry.register(DummyVisionAdjudicator(), {
            "id": "vision_dummy", "type": "vision", "role": "adjudicator",
            "name": "Dummy Vision Adjudicator", "version": "1", "enabled": True,
            "entry": "provider.py:DummyVisionAdjudicator",
        })
        registry.register(DummyTts(), {
            "id": "tts_dummy", "type": "tts", "name": "Dummy TTS",
            "version": "1", "enabled": True, "entry": "provider.py:DummyTts",
        })
        server.COMPONENTS = registry
        os.environ["DICE_TTS_PROVIDER"] = "tts_dummy"
        os.environ["DICE_VISION_ADJUDICATOR_PROVIDER"] = "vision_dummy"
        os.environ.pop("DICE_VISION_PROVIDER", None)
        cls.httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=2)
        server.COMPONENTS = cls.original_components
        if cls.original_tts_provider is None:
            os.environ.pop("DICE_TTS_PROVIDER", None)
        else:
            os.environ["DICE_TTS_PROVIDER"] = cls.original_tts_provider
        if cls.original_adjudicator_provider is None:
            os.environ.pop("DICE_VISION_ADJUDICATOR_PROVIDER", None)
        else:
            os.environ["DICE_VISION_ADJUDICATOR_PROVIDER"] = cls.original_adjudicator_provider
        if cls.original_vision_provider is None:
            os.environ.pop("DICE_VISION_PROVIDER", None)
        else:
            os.environ["DICE_VISION_PROVIDER"] = cls.original_vision_provider

    def request(self, method, path, payload=None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=3)
        body = None
        headers = {}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        data = response.read()
        response_headers = dict(response.getheaders())
        connection.close()
        return response.status, response_headers, data

    def test_provider_selection_keeps_frontend_tts_contract(self):
        status, _, data = self.request("GET", "/api/health")
        payload = json.loads(data)
        self.assertEqual(status, 200)
        self.assertEqual(payload["tts_provider"], "tts_dummy")
        self.assertEqual(payload["adjudicator_provider"], "vision_dummy")
        self.assertEqual(payload["adjudicator"]["role"], "adjudicator")
        self.assertEqual(payload["vision"]["id"], "vision_dummy")

        status, headers, data = self.request(
            "POST", "/api/tts/synthesize", {"text": "hello", "game": "dice"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["X-Dice-TTS-Provider"], "tts_dummy")
        self.assertEqual(data, WAV)

        status, headers, data = self.request(
            "POST", "/api/tts/stream", {"text": "hello", "game": "dice"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["X-Dice-TTS-Provider"], "tts_dummy")
        self.assertEqual(int.from_bytes(data[:4], "big"), len(WAV))
        self.assertEqual(data[4:-4], WAV)
        self.assertEqual(data[-4:], b"\0\0\0\0")

    def test_sse_pushes_structured_job_updates(self):
        def run(_on_log, _cancelled, on_event):
            on_event({"event": "phase", "phase": "detecting"})
            on_event({"event": "progress", "phase": "detecting", "stable_count": 1, "stable_frames": 2})
            result = {"verified": True, "winner": "LEFT"}
            on_event({"event": "result", **result})
            return result

        job = ComponentJob(run)
        with server.jobs_lock:
            server.jobs[job.id] = job
        job.start()
        connection = HTTPConnection("127.0.0.1", self.port, timeout=3)
        connection.request("GET", f"/api/adjudicate/{job.id}/stream")
        response = connection.getresponse()
        body = response.read().decode("utf-8")
        connection.close()
        job.thread.join(timeout=2)
        with server.jobs_lock:
            server.jobs.pop(job.id, None)
        self.assertEqual(response.status, 200)
        self.assertIn("event: snapshot", body)
        self.assertIn("event: complete", body)
        self.assertIn('"event":"result"', body)
        self.assertIn('"status":"success"', body)

    def test_canonical_adjudication_endpoint_uses_adjudicator_slot(self):
        status, _, data = self.request(
            "POST", "/api/adjudicate", {"game": "dice"}
        )
        self.assertEqual(status, 202)
        job_id = json.loads(data)["job_id"]

        payload = None
        for _ in range(20):
            status, _, data = self.request("GET", f"/api/adjudicate/{job_id}")
            self.assertEqual(status, 200)
            payload = json.loads(data)
            if payload["status"] not in {"queued", "running"}:
                break
            time.sleep(0.01)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["result"]["winner"], "LEFT")

        # Old clients can still read the same adjudication job during migration.
        status, _, legacy_data = self.request("GET", f"/api/analyze/{job_id}")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(legacy_data)["job_id"], job_id)

        with server.jobs_lock:
            server.jobs.pop(job_id, None)


if __name__ == "__main__":
    unittest.main()
