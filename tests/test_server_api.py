from __future__ import annotations

import json
import os
import sys
import threading
import tempfile
import time
import unittest
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import server  # noqa: E402
from core.components import ComponentRegistry  # noqa: E402
from core.games import GameRegistry  # noqa: E402
from core.jobs import ComponentJob  # noqa: E402
from core.tts import TtsProvider  # noqa: E402
from core.vision import VisionAdjudicatorProvider  # noqa: E402


WAV = b"RIFF" + b"\0" * 4 + b"WAVE" + b"\0" * 40


def test_shutdown_runtime_components_calls_provider_shutdown(monkeypatch):
    class Provider:
        id = "vision_dummy"
        type = "vision"
        role = "adjudicator"

        def __init__(self):
            self.shutdown_calls = 0

        def shutdown(self):
            self.shutdown_calls += 1

    provider = Provider()

    class Registry:
        def ids(self):
            return [provider.id]

        def get(self, component_id):
            assert component_id == provider.id
            return provider

    monkeypatch.setattr(server, "COMPONENTS", Registry())
    monkeypatch.setattr(server, "active_job_id", None)

    server._shutdown_runtime_components()

    assert provider.shutdown_calls == 1


class DummyVisionAdjudicator(VisionAdjudicatorProvider):
    id = "vision_dummy"
    type = "vision"
    role = "adjudicator"

    def health(self):
        return {"id": self.id, "type": self.type, "ok": True, "ready": True}

    def adjudicate(self, *, on_log, on_event, is_cancelled, timeout_seconds):
        return {
            "verified": True,
            "winner": "LEFT",
            "outcome": {"kind": "winner", "value": "LEFT"},
            "left_values": [6, 5, 4, 3, 2],
            "right_values": [1, 1, 1, 1, 1],
            "left_sum": 20,
            "right_sum": 5,
            "first_dice": [6, 5, 4, 3, 2],
            "second_dice": [1, 1, 1, 1, 1],
            "first_sum": 20,
            "second_sum": 5,
        }


class DummyTts(TtsProvider):
    id = "tts_dummy"

    def __init__(self):
        self.last_payload = None

    def health(self):
        return {"id": self.id, "type": self.type, "ok": True, "engine": "dummy"}

    def synthesize(self, payload):
        self.last_payload = dict(payload)
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

    def test_health_exposes_vision_profile_video_and_prewarm_metadata(self):
        original_games = server.GAMES
        original_components = server.COMPONENTS
        registry = ComponentRegistry()
        registry.register(DummyVisionAdjudicator(), {
            "id": "vision_dummy", "type": "vision", "role": "adjudicator",
            "name": "Dummy Vision Adjudicator", "version": "1", "enabled": True,
            "entry": "provider.py:DummyVisionAdjudicator",
        })
        server.COMPONENTS = registry
        games = GameRegistry()
        games.register({
            "id": "dice", "name": "Dice", "enabled": True,
            "providers": {"vision_adjudicator": "vision_dummy"}, "texts": {},
            "vision_profile": {
                "game_id": "dice", "video": {"enabled": True, "path": "/dice/", "webrtc_base_url": "http://100.118.229.28:8889"},
            },
        })
        server.GAMES = games
        try:
            with patch.object(server, "load_component_config", return_value={
                "runtime": {"mode": "resident", "prewarm_camera": True},
            }):
                status, _, data = self.request("GET", "/api/health")
        finally:
            server.GAMES = original_games
            server.COMPONENTS = original_components
        payload = json.loads(data)
        self.assertEqual(status, 200)
        self.assertEqual(payload["adjudicator"]["profile_id"], "dice")
        self.assertEqual(payload["adjudicator"]["video_path"], "/dice/")

    def test_health_exposes_multiview_profiles_and_mediamtx_base_url(self):
        original_games = server.GAMES
        original_components = server.COMPONENTS
        registry = ComponentRegistry()
        registry.register(DummyVisionAdjudicator(), {
            "id": "vision_dummy", "type": "vision", "role": "adjudicator",
            "name": "Dummy Vision Adjudicator", "version": "1", "enabled": True,
            "entry": "provider.py:DummyVisionAdjudicator",
        })
        server.COMPONENTS = registry
        games = GameRegistry()
        games.register({
            "id": "dice", "name": "Dice", "enabled": True,
            "providers": {"vision_adjudicator": "vision_dummy"}, "texts": {},
            "vision_profile": {
                "game_id": "dice", "video": {"enabled": True, "path": "/dice/", "webrtc_base_url": "http://100.118.229.28:8889"},
                "multi_view": {"enabled": True, "min_views": 2, "views": [
                    {"id": "front", "camera": "/dev/video1", "video": {"path": "/front/"}},
                    {"id": "side", "camera": "/dev/video2", "video": {"path": "/side/"}},
                ]},
            },
        })
        server.GAMES = games
        try:
            with patch.object(server, "load_component_config", return_value={
                "runtime": {"mode": "resident", "prewarm_camera": True},
            }):
                status, _, data = self.request("GET", "/api/health")
        finally:
            server.GAMES = original_games
            server.COMPONENTS = original_components
        payload = json.loads(data)
        self.assertEqual(status, 200)
        self.assertEqual(payload["adjudicator"]["mode"], "resident")
        self.assertTrue(payload["adjudicator"]["prewarm"])
        self.assertEqual(payload["adjudicator"]["mediamtx_base_url"], "http://100.118.229.28:8889")
        self.assertEqual(payload["adjudicator"]["webrtc_base_url"], "http://100.118.229.28:8889")
        self.assertTrue(payload["adjudicator"]["multi_view"]["enabled"])
        self.assertEqual(len(payload["adjudicator"]["profiles"]), 1)

    def test_games_api_exposes_only_safe_vision_video_metadata(self):
        original_games = server.GAMES
        games = GameRegistry()
        games.register({
            "id": "dice", "name": "Dice", "enabled": True, "providers": {}, "texts": {},
            "participants": {"player": "RIGHT", "agent": "LEFT"},
            "vision_profile": {
                "game_id": "dice",
                "llm": {"api_key": "SECRET", "system_prompt": "PRIVATE PROMPT", "model": "secret-model"},
                "vision": {"model": "/tmp/private.onnx"},
                "video": {"enabled": True, "path": "/dice/"},
                "multi_view": {"enabled": True, "views": [
                    {"id": "front", "camera": "/dev/video1", "video": {"path": "/front/"}},
                ]},
            },
        })
        server.GAMES = games
        try:
            status, _, data = self.request("GET", "/api/games")
        finally:
            server.GAMES = original_games
        payload = json.loads(data)
        self.assertEqual(status, 200)
        self.assertEqual(
            payload["games"][0]["participants"],
            {"player": "RIGHT", "agent": "LEFT"},
        )
        profile = payload["games"][0]["vision_profile"]
        self.assertEqual(profile["video"], {"enabled": True, "path": "/dice/"})
        self.assertEqual(profile["multi_view"]["views"][0], {"id": "front", "video": {"enabled": True, "path": "/front/"}})
        serialized = json.dumps(payload)
        for secret in ("PRIVATE PROMPT", "SECRET", "secret-model", "/tmp/private.onnx", "/dev/video1"):
            self.assertNotIn(secret, serialized)

    def test_audio_speech_stream_reads_manifest_selected_wav(self):
        import core.games as games_module

        registry = GameRegistry()
        registry.register({
            "id": "dice",
            "name": "Dice",
            "enabled": True,
            "providers": {"tts": "tts_dummy"},
            "texts": {"rules_intro": {"mode": "audio", "audio": "audio/intro.wav"}},
        })
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_path = root / "dice" / "audio" / "intro.wav"
            audio_path.parent.mkdir(parents=True)
            audio_path.write_bytes(WAV)
            original_games = server.GAMES
            original_games_root = games_module.GAMES_ROOT
            server.GAMES = registry
            games_module.GAMES_ROOT = root
            try:
                status, headers, data = self.request(
                    "POST", "/api/speech/stream", {"game": "dice", "key": "rules_intro"}
                )
            finally:
                server.GAMES = original_games
                games_module.GAMES_ROOT = original_games_root
        self.assertEqual(status, 200)
        self.assertEqual(headers["X-Dice-Speech-Key"], "rules_intro")
        self.assertEqual(headers["X-Dice-Speech-Mode"], "audio")
        self.assertEqual(int.from_bytes(data[:4], "big"), len(WAV))
        self.assertEqual(data[4:-4], WAV)
        self.assertEqual(data[-4:], b"\0\0\0\0")

    def test_tts_speech_stream_renders_manifest_values(self):
        original_games = server.GAMES
        registry = GameRegistry()
        registry.register({
            "id": "dice",
            "name": "Dice",
            "enabled": True,
            "providers": {"tts": "tts_dummy"},
            "voice": "announcer",
            "speed": 1.25,
            "texts": {
                "result_player_win": {
                    "mode": "tts",
                    "text": "玩家 {player_score}，Agent {agent_score}",
                }
            },
        })
        server.GAMES = registry
        try:
            status, headers, data = self.request(
                "POST",
                "/api/speech/stream",
                {
                    "game": "dice",
                    "key": "result_player_win",
                    "values": {"player_score": 18, "agent_score": 12},
                },
            )
        finally:
            server.GAMES = original_games
        provider = server.COMPONENTS.require("tts_dummy", expected_type="tts")
        self.assertEqual(status, 200)
        self.assertEqual(headers["X-Dice-Speech-Mode"], "tts")
        self.assertEqual(provider.last_payload["text"], "玩家 18，Agent 12")
        self.assertEqual(provider.last_payload["voice"], "announcer")
        self.assertEqual(provider.last_payload["speed"], 1.25)
        self.assertEqual(data[4:-4], WAV)

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
        for _ in range(100):
            status, _, data = self.request("GET", f"/api/adjudicate/{job_id}")
            self.assertEqual(status, 200)
            payload = json.loads(data)
            if payload["status"] not in {"queued", "running"}:
                break
            time.sleep(0.01)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["result"]["winner"], "LEFT")
        self.assertEqual(payload["result"]["winner_role"], "PLAYER")
        self.assertEqual(payload["result"]["player_score"], 20)
        self.assertEqual(payload["result"]["agent_score"], 5)

        # Old clients can still read the same adjudication job during migration.
        status, _, legacy_data = self.request("GET", f"/api/analyze/{job_id}")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(legacy_data)["job_id"], job_id)

        with server.jobs_lock:
            server.jobs.pop(job_id, None)


if __name__ == "__main__":
    unittest.main()
