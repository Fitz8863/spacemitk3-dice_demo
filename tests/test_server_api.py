from __future__ import annotations

import json
import os
import shutil
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


class DummyRemoteTts(TtsProvider):
    id = "tts_dummy_remote"

    def __init__(self):
        self.last_payload = None

    def health(self):
        return {"id": self.id, "type": self.type, "ok": True, "engine": "dummy_remote"}

    def synthesize(self, payload):
        self.last_payload = dict(payload)
        self.validate(payload)
        return WAV, {"Content-Type": "audio/wav"}


def _write_hot_reload_games_root(tmp_path, text):
    """A minimal self-contained dice manifest pointing at the local dummy."""
    games_root = tmp_path / "games"
    (games_root / "dice").mkdir(parents=True)
    manifest = {
        "id": "dice",
        "name": "Dice",
        "enabled": True,
        "participants": {"player": "LEFT", "agent": "RIGHT"},
        "providers": {"tts_local": "tts_dummy"},
        "state_machine": {
            "schema_version": 1,
            "initial": "rules",
            "states": {
                "rules": {
                    "on_enter": [{"action": "speech", "mode": "tts_local", "text": text}],
                }
            },
        },
    }
    path = games_root / "dice" / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return games_root, path


def test_state_machine_manifest_hot_reloads_without_restart(tmp_path, monkeypatch):
    """Manifest edits take effect on the next registry read - no restart."""
    games_root, manifest_path = _write_hot_reload_games_root(tmp_path, "旧台词")
    _patch_dummy_registry(monkeypatch)
    monkeypatch.setattr(server, "_GAMES_ROOT", games_root)
    monkeypatch.setattr(server, "GAMES", server.load_games(games_root))
    monkeypatch.setattr(server, "_GAMES_MTIMES", server._manifest_mtimes())

    def entry_text():
        state = server.get_games().get("dice")["state_machine"]["states"]["rules"]
        return state["on_enter"][0]["text"]

    assert entry_text() == "旧台词"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["state_machine"]["states"]["rules"]["on_enter"][0]["text"] = "热加载台词"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    os.utime(manifest_path, (time.time() + 5, time.time() + 5))

    assert entry_text() == "热加载台词"


def test_hot_reload_keeps_last_good_config_on_broken_manifest(tmp_path, monkeypatch):
    """A typo that breaks a manifest must not make the game vanish."""
    games_root, manifest_path = _write_hot_reload_games_root(tmp_path, "旧台词")
    _patch_dummy_registry(monkeypatch)
    monkeypatch.setattr(server, "_GAMES_ROOT", games_root)
    monkeypatch.setattr(server, "GAMES", server.load_games(games_root))
    monkeypatch.setattr(server, "_GAMES_MTIMES", server._manifest_mtimes())

    manifest_path.write_text('{ "id": "dice", broken', encoding="utf-8")
    os.utime(manifest_path, (time.time() + 5, time.time() + 5))

    games = server.get_games()
    # The registry keeps serving the last good dice manifest.
    assert "dice" in [m["id"] for m in games.all()]
    state = games.get("dice")["state_machine"]["states"]["rules"]
    assert state["on_enter"][0]["mode"] == "tts_local"


def _patch_dummy_registry(monkeypatch):
    """Real components plus the local DummyTts, so no network is touched."""
    combined = ComponentRegistry()
    for component_id in server.COMPONENTS.ids():
        combined.register(
            server.COMPONENTS.get(component_id),
            server.COMPONENTS.get_manifest(component_id),
        )
    combined.register(DummyTts(), {
        "id": "tts_dummy", "type": "tts", "name": "Dummy TTS",
        "version": "1", "enabled": True, "entry": "provider.py:DummyTts",
    })
    monkeypatch.setattr(server, "COMPONENTS", combined)


ROUND_MACHINE = {
    "schema_version": 1,
    "initial": "rules",
    "states": {
        "rules": {
            "on_enter": [{"action": "speech", "mode": "tts_local", "text": "规则"}],
            "on_intent": {
                "confirm": {"to": "analysis"},
                "back": {"exit": True},
            },
        },
        "analysis": {
            "on_enter": [
                {"action": "speech", "mode": "audio", "audio": "audio/intro.wav", "await": True},
                {"action": "adjudicate"},
            ],
            "on_event": {
                "adjudication.result": {"to": "result"},
                "adjudication.diagnosis": {"to": "failed"},
            },
        },
        "failed": {
            "on_enter": [{"action": "speech", "mode": "tts_local", "text": "重试"}],
            "on_intent": {"retry": {"to": "analysis"}},
        },
        "result": {
            "on_enter": [{
                "action": "speech",
                "select_by": "winner_role",
                "cases": {
                    "PLAYER": {"mode": "tts_local", "text": "玩家胜 {player_score}"},
                    "AGENT": {"mode": "tts_local", "text": "Agent 胜 {agent_score}"},
                    "TIE": {"mode": "tts_local", "text": "平局"},
                },
            }],
            "on_intent": {"new_round": {"to": "rules"}},
        },
    },
}


def _write_round_games_root(tmp_path, with_audio=True):
    games_root = tmp_path / "games"
    game_dir = games_root / "dice"
    game_dir.mkdir(parents=True)
    if with_audio:
        (game_dir / "audio").mkdir()
        (game_dir / "audio" / "intro.wav").write_bytes(WAV)
    real_profile = json.loads(
        (ROOT / "backend/games/dice/manifest.json").read_text(encoding="utf-8")
    )["vision_profile"]
    manifest = {
        "id": "dice",
        "name": "Dice",
        "enabled": True,
        "participants": {"player": "LEFT", "agent": "RIGHT"},
        "providers": {
            "tts_local": "tts_dummy",
            "tts_remote": "tts_dummy_remote",
            "vision_adjudicator": "vision_dummy",
        },
        "voice": "announcer",
        "speed": 1.25,
        "state_machine": ROUND_MACHINE,
        "vision_profile": real_profile,
    }
    (game_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return games_root


def _round_server_env(monkeypatch, tmp_path, with_audio=True):
    games_root = _write_round_games_root(tmp_path, with_audio=with_audio)
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
    registry.register(DummyRemoteTts(), {
        "id": "tts_dummy_remote", "type": "tts", "name": "Dummy Remote TTS",
        "version": "1", "enabled": True, "entry": "provider.py:DummyRemoteTts",
    })
    monkeypatch.setattr(server, "COMPONENTS", registry)
    monkeypatch.setattr(server, "_GAMES_ROOT", games_root)
    monkeypatch.setattr(server, "GAMES", server.load_games(games_root))
    monkeypatch.setattr(server, "_GAMES_MTIMES", server._manifest_mtimes())
    monkeypatch.setattr(server, "rounds", {})
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread, httpd.server_address[1]


def _round_wait_until(port, round_id, predicate, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        connection = HTTPConnection("127.0.0.1", port, timeout=3)
        connection.request("GET", f"/api/game/rounds/{round_id}")
        response = connection.getresponse()
        snapshot = json.loads(response.read())
        connection.close()
        if predicate(snapshot):
            return snapshot
        time.sleep(0.05)
    raise AssertionError(f"round condition not reached: {snapshot}")


def _round_speech_events(snapshot):
    return [e for e in snapshot["events"] if e.get("event") == "speech"]


def _round_post(port, path, payload=None):
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    body = json.dumps(payload or {}).encode("utf-8")
    connection.request(
        "POST", path, body=body, headers={"Content-Type": "application/json"}
    )
    response = connection.getresponse()
    data = response.read()
    headers = dict(response.getheaders())
    connection.close()
    return response.status, headers, data


def test_round_api_creates_and_drives_full_flow(tmp_path, monkeypatch):
    httpd, thread, port = _round_server_env(monkeypatch, tmp_path)
    try:
        status, _, data = _round_post(port, "/api/game/rounds", {"game": "dice"})
        assert status == 201
        snapshot = json.loads(data)
        round_id = snapshot["round_id"]
        assert snapshot["status"] == "running"
        assert snapshot["state"] == "rules"

        # on_enter speech emits a directive; the frame endpoint resolves it
        # through the local slot provider.
        snapshot = _round_wait_until(
            port, round_id, lambda s: _round_speech_events(s)
        )
        directive = _round_speech_events(snapshot)[0]
        assert directive["mode"] == "tts_local"
        assert directive["voice"] == "announcer"
        assert directive["speed"] == 1.25

        status, headers, data = _round_post(
            port, f"/api/game/rounds/{round_id}/speech",
            {"directive_id": directive["directive_id"]},
        )
        assert status == 200
        assert headers["X-Dice-TTS-Provider"] == "tts_dummy"
        assert int.from_bytes(data[:4], "big") == len(WAV)
        assert data[4:-4] == WAV
        assert data[-4:] == b"\0\0\0\0"

        status, _, _ = _round_post(
            port, f"/api/game/rounds/{round_id}/intents", {"intent": "confirm"}
        )
        assert status == 200
        snapshot = _round_wait_until(
            port, round_id, lambda s: s["state"] == "analysis"
        )
        # The awaited audio directive must block adjudication until acked.
        audio = [e for e in _round_speech_events(snapshot) if e.get("mode") == "audio"]
        assert audio, snapshot["events"]

        status, headers, data = _round_post(
            port, f"/api/game/rounds/{round_id}/speech",
            {"directive_id": audio[0]["directive_id"]},
        )
        assert status == 200
        assert headers["X-Dice-Speech-Mode"] == "audio"
        assert data[4:-4] == WAV

        status, _, _ = _round_post(
            port, f"/api/game/rounds/{round_id}/intents",
            {"intent": "speech_done", "directive_id": audio[0]["directive_id"]},
        )
        assert status == 200

        snapshot = _round_wait_until(
            port, round_id, lambda s: s["state"] == "result"
        )
        assert snapshot["result"]["winner_role"] == "PLAYER"
        assert snapshot["result"]["player_score"] == 20
        announce = [
            e for e in _round_speech_events(snapshot)
            if "玩家胜" in e.get("text", "")
        ]
        assert announce and announce[0]["text"] == "玩家胜 20"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_round_api_rejects_unknown_intent_and_unknown_directive(tmp_path, monkeypatch):
    httpd, thread, port = _round_server_env(monkeypatch, tmp_path)
    try:
        _, _, data = _round_post(port, "/api/game/rounds", {"game": "dice"})
        round_id = json.loads(data)["round_id"]

        status, _, data = _round_post(
            port, f"/api/game/rounds/{round_id}/intents", {"intent": "stop_shake"}
        )
        assert status == 409
        assert "ROUND_INTENT_REJECTED" in data.decode("utf-8")

        status, _, data = _round_post(
            port, f"/api/game/rounds/{round_id}/speech", {"directive_id": "ghost"}
        )
        assert status == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_round_api_cancel_and_supersede_end_running_rounds(tmp_path, monkeypatch):
    httpd, thread, port = _round_server_env(monkeypatch, tmp_path)
    try:
        _, _, data = _round_post(port, "/api/game/rounds", {"game": "dice"})
        first_id = json.loads(data)["round_id"]
        # Creating the next round supersedes (cancels) the stale one.
        status, _, data = _round_post(port, "/api/game/rounds", {"game": "dice"})
        assert status == 201
        second_id = json.loads(data)["round_id"]
        _round_wait_until(port, first_id, lambda s: s["status"] == "cancelled")

        status, _, _ = _round_post(
            port, f"/api/game/rounds/{second_id}/cancel"
        )
        assert status == 200
        snapshot = _round_wait_until(port, second_id, lambda s: s["status"] == "cancelled")
        assert snapshot["status"] == "cancelled"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_round_api_exit_intent_and_sse_stream(tmp_path, monkeypatch):
    httpd, thread, port = _round_server_env(monkeypatch, tmp_path)
    try:
        _, _, data = _round_post(port, "/api/game/rounds", {"game": "dice"})
        round_id = json.loads(data)["round_id"]
        _round_post(port, f"/api/game/rounds/{round_id}/intents", {"intent": "back"})

        connection = HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request("GET", f"/api/game/rounds/{round_id}/stream")
        response = connection.getresponse()
        body = response.read().decode("utf-8")
        connection.close()
        assert response.status == 200
        assert "event: snapshot" in body
        assert "event: complete" in body
        assert '"round_complete"' in body

        # A finished round rejects further intents.
        status, _, _ = _round_post(
            port, f"/api/game/rounds/{round_id}/intents", {"intent": "confirm"}
        )
        assert status == 409
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


class ServerApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_components = server.COMPONENTS
        cls.original_games = server.GAMES
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
        registry.register(DummyRemoteTts(), {
            "id": "tts_dummy_remote", "type": "tts", "name": "Dummy Remote TTS",
            "version": "1", "enabled": True, "entry": "provider.py:DummyRemoteTts",
        })
        server.COMPONENTS = registry
        # Provider selection now has a single configuration source: swap in a
        # game registry whose dice manifest routes the semantic slots to the
        # dummy providers, exactly like a deployment manifest would.
        games = GameRegistry()
        dice_manifest = dict(server.GAMES.get("dice"))
        dice_manifest["providers"] = {
            "tts_local": "tts_dummy",
            "tts_remote": "tts_dummy_remote",
            "vision_adjudicator": "vision_dummy",
        }
        games.register(dice_manifest)
        server.GAMES = games
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
        server.GAMES = cls.original_games

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
        self.assertEqual(payload["tts_remote_provider"], "tts_dummy_remote")
        self.assertTrue(payload["tts_remote"]["configured"])

    def test_tts_endpoints_use_local_slot_provider(self):
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
            "id": "dice", "name": "Dice", "enabled": True, "providers": {},
            "participants": {"player": "RIGHT", "agent": "LEFT"},
            "state_machine": {
                "initial": "rules",
                "states": {"rules": {"on_enter": [
                    {"action": "speech", "mode": "tts_local", "text": "PRIVATE LINE"}
                ]}},
            },
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
        for secret in ("PRIVATE PROMPT", "SECRET", "secret-model", "/tmp/private.onnx", "/dev/video1", "PRIVATE LINE"):
            self.assertNotIn(secret, serialized)

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


def test_round_cancel_drains_body_on_keep_alive_connection(tmp_path, monkeypatch):
    """An unread cancel body must not poison the next request on the connection.

    The browser reuses one HTTP/1.1 connection; a POST body the handler never
    reads is parsed as the next request line ("{}POST ...") and fails it with
    501 — which surfaced as "对局创建失败" on the second game entry.
    """
    httpd, thread, port = _round_server_env(monkeypatch, tmp_path)
    try:
        connection = HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request(
            "POST", "/api/game/rounds",
            body=json.dumps({"game": "dice"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        body = response.read()
        assert response.status == 201, body
        round_id = json.loads(body)["round_id"]

        # Cancel carries a body on this same keep-alive connection.
        connection.request(
            "POST", f"/api/game/rounds/{round_id}/cancel",
            body="{}",
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        response.read()
        assert response.status == 200

        # The next request on the same connection must not see stale bytes.
        connection.request(
            "POST", "/api/game/rounds",
            body=json.dumps({"game": "dice"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        body = response.read()
        assert response.status == 201, (response.status, body)
        connection.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)
