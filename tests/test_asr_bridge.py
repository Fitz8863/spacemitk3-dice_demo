from __future__ import annotations

import json
import sys
import threading
import time
import unittest
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import pytest  # noqa: E402

import server  # noqa: E402
from core.asr import AsrProvider, AsrSessionError  # noqa: E402
from core.asr_bridge import (  # noqa: E402
    AsrIntentBridge,
    match_phrase_intent,
    normalize_speech_text,
)
from core.components import ComponentRegistry  # noqa: E402
from core.games import public_game_manifest, validate_asr_section  # noqa: E402
from core.state_machine import GameRound, IntentRejectedError  # noqa: E402
from core.tts import TtsProvider  # noqa: E402
import core.asr_bridge as asr_bridge_module  # noqa: E402


WAV = b"RIFF" + b"\0" * 4 + b"WAVE" + b"\0" * 40


def _round_manifest(asr=None):
    manifest = {
        "id": "dice",
        "providers": {"tts_local": "tts_dummy"},
        "state_machine": {
            "schema_version": 1,
            "initial": "rules",
            "states": {
                "rules": {
                    "on_enter": [{"action": "speech", "mode": "tts_local", "text": "规则"}],
                    "on_intent": {"confirm": {"to": "ready"}},
                },
                "ready": {
                    "on_enter": [{"action": "speech", "mode": "tts_local", "text": "准备"}],
                    "on_intent": {"back": {"exit": True}},
                },
            },
        },
    }
    if asr is not None:
        manifest["asr"] = asr
    return manifest


class PhraseMatchingTests(unittest.TestCase):
    def test_normalize_strips_whitespace_and_lowercases(self):
        self.assertEqual(normalize_speech_text("  OK 好 的 "), "ok好的")

    def test_match_finds_substring_trigger(self):
        phrases = {"confirm": ["确认"], "back": ["返回"]}
        self.assertEqual(match_phrase_intent(phrases, "那我就确认了"), "confirm")
        self.assertEqual(match_phrase_intent(phrases, "返回 吧"), "back")

    def test_match_tolerates_case_and_spacing(self):
        self.assertEqual(match_phrase_intent({"start": ["START"]}, "go start now"), "start")

    def test_no_match_returns_none(self):
        self.assertIsNone(match_phrase_intent({"confirm": ["确认"]}, "今天天气不错"))
        self.assertIsNone(match_phrase_intent({"confirm": ["确认"]}, "  "))


class SpeechGateTests(unittest.TestCase):
    """The engine's speech registration backs the ASR gate."""

    def _round(self, **kwargs):
        round_ = GameRound(
            game_id="dice",
            manifest=_round_manifest(),
            log=lambda _line: None,
            **kwargs,
        )
        round_.start()
        return round_

    def _wait_for_speech(self, round_, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snapshot = round_.snapshot()
            directive = next(
                (
                    event
                    for event in snapshot["events"]
                    if event.get("event") == "speech"
                ),
                None,
            )
            if directive:
                return snapshot, directive
            time.sleep(0.01)
        raise AssertionError("speech directive never emitted")

    def test_speech_registers_gate_until_acknowledged(self):
        round_ = self._round()
        snapshot, directive = self._wait_for_speech(round_)
        self.assertTrue(round_.speech_active)
        self.assertTrue(snapshot["speech_active"])
        round_.submit_intent("speech_done", {"directive_id": directive["directive_id"]})
        self.assertFalse(round_.speech_active)
        round_.cancel()

    def test_acknowledging_unknown_directive_is_ignored(self):
        round_ = self._round()
        self._wait_for_speech(round_)
        round_.submit_intent("speech_done", {"directive_id": "nope"})
        self.assertTrue(round_.speech_active)
        round_.cancel()

    def test_unacknowledged_directive_expires(self):
        round_ = self._round(speech_ack_fallback_seconds=0.05)
        self._wait_for_speech(round_)
        deadline = time.monotonic() + 1.0
        while round_.speech_active and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(round_.speech_active)

    def test_terminal_round_clears_gate(self):
        round_ = self._round()
        self._wait_for_speech(round_)
        round_.cancel()
        self.assertFalse(round_.speech_active)


class FakeRound:
    def __init__(self, manifest, *, speech_active=False, status="running", rejected=()):
        self.manifest = manifest
        self.id = "abc12345"
        self._speech_active = speech_active
        self.status = status
        self.rejected = set(rejected)
        self.submitted = []

    @property
    def speech_active(self):
        return self._speech_active

    def submit_intent(self, name, payload=None):
        if self.status != "running" or name in self.rejected:
            raise IntentRejectedError(self.id, "rules", name)
        self.submitted.append((name, dict(payload or {})))


class FakeAsr(AsrProvider):
    id = "asr_dummy"
    type = "asr"

    def __init__(self, *, raise_on_start=False):
        self.sessions = []
        self.stopped = []
        self._raise_on_start = raise_on_start

    def health(self):
        return {"id": self.id, "type": self.type, "ok": True}

    def start_session(self, on_sentence, *, on_log=None):
        if self._raise_on_start:
            raise AsrSessionError("dummy refused to start")
        session = {"on_sentence": on_sentence, "on_log": on_log, "alive": True}
        self.sessions.append(session)
        return session

    def stop_session(self, handle):
        handle["alive"] = False
        self.stopped.append(handle)


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.provider = FakeAsr()
        self.registry = ComponentRegistry()
        self.registry.register(self.provider, {
            "id": "asr_dummy",
            "type": "asr",
            "entry": "provider.py:FakeAsr",
        })
        self.logs = []
        self.bridge = AsrIntentBridge(
            components=self.registry, log=self.logs.append
        )

    def _enabled_manifest(self):
        manifest = _round_manifest(asr={
            "enabled": True,
            "phrases": {"confirm": ["确认"], "back": ["返回"]},
        })
        manifest["providers"]["asr"] = "asr_dummy"
        return manifest

    def test_disabled_manifest_starts_nothing(self):
        round_ = FakeRound(_round_manifest(asr={"enabled": False, "phrases": {"confirm": ["确认"]}}))
        self.assertFalse(self.bridge.start_for_round(round_))
        self.assertEqual(self.provider.sessions, [])

    def test_missing_provider_slot_is_logged(self):
        manifest = self._enabled_manifest()
        manifest["providers"] = {}
        round_ = FakeRound(manifest)
        self.assertFalse(self.bridge.start_for_round(round_))
        self.assertTrue(any("providers.asr" in line for line in self.logs))

    def test_session_starts_and_submits_matched_intent(self):
        round_ = FakeRound(self._enabled_manifest())
        self.assertTrue(self.bridge.start_for_round(round_))
        self.assertEqual(len(self.provider.sessions), 1)
        self.provider.sessions[0]["on_sentence"]("那我就确认了")
        self.assertEqual(round_.submitted, [("confirm", {"source": "asr", "text": "那我就确认了"})])

    def test_unmatched_sentence_is_ignored(self):
        round_ = FakeRound(self._enabled_manifest())
        self.bridge.start_for_round(round_)
        self.provider.sessions[0]["on_sentence"]("今天天气不错")
        self.assertEqual(round_.submitted, [])

    def test_speech_gate_suppresses_matching(self):
        round_ = FakeRound(self._enabled_manifest(), speech_active=True)
        self.bridge.start_for_round(round_)
        self.provider.sessions[0]["on_sentence"]("确认")
        self.assertEqual(round_.submitted, [])
        self.assertTrue(any("speech is playing" in line for line in self.logs))

    def test_rejected_intent_is_swallowed(self):
        round_ = FakeRound(self._enabled_manifest(), rejected={"confirm"})
        self.bridge.start_for_round(round_)
        self.provider.sessions[0]["on_sentence"]("确认")  # must not raise
        self.assertEqual(round_.submitted, [])

    def test_provider_startup_failure_returns_false(self):
        provider = FakeAsr(raise_on_start=True)
        registry = ComponentRegistry()
        registry.register(provider, {"id": "asr_dummy", "type": "asr", "entry": "provider.py:FakeAsr"})
        bridge = AsrIntentBridge(components=registry, log=self.logs.append)
        self.assertFalse(bridge.start_for_round(FakeRound(self._enabled_manifest())))
        self.assertTrue(any("failed to start" in line for line in self.logs))

    @patch.object(asr_bridge_module, "_WATCHER_POLL_SECONDS", 0.05)
    def test_round_ending_stops_session(self):
        round_ = FakeRound(self._enabled_manifest())
        self.bridge.start_for_round(round_)
        round_.status = "exited"
        deadline = time.monotonic() + 2.0
        while not self.provider.stopped and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(self.provider.stopped), 1)
        self.assertFalse(self.provider.stopped[0]["alive"])

    @patch.object(asr_bridge_module, "_WATCHER_POLL_SECONDS", 0.05)
    def test_new_round_stops_previous_session(self):
        first = FakeRound(self._enabled_manifest())
        self.bridge.start_for_round(first)
        second = FakeRound(self._enabled_manifest())
        self.bridge.start_for_round(second)
        self.assertEqual(len(self.provider.stopped), 1)
        self.assertEqual(len(self.provider.sessions), 2)


class ValidateAsrSectionTests(unittest.TestCase):
    def _machine(self):
        return _round_manifest()["state_machine"]

    def test_valid_section_passes(self):
        result = validate_asr_section(
            {"enabled": True, "phrases": {"confirm": ["确认"]}}, self._machine()
        )
        self.assertEqual(result["phrases"], {"confirm": ["确认"]})

    def test_unknown_intent_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_asr_section(
                {"enabled": True, "phrases": {"confim": ["确认"]}}, self._machine()
            )

    def test_builtin_speech_done_is_not_remappable(self):
        with self.assertRaises(ValueError):
            validate_asr_section(
                {"enabled": True, "phrases": {"speech_done": ["完成"]}}, self._machine()
            )

    def test_phrase_list_must_be_non_empty_strings(self):
        with self.assertRaises(ValueError):
            validate_asr_section({"enabled": True, "phrases": {"confirm": []}}, self._machine())
        with self.assertRaises(ValueError):
            validate_asr_section({"enabled": True, "phrases": {"confirm": [" "]}}, self._machine())

    def test_enabled_must_be_boolean(self):
        with self.assertRaises(ValueError):
            validate_asr_section(
                {"enabled": "yes", "phrases": {"confirm": ["确认"]}}, self._machine()
            )

    def test_public_projection_exposes_only_enabled(self):
        manifest = _round_manifest(asr={"enabled": True, "phrases": {"confirm": ["确认"]}})
        public = public_game_manifest(manifest)
        self.assertEqual(public["asr"], {"enabled": True})
        self.assertNotIn("phrases", public["asr"])


# ---- server-level integration (real engine + real bridge, dummy provider) ----

ASR_MACHINE = {
    "schema_version": 1,
    "initial": "rules",
    "states": {
        "rules": {
            "on_enter": [{"action": "speech", "mode": "tts_local", "text": "规则"}],
            "on_intent": {"confirm": {"to": "ready"}},
        },
        "ready": {
            "on_enter": [{"action": "speech", "mode": "tts_local", "text": "准备"}],
            "on_intent": {"back": {"exit": True}},
        },
    },
}


class DummyTtsForAsr(TtsProvider):
    id = "tts_dummy"

    def health(self):
        return {"id": self.id, "type": self.type, "ok": True}

    def synthesize(self, payload):
        self.validate(payload)
        return WAV, {"Content-Type": "audio/wav"}


def _write_asr_games_root(tmp_path, asr_section):
    games_root = tmp_path / "games"
    (games_root / "dice").mkdir(parents=True)
    manifest = {
        "id": "dice",
        "name": "Dice",
        "enabled": True,
        "participants": {"player": "LEFT", "agent": "RIGHT"},
        "providers": {"tts_local": "tts_dummy", "asr": "asr_dummy"},
        "state_machine": ASR_MACHINE,
    }
    if asr_section is not None:
        manifest["asr"] = asr_section
    (games_root / "dice" / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return games_root


def test_round_with_asr_enabled_starts_session_and_voice_confirms(tmp_path, monkeypatch):
    games_root = _write_asr_games_root(tmp_path, {
        "enabled": True,
        "phrases": {"confirm": ["确认"], "back": ["返回"]},
    })
    provider = FakeAsr()
    registry = ComponentRegistry()
    registry.register(DummyTtsForAsr(), {
        "id": "tts_dummy", "type": "tts", "entry": "provider.py:DummyTtsForAsr",
    })
    registry.register(provider, {
        "id": "asr_dummy", "type": "asr", "entry": "provider.py:FakeAsr",
    })
    monkeypatch.setattr(server, "COMPONENTS", registry)
    monkeypatch.setattr(server, "_GAMES_ROOT", games_root)
    monkeypatch.setattr(server, "GAMES", server.load_games(games_root))
    monkeypatch.setattr(server, "_GAMES_MTIMES", server._manifest_mtimes())
    monkeypatch.setattr(server, "rounds", {})
    logs = []
    monkeypatch.setattr(
        server, "ASR_BRIDGE", AsrIntentBridge(components=registry, log=logs.append)
    )
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    try:
        connection = HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request("POST", "/api/game/rounds", body=b'{"game":"dice"}',
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        snapshot = json.loads(response.read())
        connection.close()
        assert response.status == 201
        round_id = snapshot["round_id"]
        assert snapshot["state"] == "rules"

        # The ASR session must be listening for this round.
        deadline = time.monotonic() + 5.0
        while not provider.sessions and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(provider.sessions) == 1

        # While the rules announcement is playing, voice confirmation is gated.
        deadline = time.monotonic() + 5.0
        directive = None
        while directive is None and time.monotonic() < deadline:
            connection = HTTPConnection("127.0.0.1", port, timeout=3)
            connection.request("GET", f"/api/game/rounds/{round_id}")
            snapshot = json.loads(connection.getresponse().read())
            connection.close()
            directive = next(
                (e for e in snapshot["events"] if e.get("event") == "speech"), None
            )
        assert directive is not None
        provider.sessions[0]["on_sentence"]("确认")
        assert any("speech is playing" in line for line in logs)

        # After the announcement is acknowledged, the same sentence confirms.
        connection = HTTPConnection("127.0.0.1", port, timeout=3)
        connection.request(
            "POST", f"/api/game/rounds/{round_id}/intents",
            body=json.dumps({"intent": "speech_done",
                             "directive_id": directive["directive_id"]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        connection.getresponse().read()
        connection.close()
        provider.sessions[0]["on_sentence"]("确认")

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            connection = HTTPConnection("127.0.0.1", port, timeout=3)
            connection.request("GET", f"/api/game/rounds/{round_id}")
            snapshot = json.loads(connection.getresponse().read())
            connection.close()
            if snapshot["state"] == "ready":
                break
            time.sleep(0.05)
        assert snapshot["state"] == "ready"
    finally:
        httpd.shutdown()


def test_round_with_asr_disabled_starts_no_session(tmp_path, monkeypatch):
    games_root = _write_asr_games_root(tmp_path, {
        "enabled": False,
        "phrases": {"confirm": ["确认"]},
    })
    provider = FakeAsr()
    registry = ComponentRegistry()
    registry.register(DummyTtsForAsr(), {
        "id": "tts_dummy", "type": "tts", "entry": "provider.py:DummyTtsForAsr",
    })
    registry.register(provider, {
        "id": "asr_dummy", "type": "asr", "entry": "provider.py:FakeAsr",
    })
    monkeypatch.setattr(server, "COMPONENTS", registry)
    monkeypatch.setattr(server, "_GAMES_ROOT", games_root)
    monkeypatch.setattr(server, "GAMES", server.load_games(games_root))
    monkeypatch.setattr(server, "_GAMES_MTIMES", server._manifest_mtimes())
    monkeypatch.setattr(server, "rounds", {})
    monkeypatch.setattr(
        server, "ASR_BRIDGE", AsrIntentBridge(components=registry, log=lambda _l: None)
    )
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    try:
        connection = HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request("POST", "/api/game/rounds", body=b'{"game":"dice"}',
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        snapshot = json.loads(response.read())
        connection.close()
        assert response.status == 201
        assert provider.sessions == []  # disabled means no listening at all
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    unittest.main()
