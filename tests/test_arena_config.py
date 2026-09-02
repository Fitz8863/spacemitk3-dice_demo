from __future__ import annotations

import json
import sys
import threading
import time
import types
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
from core.asr import AsrProvider  # noqa: E402
from core.asr_bridge import AsrIntentBridge  # noqa: E402
from core.arena_config import (  # noqa: E402
    ArenaConfigError,
    arena_asr_enabled,
    arena_slot_value,
    load_arena_config,
    resolve_local_tts_pin,
    validate_arena_config,
    with_global_defaults,
)
from core.components import ComponentRegistry  # noqa: E402
from core.games import GameRegistry, run_game  # noqa: E402
from core.tts import TtsProvider  # noqa: E402
from core.tts_dispatch import TtsDispatcher  # noqa: E402


WAV = b"RIFF" + b"\0" * 4 + b"WAVE" + b"\0" * 40

VALID_ARENA = {
    "schema_version": 1,
    "providers": {
        "tts_local": "tts_moss_nano",
        "tts_remote": "tts_gptsovits",
        "asr": "asr_zipformer",
        "vision_adjudicator": "vision_yolov8_adjudicator",
    },
    "voice": "default",
    "speed": 1.0,
    "asr_enabled": True,
}


class ValidationTests(unittest.TestCase):
    def test_valid_config_passes(self):
        self.assertEqual(validate_arena_config(VALID_ARENA), VALID_ARENA)

    def test_wrong_schema_version(self):
        with self.assertRaises(ArenaConfigError):
            validate_arena_config({**VALID_ARENA, "schema_version": 2})

    def test_providers_must_map_to_nonempty_ids(self):
        with self.assertRaises(ArenaConfigError):
            validate_arena_config({**VALID_ARENA, "providers": {"tts_local": "  "}})
        with self.assertRaises(ArenaConfigError):
            validate_arena_config({**VALID_ARENA, "providers": ["tts_moss_nano"]})

    def test_speed_must_be_positive_number(self):
        with self.assertRaises(ArenaConfigError):
            validate_arena_config({**VALID_ARENA, "speed": 0})
        with self.assertRaises(ArenaConfigError):
            validate_arena_config({**VALID_ARENA, "speed": True})

    def test_asr_enabled_must_be_boolean(self):
        with self.assertRaises(ArenaConfigError):
            validate_arena_config({**VALID_ARENA, "asr_enabled": "yes"})


class LoadTests(unittest.TestCase):
    def test_missing_file_means_no_defaults(self):
        self.assertEqual(load_arena_config("/nonexistent/config.json"), {})

    def test_broken_file_raises(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text("{ broken", encoding="utf-8")
            with self.assertRaises(ArenaConfigError):
                load_arena_config(path)

    def test_valid_file_loads(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(VALID_ARENA), encoding="utf-8")
            self.assertEqual(load_arena_config(path), VALID_ARENA)


class AccessorTests(unittest.TestCase):
    def test_slot_value_and_defaults(self):
        self.assertEqual(arena_slot_value(VALID_ARENA, "tts_local"), "tts_moss_nano")
        self.assertEqual(arena_slot_value(VALID_ARENA, "missing_slot"), "")
        self.assertEqual(arena_slot_value(None, "tts_local"), "")
        self.assertTrue(arena_asr_enabled(VALID_ARENA))
        self.assertTrue(arena_asr_enabled(None))
        self.assertFalse(arena_asr_enabled({**VALID_ARENA, "asr_enabled": False}))


class MergeTests(unittest.TestCase):
    def test_game_providers_win_per_slot(self):
        manifest = {"providers": {"tts_local": "tts_qwen3"}}
        merged = with_global_defaults(manifest, VALID_ARENA)
        self.assertEqual(merged["providers"]["tts_local"], "tts_qwen3")
        self.assertEqual(merged["providers"]["tts_remote"], "tts_gptsovits")
        self.assertEqual(merged["providers"]["asr"], "asr_zipformer")

    def test_game_without_providers_gets_global_slots(self):
        merged = with_global_defaults({"id": "dice"}, VALID_ARENA)
        self.assertEqual(merged["providers"]["vision_adjudicator"], "vision_yolov8_adjudicator")

    def test_voice_speed_underlay_only_when_absent(self):
        merged = with_global_defaults({"voice": "announcer"}, VALID_ARENA)
        self.assertEqual(merged["voice"], "announcer")
        self.assertEqual(merged["speed"], 1.0)
        merged = with_global_defaults({"voice": "a", "speed": 1.5}, VALID_ARENA)
        self.assertEqual(merged["speed"], 1.5)

    def test_asr_breaker_ands_with_game_switch(self):
        game = {"asr": {"enabled": True, "phrases": {"confirm": ["确认"]}}}
        merged = with_global_defaults(game, VALID_ARENA)
        self.assertTrue(merged["asr"]["enabled"])
        merged = with_global_defaults(game, {**VALID_ARENA, "asr_enabled": False})
        self.assertFalse(merged["asr"]["enabled"])
        # A game without an asr section gains nothing.
        merged = with_global_defaults({"id": "dice"}, {**VALID_ARENA, "asr_enabled": False})
        self.assertNotIn("asr", merged)

    def test_source_manifest_is_not_mutated(self):
        manifest = {"providers": {"tts_local": "tts_qwen3"}, "asr": {"enabled": True}}
        with_global_defaults(manifest, {**VALID_ARENA, "asr_enabled": False})
        self.assertEqual(manifest["providers"], {"tts_local": "tts_qwen3"})
        self.assertTrue(manifest["asr"]["enabled"])


class LocalTtsPinTests(unittest.TestCase):
    def test_arena_only(self):
        self.assertEqual(resolve_local_tts_pin(VALID_ARENA, []), "tts_moss_nano")

    def test_game_override_agreeing_with_arena(self):
        games = [{"enabled": True, "providers": {"tts_local": "tts_moss_nano"}}]
        self.assertEqual(resolve_local_tts_pin(VALID_ARENA, games), "tts_moss_nano")

    def test_conflicting_selections_refuse(self):
        games = [{"enabled": True, "providers": {"tts_local": "tts_qwen3"}}]
        with self.assertRaises(ArenaConfigError):
            resolve_local_tts_pin(VALID_ARENA, games)

    def test_two_games_disagreeing_refuse(self):
        arena = {k: v for k, v in VALID_ARENA.items() if k != "providers"}
        games = [
            {"enabled": True, "providers": {"tts_local": "tts_moss_nano"}},
            {"enabled": True, "providers": {"tts_local": "tts_qwen3"}},
        ]
        with self.assertRaises(ArenaConfigError):
            resolve_local_tts_pin(arena, games)

    def test_disabled_games_are_ignored(self):
        games = [
            {"enabled": True, "providers": {"tts_local": "tts_moss_nano"}},
            {"enabled": False, "providers": {"tts_local": "tts_qwen3"}},
        ]
        self.assertEqual(resolve_local_tts_pin(VALID_ARENA, games), "tts_moss_nano")

    def test_nothing_configured_is_none(self):
        self.assertIsNone(resolve_local_tts_pin({}, []))


class DummyTts(TtsProvider):
    id = "tts_dummy"

    def health(self):
        return {"id": self.id, "type": self.type, "ok": True}

    def synthesize(self, payload):
        self.validate(payload)
        return WAV, {"Content-Type": "audio/wav"}


class DispatcherLadderTests(unittest.TestCase):
    def setUp(self):
        self.registry = ComponentRegistry()
        self.registry.register(DummyTts(), {
            "id": "tts_dummy", "type": "tts", "entry": "provider.py:DummyTts",
        })
        self.games = GameRegistry()
        self.games.register({
            "id": "dice", "name": "Dice", "enabled": True,
            "providers": {"tts_local": "tts_dummy"},
        })
        self.games.register({
            "id": "bare", "name": "Bare", "enabled": True,
        })

    def test_pin_overrides_manifest(self):
        dispatcher = TtsDispatcher(
            self.registry, self.games, pinned_local_tts="tts_dummy"
        )
        self.assertEqual(dispatcher.provider_id("dice"), "tts_dummy")

    def test_arena_fallback_fills_unconfigured_slots(self):
        arena = {"providers": {"tts_local": "tts_dummy", "tts_remote": "tts_dummy"}}
        dispatcher = TtsDispatcher(
            self.registry, self.games,
            slot_fallbacks=lambda slot: arena_slot_value(arena, slot),
        )
        # "bare" declares no providers at all: the arena provides both slots.
        self.assertEqual(dispatcher.provider_id("bare"), "tts_dummy")
        entry = {"mode": "tts_remote"}
        self.assertEqual(dispatcher.provider_id_for_speech_entry(entry, "bare"), "tts_dummy")

    def test_builtin_default_still_applies_without_arena(self):
        dispatcher = TtsDispatcher(self.registry, self.games)
        # "bare" has no tts_local anywhere: the historic builtin default wins.
        self.assertEqual(dispatcher.provider_id("bare"), "tts_qwen3")

    def test_remote_without_any_source_raises(self):
        dispatcher = TtsDispatcher(self.registry, self.games)
        with self.assertRaises(Exception):
            dispatcher.provider_id_for_speech_entry({"mode": "tts_remote"}, "bare")


class RunGameDefaultsTests(unittest.TestCase):
    def test_defaults_underlay_the_pipeline_manifest(self):
        recorded = {}

        def fake_run(on_log, is_cancelled, timeout_seconds, *, components, manifest, on_event):
            recorded["manifest"] = manifest
            return {"ok": True}

        registry = GameRegistry()
        registry.register({
            "id": "dice", "name": "Dice", "enabled": True,
            "providers": {"tts_local": "tts_moss_nano"},
        })
        stub = types.ModuleType("games.dice.pipeline")
        stub.run = fake_run
        with patch.dict(sys.modules, {"games.dice.pipeline": stub}):
            result = run_game(
                registry, "dice", lambda _l: None, lambda: False, lambda _e: None,
                30, components=None,
                defaults={"providers": {"vision_adjudicator": "vision_yolov8_adjudicator"}},
            )
        self.assertEqual(result, {"ok": True})
        # The game slot wins; the arena fills what the manifest leaves open.
        self.assertEqual(recorded["manifest"]["providers"]["tts_local"], "tts_moss_nano")
        self.assertEqual(
            recorded["manifest"]["providers"]["vision_adjudicator"],
            "vision_yolov8_adjudicator",
        )


# ---- componentctl: arena-wide referenced collection + selected fallback ----

class ReferencedArenaTests(unittest.TestCase):
    @staticmethod
    def _game(providers=None, *, enabled=True, pins=()):
        on_enter = [
            {"action": "speech", "mode": "tts_local", "text": "台词", **({"provider": pin} if pin else {})}
            for pin in pins
        ] or [{"action": "speech", "mode": "tts_local", "text": "台词"}]
        manifest = {
            "id": "dice", "name": "Dice", "enabled": enabled,
            "state_machine": {
                "schema_version": 1, "initial": "rules",
                "states": {"rules": {"on_enter": on_enter}},
            },
        }
        if providers is not None:
            manifest["providers"] = providers
        return manifest

    def test_arena_slots_first_then_games_and_pins(self):
        from componentctl import _referenced_tts_providers_arena

        arena = {"providers": {"tts_local": "tts_a", "tts_remote": "tts_b"}}
        games = [self._game(providers={"tts_local": "tts_a", "tts_remote": "tts_c"}, pins=("tts_d",))]
        self.assertEqual(
            _referenced_tts_providers_arena(arena, games),
            ["tts_a", "tts_b", "tts_c", "tts_d"],
        )

    def test_game_override_agreeing_with_arena_deduplicates(self):
        from componentctl import _referenced_tts_providers_arena

        arena = {"providers": {"tts_local": "tts_a"}}
        games = [self._game(providers={"tts_local": "tts_a"})]
        self.assertEqual(_referenced_tts_providers_arena(arena, games), ["tts_a"])

    def test_historic_default_when_no_local_engine_anywhere(self):
        from componentctl import _referenced_tts_providers_arena

        arena = {"providers": {"tts_remote": "tts_c"}}
        games = [self._game(providers={"tts_remote": "tts_c"})]
        self.assertEqual(
            _referenced_tts_providers_arena(arena, games),
            ["tts_qwen3", "tts_c"],
        )

    def test_disabled_games_are_skipped(self):
        from componentctl import _referenced_tts_providers_arena

        arena = {"providers": {"tts_local": "tts_a"}}
        games = [
            self._game(providers={"tts_local": "tts_a"}),
            self._game(providers={"tts_local": "tts_x"}, enabled=False),
        ]
        self.assertEqual(_referenced_tts_providers_arena(arena, games), ["tts_a"])

    def test_selected_provider_id_falls_back_to_arena(self):
        import componentctl

        games_registry = GameRegistry()
        games_registry.register({
            "id": "dice", "name": "Dice", "enabled": True,
            "participants": {"player": "LEFT", "agent": "RIGHT"},
        })
        with patch.object(componentctl, "load_games", return_value=games_registry), \
                patch.object(componentctl, "load_arena_config",
                             return_value={"providers": {"tts_local": "tts_a",
                                                         "vision_adjudicator": "vision_x"}}):
            self.assertEqual(componentctl._selected_provider_id("tts_local", "dice"), "tts_a")
            self.assertEqual(
                componentctl._selected_provider_id("vision_adjudicator", "dice"), "vision_x"
            )
            # Slots with neither manifest nor arena value keep their builtin
            # fallbacks.
            self.assertEqual(componentctl._selected_provider_id("tts_local", "dice"), "tts_a")


# ---- server-level integration: the real wiring, tmp manifests + tmp arena ----

MACHINE = {
    "schema_version": 1,
    "initial": "rules",
    "states": {
        "rules": {
            "on_enter": [{"action": "speech", "mode": "tts_local", "text": "规则"}],
            "on_intent": {"confirm": {"to": "ready"}},
        },
        "ready": {"on_intent": {"back": {"exit": True}}},
    },
}


class DummyAsrForArena(AsrProvider):
    id = "asr_dummy"
    type = "asr"

    def __init__(self):
        self.sessions = []

    def health(self):
        return {"id": self.id, "type": self.type, "ok": True}

    def start_session(self, on_sentence, *, on_log=None):
        self.sessions.append({"on_sentence": on_sentence})
        return {"alive": True}

    def stop_session(self, handle):
        pass


def _setup_server(monkeypatch, tmp_path, *, manifest_providers, arena_payload):
    games_root = tmp_path / "games"
    (games_root / "dice").mkdir(parents=True)
    manifest = {
        "id": "dice",
        "name": "Dice",
        "enabled": True,
        "participants": {"player": "LEFT", "agent": "RIGHT"},
        "state_machine": MACHINE,
    }
    if manifest_providers is not None:
        manifest["providers"] = manifest_providers
    (games_root / "dice" / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    arena_path = tmp_path / "config.json"
    arena_path.write_text(json.dumps(arena_payload), encoding="utf-8")

    provider = DummyAsrForArena()
    registry = ComponentRegistry()
    registry.register(DummyTts(), {
        "id": "tts_dummy", "type": "tts", "entry": "provider.py:DummyTts",
    })
    registry.register(provider, {
        "id": "asr_dummy", "type": "asr", "entry": "provider.py:DummyAsrForArena",
    })
    monkeypatch.setattr(server, "COMPONENTS", registry)
    monkeypatch.setattr(server, "_GAMES_ROOT", games_root)
    monkeypatch.setattr(server, "GAMES", server.load_games(games_root))
    monkeypatch.setattr(server, "_GAMES_MTIMES", server._manifest_mtimes())
    monkeypatch.setattr(server, "rounds", {})
    monkeypatch.setattr(server, "ARENA_CONFIG_PATH", arena_path)
    monkeypatch.setattr(server, "_ARENA_CONFIG", {})
    monkeypatch.setattr(server, "_ARENA_MTIME", None)
    monkeypatch.setattr(server, "ASR_BRIDGE",
                        AsrIntentBridge(components=registry, log=lambda _l: None))
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, provider


def test_round_uses_arena_tts_slot_when_manifest_has_none(tmp_path, monkeypatch):
    """A game manifest without provider slots still speaks via the arena slots."""
    httpd, _provider = _setup_server(
        monkeypatch, tmp_path,
        manifest_providers=None,
        arena_payload={"schema_version": 1, "providers": {"tts_local": "tts_dummy"}},
    )
    port = httpd.server_address[1]
    try:
        connection = HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request("POST", "/api/game/rounds", body=b'{"game":"dice"}',
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        snapshot = json.loads(response.read())
        connection.close()
        assert response.status == 201

        deadline = time.monotonic() + 5.0
        directive = None
        while directive is None and time.monotonic() < deadline:
            connection = HTTPConnection("127.0.0.1", port, timeout=3)
            connection.request("GET", f"/api/game/rounds/{snapshot['round_id']}")
            current = json.loads(connection.getresponse().read())
            connection.close()
            directive = next(
                (e for e in current["events"] if e.get("event") == "speech"), None
            )
        assert directive is not None

        connection = HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request(
            "POST", f"/api/game/rounds/{snapshot['round_id']}/speech",
            body=json.dumps({"directive_id": directive["directive_id"]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        response.read()
        provider_header = response.getheader("X-Dice-TTS-Provider")
        connection.close()
        assert response.status == 200
        assert provider_header == "tts_dummy"
    finally:
        httpd.shutdown()


def test_arena_asr_breaker_disables_voice_input(tmp_path, monkeypatch):
    """asr_enabled=false kills ASR globally even when the game opts in."""
    httpd, provider = _setup_server(
        monkeypatch, tmp_path,
        manifest_providers={"asr": "asr_dummy"},
        arena_payload={"schema_version": 1, "providers": {"asr": "asr_dummy"},
                       "asr_enabled": False},
    )
    manifest_path = tmp_path / "games/dice/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["asr"] = {"enabled": True, "phrases": {"confirm": ["确认"]}}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    # bump mtime so the hot reload picks the game's asr section up
    import os

    os.utime(manifest_path, (time.time() + 5, time.time() + 5))
    port = httpd.server_address[1]
    try:
        connection = HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request("POST", "/api/game/rounds", body=b'{"game":"dice"}',
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        response.read()
        connection.close()
        assert response.status == 201
        time.sleep(0.5)
        assert provider.sessions == []
    finally:
        httpd.shutdown()


def test_startup_refuses_conflicting_local_tts(tmp_path, monkeypatch):
    """main() exits when the arena and a game name different local engines."""
    games_root = tmp_path / "games"
    (games_root / "dice").mkdir(parents=True)
    (games_root / "dice" / "manifest.json").write_text(json.dumps({
        "id": "dice", "name": "Dice", "enabled": True,
        "participants": {"player": "LEFT", "agent": "RIGHT"},
        "providers": {"tts_local": "tts_qwen3"},
        "state_machine": MACHINE,
    }), encoding="utf-8")
    arena_path = tmp_path / "config.json"
    arena_path.write_text(json.dumps({
        "schema_version": 1, "providers": {"tts_local": "tts_moss_nano"},
    }), encoding="utf-8")
    monkeypatch.setattr(server, "_GAMES_ROOT", games_root)
    monkeypatch.setattr(server, "GAMES", server.load_games(games_root))
    monkeypatch.setattr(server, "_GAMES_MTIMES", server._manifest_mtimes())
    monkeypatch.setattr(server, "ARENA_CONFIG_PATH", arena_path)
    monkeypatch.setattr(server, "_ARENA_CONFIG", {})
    monkeypatch.setattr(server, "_ARENA_MTIME", None)

    monkeypatch.setattr(sys, "argv", ["server.py", "--port", "0"])
    with pytest.raises(SystemExit):
        server.main()


if __name__ == "__main__":
    unittest.main()
