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
    arena_standby,
    load_arena_config,
    resolve_local_tts_pin,
    validate_arena_config,
    with_global_defaults,
)
from core.components import ComponentRegistry  # noqa: E402
from core.games import GameRegistry, run_game  # noqa: E402
from core.state_machine import GameRound  # noqa: E402
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
    "participants": {"player": "LEFT", "agent": "RIGHT"},
    "voice": "default",
    "speed": 1.0,
    "asr_enabled": True,
    "standby": {
        "enabled": True,
        "idle_seconds": 120,
        "boot_standby": True,
        "wake_phrases": ["小骰子", "醒醒"],
    },
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

    def test_standby_validation_and_defaults(self):
        # Validation: bad types rejected.
        with self.assertRaises(ArenaConfigError):
            validate_arena_config({**VALID_ARENA, "standby": {"idle_seconds": 1}})
        with self.assertRaises(ArenaConfigError):
            validate_arena_config({**VALID_ARENA, "standby": {"boot_standby": "yes"}})
        with self.assertRaises(ArenaConfigError):
            validate_arena_config({
                **VALID_ARENA, "standby": {"wake_phrases": ["  "]},
            })
        # Normalization: absent section falls back to browser-safe defaults.
        normalized = arena_standby(None)
        self.assertTrue(normalized["enabled"])
        self.assertEqual(normalized["idle_seconds"], 120)
        self.assertFalse(normalized["boot_standby"])
        self.assertEqual(normalized["wake_phrases"], [])
        normalized = arena_standby(VALID_ARENA)
        self.assertTrue(normalized["boot_standby"])
        self.assertEqual(normalized["wake_phrases"], ["小骰子", "醒醒"])


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

    def test_participants_underlay(self):
        arena = {**VALID_ARENA, "participants": {"player": "LEFT", "agent": "RIGHT"}}
        # A game that declares nothing inherits the deployment table mapping.
        merged = with_global_defaults({"id": "dice"}, arena)
        self.assertEqual(merged["participants"], {"player": "LEFT", "agent": "RIGHT"})
        # A game that declares its own mapping wins.
        merged = with_global_defaults(
            {"participants": {"player": "RIGHT", "agent": "LEFT"}}, arena
        )
        self.assertEqual(merged["participants"], {"player": "RIGHT", "agent": "LEFT"})

    def test_invalid_arena_participants_rejected(self):
        with self.assertRaises(ArenaConfigError):
            validate_arena_config({
                **VALID_ARENA, "participants": {"player": "LEFT", "agent": "LEFT"},
            })
        with self.assertRaises(ArenaConfigError):
            validate_arena_config({**VALID_ARENA, "participants": {"player": "UP"}})

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


def test_real_dice_manifest_and_arena_config_compose():
    """Regression: the slimmed dice manifest resolves every slot via the arena.

    The dice manifest deliberately declares no providers/voice/speed anymore;
    if this breaks, either the manifest regained slots or the packaged
    backend/config.json lost them.
    """
    from core.games import GAMES_ROOT, load_games
    from core.arena_config import load_arena_config as load_arena_path

    games = load_games(GAMES_ROOT)
    manifest = games.get("dice")
    # load_games always injects a (possibly empty) providers dict; the slimmed
    # manifest must declare no slots of its own.
    assert not manifest.get("providers")
    assert "voice" not in manifest
    assert "participants" not in manifest
    arena = load_arena_path(ROOT / "backend" / "config.json")
    merged = with_global_defaults(manifest, arena)
    for slot in ("tts_local", "tts_remote", "asr", "vision_adjudicator"):
        assert merged["providers"][slot], f"slot {slot} must resolve via the arena"
    assert merged["participants"] == {"player": "LEFT", "agent": "RIGHT"}
    assert "voice" in merged
    assert "speed" in merged
    enabled = [m for m in games.all() if m.get("enabled", False)]
    assert resolve_local_tts_pin(arena, enabled) == merged["providers"]["tts_local"]


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


def test_participants_inherit_from_arena_end_to_end(tmp_path, monkeypatch):
    """A game manifest without participants loads and plays via the arena mapping."""
    games_root = tmp_path / "games"
    (games_root / "dice").mkdir(parents=True)
    (games_root / "dice" / "manifest.json").write_text(json.dumps({
        "id": "dice", "name": "Dice", "enabled": True,
        "state_machine": MACHINE,
    }), encoding="utf-8")
    arena_path = tmp_path / "config.json"
    arena_path.write_text(json.dumps({
        "schema_version": 1,
        "providers": {"tts_local": "tts_dummy"},
        "participants": {"player": "LEFT", "agent": "RIGHT"},
    }), encoding="utf-8")

    registry = ComponentRegistry()
    registry.register(DummyTts(), {
        "id": "tts_dummy", "type": "tts", "entry": "provider.py:DummyTts",
    })
    monkeypatch.setattr(server, "COMPONENTS", registry)
    monkeypatch.setattr(server, "_GAMES_ROOT", games_root)
    monkeypatch.setattr(server, "GAMES", server.load_games(games_root))
    monkeypatch.setattr(server, "_GAMES_MTIMES", server._manifest_mtimes())
    monkeypatch.setattr(server, "rounds", {})
    monkeypatch.setattr(server, "ARENA_CONFIG_PATH", arena_path)
    monkeypatch.setattr(server, "_ARENA_CONFIG", {})
    monkeypatch.setattr(server, "_ARENA_MTIME", None)
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    try:
        # The public projection must carry the arena-inherited mapping.
        connection = HTTPConnection("127.0.0.1", port, timeout=3)
        connection.request("GET", "/api/games")
        games = json.loads(connection.getresponse().read())["games"]
        connection.close()
        dice = next(g for g in games if g["id"] == "dice")
        assert dice["participants"] == {"player": "LEFT", "agent": "RIGHT"}

        # And a round on that game must start (the merged manifest carries it).
        connection = HTTPConnection("127.0.0.1", port, timeout=3)
        connection.request("POST", "/api/game/rounds", body=b'{"game":"dice"}',
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        snapshot = json.loads(response.read())
        connection.close()
        assert response.status == 201
        assert snapshot["state"] == "rules"
    finally:
        httpd.shutdown()


def test_round_without_any_participants_is_rejected(tmp_path, monkeypatch):
    """No game mapping and no arena mapping = clear error, not a mid-round crash."""
    games_root = tmp_path / "games"
    (games_root / "dice").mkdir(parents=True)
    (games_root / "dice" / "manifest.json").write_text(json.dumps({
        "id": "dice", "name": "Dice", "enabled": True,
        "state_machine": MACHINE,
    }), encoding="utf-8")
    arena_path = tmp_path / "config.json"
    arena_path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    monkeypatch.setattr(server, "_GAMES_ROOT", games_root)
    monkeypatch.setattr(server, "GAMES", server.load_games(games_root))
    monkeypatch.setattr(server, "_GAMES_MTIMES", server._manifest_mtimes())
    monkeypatch.setattr(server, "rounds", {})
    monkeypatch.setattr(server, "ARENA_CONFIG_PATH", arena_path)
    monkeypatch.setattr(server, "_ARENA_CONFIG", {})
    monkeypatch.setattr(server, "_ARENA_MTIME", None)

    with pytest.raises(Exception) as excinfo:
        server.create_round("dice")
    assert "participants" in str(excinfo.value)


def test_manifest_hot_reload_does_not_deadlock_on_drift_check(tmp_path, monkeypatch):
    """Regression 2026-09-03: the local-TTS drift check re-entered get_games()
    while _GAMES_LOCK was held (a non-reentrant lock) — the first manifest
    hot reload after server start froze the whole process; every request
    needing get_games() piled up forever.  The check must run outside the
    lock; this test fails (thread never returns) if it ever regresses.
    """
    import os

    games_root = tmp_path / "games"
    (games_root / "dice").mkdir(parents=True)
    manifest_path = games_root / "dice" / "manifest.json"
    manifest_path.write_text(json.dumps({
        "id": "dice", "name": "Dice", "enabled": True,
        "state_machine": MACHINE,
    }), encoding="utf-8")
    arena_path = tmp_path / "config.json"
    arena_path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    monkeypatch.setattr(server, "_GAMES_ROOT", games_root)
    monkeypatch.setattr(server, "GAMES", server.load_games(games_root))
    monkeypatch.setattr(server, "_GAMES_MTIMES", server._manifest_mtimes())
    monkeypatch.setattr(server, "ARENA_CONFIG_PATH", arena_path)
    monkeypatch.setattr(server, "_ARENA_CONFIG", {})
    monkeypatch.setattr(server, "_ARENA_MTIME", None)

    # The hot-reload trigger: manifest mtime changed since the last read.
    os.utime(manifest_path, (time.time() + 5, time.time() + 5))

    outcome: dict = {}
    thread = threading.Thread(target=lambda: outcome.update(
        games=server.get_games()), daemon=True)
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive(), "get_games() deadlocked on hot reload"
    assert outcome["games"].get("dice") is not None


# ---- SSE presence: a running round needs a live browser (B+A plan) ----

def _presence_round_manifest():
    return {
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
                "ready": {"on_intent": {"back": {"exit": True}}},
            },
        },
    }


class SsePresenceTests(unittest.TestCase):
    def setUp(self):
        server._sse_connections.clear()
        server._sse_presence.clear()
        server.rounds.clear()

    def _round(self):
        round_ = GameRound(
            game_id="dice", manifest=_presence_round_manifest(), log=lambda _l: None,
        )
        round_.start()
        server.rounds[round_.id] = round_  # 生产路径都经 create_round 注册
        return round_

    def tearDown(self):
        server._sse_connections.clear()
        server._sse_presence.clear()
        server.rounds.clear()

    def test_fresh_round_without_consumer_enters_grace(self):
        """create_round 登记宽限：无人连接 SSE 的回合（curl 场景）会被看门收回。"""
        round_ = self._round()
        server._sse_stream_closed(round_.id)  # create_round 的初始登记
        self.assertIn(round_.id, server._sse_presence)
        self.assertEqual(server._sse_connections.get(round_.id), None)

    def test_open_stream_clears_grace_and_counts(self):
        round_ = self._round()
        server._sse_stream_closed(round_.id)
        server._sse_stream_opened(round_.id)
        self.assertNotIn(round_.id, server._sse_presence)  # 重连撤销宽限
        self.assertEqual(server._sse_connections[round_.id], 1)

    def test_grace_expires_cancels_round_and_mic_follows(self):
        round_ = self._round()
        server._sse_stream_closed(round_.id)
        # 把宽限线拨到过去（watchdog 线程之外直接驱动看门函数）
        server._sse_presence[round_.id] = time.monotonic() - 1.0
        server._sse_cancel_stale_rounds()
        self.assertEqual(round_.status, "cancelled")
        self.assertNotIn(round_.id, server._sse_presence)
        # 取消即终态 → asr 桥接看门会停麦（桥接由 create_round 挂，这里验证终态语义）

    def test_grace_not_expired_leaves_round_alone(self):
        round_ = self._round()
        server._sse_stream_closed(round_.id)
        server._sse_cancel_stale_rounds()
        self.assertEqual(round_.status, "running")

    def test_second_consumer_keeps_round_alive_after_first_leaves(self):
        round_ = self._round()
        server._sse_stream_opened(round_.id)
        server._sse_stream_opened(round_.id)
        server._sse_stream_closed(round_.id)  # 第一个离开
        self.assertEqual(server._sse_connections[round_.id], 1)
        self.assertNotIn(round_.id, server._sse_presence)  # 仍有消费者，不进入宽限
        # 最后一个也离开才开始计时
        server._sse_stream_closed(round_.id)
        self.assertIn(round_.id, server._sse_presence)

    def test_terminal_rounds_are_pruned_from_presence(self):
        round_ = self._round()
        server._sse_stream_closed(round_.id)
        round_.cancel()
        server._sse_cancel_stale_rounds()
        self.assertNotIn(round_.id, server._sse_presence)


def test_standby_endpoints_and_projection(tmp_path, monkeypatch):
    """待机端点：开启/停止语音监听，唤醒事件进轮询总线，/api/games 带 standby 节。"""
    games_root = tmp_path / "games"
    (games_root / "dice").mkdir(parents=True)
    (games_root / "dice" / "manifest.json").write_text(json.dumps({
        "id": "dice", "name": "Dice", "enabled": True,
        "participants": {"player": "LEFT", "agent": "RIGHT"},
        "state_machine": MACHINE,
    }), encoding="utf-8")
    arena_path = tmp_path / "config.json"
    arena_path.write_text(json.dumps({
        "schema_version": 1,
        "providers": {"tts_local": "tts_dummy", "asr": "asr_dummy"},
        "standby": {"boot_standby": True, "wake_phrases": ["醒醒"]},
    }), encoding="utf-8")

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
    monkeypatch.setattr(
        server, "ASR_BRIDGE",
        __import__("core.asr_bridge", fromlist=["AsrIntentBridge"]).AsrIntentBridge(
            components=registry, log=lambda _l: None,
        ),
    )
    monkeypatch.setattr(server, "_STANDBY_EVENTS", [])
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]

    def post(path, body):
        connection = HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request("POST", path, body=json.dumps(body).encode("utf-8"),
                            headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        data = json.loads(response.read())
        connection.close()
        return response.status, data

    def get(path):
        connection = HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request("GET", path)
        response = connection.getresponse()
        data = json.loads(response.read())
        connection.close()
        return response.status, data

    try:
        # The projection carries the browser-safe standby settings.
        _, games_payload = get("/api/games")
        assert games_payload["standby"]["boot_standby"] is True
        assert games_payload["standby"]["wake_phrases"] == ["醒醒"]

        # A fresh listening session is a clean event epoch: pre-existing
        # events (an earlier standby period's room noise) must not replay.
        monkeypatch.setattr(server, "_STANDBY_EVENT_SEQUENCE", 41)
        with server._STANDBY_EVENTS_LOCK:
            server._STANDBY_EVENTS.append({
                "event": "asr", "status": "wake", "text": "旧唤醒",
                "timestamp_ms": 1, "sequence": 41,
            })
        status, payload = post("/api/asr/standby", {"listen": True})
        assert status == 200 and payload["listening"] is True
        assert payload["cursor"] == 41  # the new epoch starts after history
        assert len(provider.sessions) == 1
        _, events_payload = get("/api/asr/standby/events")
        assert events_payload["events"] == []  # history was cleared on listen

        # A wake word lands on the event bus with a sequence after the cursor.
        provider.sessions[0]["on_sentence"]("醒醒啊")
        _, events_payload = get("/api/asr/standby/events")
        wakes = [e for e in events_payload["events"] if e.get("status") == "wake"]
        assert wakes and wakes[-1]["text"] == "醒醒啊"
        assert wakes[-1]["sequence"] > payload["cursor"]

        # Listen off → no session left.
        status, payload = post("/api/asr/standby", {"listen": False})
        assert status == 200 and payload["listening"] is False
        # The bridge stop path runs through the provider; the next listen
        # would create a fresh session.
        status, payload = post("/api/asr/standby", {"listen": True})
        assert payload["listening"] is True
        assert len(provider.sessions) == 2
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    unittest.main()
