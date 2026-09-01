from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.components import ComponentRegistry, _validate_manifest  # noqa: E402
from core.tts import TtsProvider  # noqa: E402
from core.tts_config import TtsConfigError, validate_tts_component_config  # noqa: E402
from core.games import normalize_speech_entry  # noqa: E402
from core.tts_dispatch import TtsDispatcher  # noqa: E402
from components.tts_qwen3.settings import load_settings as load_qwen_settings  # noqa: E402
from components.tts_moss_nano.settings import load_settings as load_moss_settings  # noqa: E402
from componentctl import _command_for  # noqa: E402


WAV = b"RIFF" + b"\0" * 4 + b"WAVE" + b"\0" * 40


class DummyTts(TtsProvider):
    id = "tts_dummy"

    def health(self):
        return {"ok": True, "engine": "dummy"}

    def synthesize(self, payload):
        self.validate(payload)
        return WAV, {"Content-Type": "audio/wav"}


class DummyCloudTts(TtsProvider):
    id = "tts_dummy_cloud"

    def synthesize(self, payload):
        self.validate(payload)
        return WAV, {"Content-Type": "audio/wav", "X-TTS-Runtime": "cloud"}


class MissingSynthesize(TtsProvider):
    id = "tts_missing"


class DispatcherTests(unittest.TestCase):
    def setUp(self):
        self.registry = ComponentRegistry()
        self.provider = DummyTts()
        self.registry.register(self.provider, {
            "id": "tts_dummy",
            "type": "tts",
            "entry": "provider.py:DummyTts",
        })
        self.games = {"dice": {"enabled": True, "providers": {"tts": "tts_dummy"}}}

    def test_dispatcher_uses_backend_selection_not_request_override(self):
        dispatcher = TtsDispatcher(self.registry, self.games)
        selected = dispatcher.provider({"game": "dice", "provider": "some_other"})
        self.assertIs(selected, self.provider)

    def test_dispatcher_delegates_synthesis_and_stream(self):
        dispatcher = TtsDispatcher(self.registry, self.games)
        audio, headers = dispatcher.synthesize({"game": "dice", "text": "hello"})
        self.assertEqual(audio, WAV)
        self.assertEqual(headers["Content-Type"], "audio/wav")
        frames = []
        dispatcher.stream({"game": "dice", "text": "hello"}, frames.append)
        self.assertEqual(frames, [WAV])

    def test_cloud_provider_uses_the_same_dispatch_contract_without_lifecycle(self):
        registry = ComponentRegistry()
        provider = DummyCloudTts()
        registry.register(provider, {
            "id": provider.id,
            "type": "tts",
            "entry": "provider.py:DummyCloudTts",
            "config": "config.json",
        })
        games = {"dice": {"enabled": True, "providers": {"tts": provider.id}}}
        dispatcher = TtsDispatcher(registry, games)
        audio, headers = dispatcher.synthesize({"game": "dice", "text": "cloud"})
        self.assertEqual(audio, WAV)
        self.assertEqual(headers["X-TTS-Runtime"], "cloud")
        self.assertIsNone(_command_for({"id": provider.id, "lifecycle": {}}, "start"))


class TtsContractTests(unittest.TestCase):
    def test_speech_manifest_entries_normalize_legacy_text_and_audio_modes(self):
        self.assertEqual(
            normalize_speech_entry("欢迎"),
            {"mode": "tts", "text": "欢迎"},
        )
        self.assertEqual(
            normalize_speech_entry({"mode": "audio", "audio": "audio/rules.wav"}),
            {"mode": "audio", "audio": "audio/rules.wav"},
        )
        with self.assertRaises(ValueError):
            normalize_speech_entry({"mode": "audio", "audio": "../secret.wav"})
        with self.assertRaises(ValueError):
            normalize_speech_entry({"mode": "unknown", "text": "bad"})

    def test_tts_provider_is_a_real_abstract_base_class(self):
        with self.assertRaises(TypeError):
            MissingSynthesize()

    def test_component_config_validates_common_runtime_fields(self):
        validate_tts_component_config({
            "runtime": {"kind": "cloud", "base_url": "https://tts.example.com"},
            "voice": {"default": "default"},
        })
        with self.assertRaises(TtsConfigError):
            validate_tts_component_config({"runtime": {"kind": "unknown"}})
        with self.assertRaises(TtsConfigError):
            validate_tts_component_config({"runtime": {"kind": "cloud", "base_url": "not a url"}})

    def test_local_tts_runtime_must_stay_on_loopback(self):
        with self.assertRaisesRegex(TtsConfigError, "loopback"):
            validate_tts_component_config({
                "runtime": {"kind": "local", "base_url": "http://192.168.1.5:18082"},
            })
        with self.assertRaisesRegex(TtsConfigError, "loopback"):
            validate_tts_component_config({
                "runtime": {"kind": "local", "host": "0.0.0.0", "port": 18082},
            })
        # Cloud and externally managed providers may use remote origins.
        validate_tts_component_config({
            "runtime": {"kind": "cloud", "base_url": "https://tts.example.com"},
        })

    def test_tts_manifest_requires_component_config(self):
        with self.assertRaisesRegex(ValueError, "component-local config"):
            _validate_manifest(ROOT / "backend" / "components" / "tts_dummy" / "manifest.json", {
                "id": "tts_dummy",
                "type": "tts",
                "entry": "provider.py:DummyTts",
            })

    def test_settings_are_the_effective_config_source(self):
        qwen = {
            "runtime": {
                "kind": "local",
                "root": "/tmp/qwen-root",
                "model_dir": "model",
                "host": "127.0.0.1",
                "port": 19080,
                "base_url": "http://127.0.0.1:19080",
            },
            "voice": {"default": "narrator", "speaker_file": "voice.spk.bin"},
            "generation": {"timeout_seconds": 17, "speed": 1.25, "chunk_chars": 31},
        }
        moss = {
            "runtime": {
                "kind": "local",
                "root": "/tmp/moss-root",
                "model_dir": "models/custom",
                "host": "127.0.0.1",
                "port": 19082,
                "base_url": "http://127.0.0.1:19082",
            },
            "voice": {"mode": "builtin", "name": "TestVoice"},
            "generation": {"max_new_frames": 80, "voice_clone_max_text_tokens": 20, "first_chunk_text_tokens": 8, "seed": 9},
            "startup": {"warmup_text": "warm", "start_timeout_seconds": 4},
            "limits": {"request_timeout_seconds": 12},
            "execution_provider": {"intra_thread_num": 2, "inter_thread_num": 1, "intra_thread_affinity": "1;2", "disable_op_type_filter": "Add"},
        }
        qwen_settings = load_qwen_settings(qwen)
        moss_settings = load_moss_settings(moss)
        self.assertEqual(qwen_settings.default_voice, "narrator")
        self.assertEqual(qwen_settings.model_dir, Path("/tmp/qwen-root/model"))
        self.assertEqual(qwen_settings.url, "http://127.0.0.1:19080")
        self.assertEqual(moss_settings.voice, "TestVoice")
        self.assertEqual(moss_settings.model_dir, Path("/tmp/moss-root/models/custom"))
        self.assertEqual(moss_settings.max_new_frames, 80)

    def test_checked_in_runtime_paths_and_assets_match_component_configs(self):
        qwen = load_qwen_settings(json.loads((ROOT / "backend/components/tts_qwen3/config.json").read_text()))
        moss = load_moss_settings(json.loads((ROOT / "backend/components/tts_moss_nano/config.json").read_text()))
        self.assertTrue((qwen.model_dir / qwen.speaker_file).is_file())
        self.assertTrue((moss.model_dir / "browser_poc_manifest.json").is_file())
        if moss.voice_mode == "clone":
            self.assertIsNotNone(moss.reference_audio)
            self.assertTrue(moss.reference_audio.is_file(), moss.reference_audio)
        else:
            self.assertIsNone(moss.reference_audio)

    def test_runtime_configs_are_valid_and_cli_sources_are_removed(self):
        for component_id in ("tts_qwen3", "tts_moss_nano"):
            config_path = ROOT / "backend" / "components" / component_id / "config.json"
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            validate_tts_component_config(payload)

        removed = (
            ROOT / "tts" / "qwen3-tts" / "qwen3_tts_interactive.py",
            ROOT / "tts" / "qwen3-tts" / "run_interactive.sh",
            ROOT / "tts" / "moss-tts-nano" / "run_demo.sh",
            ROOT / "tts" / "moss-tts-nano" / "run_interactive.sh",
            ROOT / "tts" / "moss-tts-nano" / "run_voice_clone.sh",
            ROOT / "tts" / "moss-tts-nano" / "src" / "moss_spacemit_demo.py",
            ROOT / "tts" / "moss-tts-nano" / "src" / "moss_spacemit_interactive.py",
        )
        for path in removed:
            self.assertFalse(path.exists(), path)


if __name__ == "__main__":
    unittest.main()
