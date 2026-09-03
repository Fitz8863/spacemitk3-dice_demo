from __future__ import annotations

import base64
import json
import os
import stat
import sys
import tempfile
import textwrap
import unittest
import wave
from io import BytesIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.errors import TtsServiceError, TtsValidationError  # noqa: E402
from core.tts import TtsProvider  # noqa: E402
import components.tts_matcha.provider as matcha_module  # noqa: E402
from components.tts_matcha.provider import (  # noqa: E402
    MatchaConfigError,
    TtsMatcha,
    load_config,
    merge_wav_frames,
)


VALID_CONFIG = {
    "schema_version": 1,
    "runtime": {
        "kind": "local",
        "root": "tts/matcha-tts",
        "binary": "build-cpp/matcha_tts_service",
        "model_dir": "matcha-model",
        "sherpa_lib_dir": "runtime/sherpa_onnx/lib",
        "ep_threads": 2,
        "ep_affinity": "8;9",
        "enable_affinity": True,
        "start_timeout_seconds": 60,
        "terminate_grace_seconds": 5,
        "request_timeout_seconds": 120,
    },
    "startup": {"warmup_text": "你好。"},
    "generation": {"chunk_target": 40, "chunk_max": 90},
    "voice": {"speaker_id": 0},
}

# A tiny but valid mono 16 kHz 16-bit WAV.
def _make_wav(samples: int = 1600, rate: int = 16000) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(b"\x00\x01" * samples)
    return buffer.getvalue()


# Stand-in for the C++ service: same stdin/stdout protocol, WAV frames with a
# fixed payload, request lines recorded for assertions. Modes emulate the
# failure paths of the real binary.
FAKE_SERVICE = textwrap.dedent(
    """
    #!/usr/bin/env python3
    import base64, json, os, sys, time
    WAV = base64.b64decode("{wav_b64}")
    log = open(os.environ["FAKE_SERVICE_LOG"], "a", encoding="utf-8")

    def emit(event):
        print(json.dumps(event, ensure_ascii=False), flush=True)

    mode = os.environ.get("FAKE_SERVICE_MODE", "ok")
    if mode == "no-ready":
        time.sleep(30)
        sys.exit(0)
    if mode == "startup-error":
        emit({{"event": "error", "id": "", "message": "warmup synthesis failed"}})
        sys.exit(1)

    emit({{"event": "ready", "sample_rate": 16000, "voice": "0",
           "ep_threads": 2, "ep_affinity": "8;9"}})
    for line in sys.stdin:
        line = line.rstrip("\\n")
        if not line or line.startswith("#"):
            continue
        log.write(line + "\\n")
        log.flush()
        request_id, speed, text = line.split("\\t", 2)
        if mode == "request-error" and text.startswith("坏"):
            emit({{"event": "error", "id": request_id, "message": "sentence 1 synthesis failed"}})
            continue
        for seq, sentence in enumerate(text.split("|"), start=1):
            if mode == "die-mid-request" and sentence.startswith("第二"):
                sys.exit(3)
            emit({{"event": "sentence", "id": request_id, "seq": seq, "text": sentence}})
            emit({{"event": "audio", "id": request_id, "seq": seq,
                   "sample_rate": 16000, "duration_seconds": 0.1,
                   "wav_b64": base64.b64encode(WAV).decode("ascii")}})
        emit({{"event": "done", "id": request_id, "sentences": text.count("|") + 1,
               "audio_seconds": 0.2, "elapsed_seconds": 0.01}})
    """
)


def _write_fake_service(root: Path) -> Path:
    # lstrip: the template's opening newline would push the shebang off line 1.
    payload = FAKE_SERVICE.format(
        wav_b64=base64.b64encode(_make_wav()).decode("ascii")
    ).lstrip("\n")
    path = root / "fake_matcha_service.py"
    path.write_text(payload, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _make_provider(root: Path, *, config: dict | None = None, mode: str = "ok") -> TtsMatcha:
    package = root / "backend" / "components" / "tts_matcha"
    package.mkdir(parents=True, exist_ok=True)
    payload = json.loads(json.dumps(config or VALID_CONFIG))
    payload["runtime"]["root"] = "."
    payload["runtime"]["binary"] = "fake_matcha_service.py"
    payload["runtime"]["model_dir"] = "."
    payload["runtime"]["sherpa_lib_dir"] = "."
    payload["runtime"]["start_timeout_seconds"] = 5
    payload["runtime"]["request_timeout_seconds"] = 5
    (package / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    _write_fake_service(root)
    # Stand-in for the board-only sherpa runtime so health sees a complete
    # asset layout.
    (root / "libsherpa-onnx-c-api.so").write_text("", encoding="utf-8")
    log_path = root / "requests.log"
    log_path.write_text("", encoding="utf-8")
    os.environ["FAKE_SERVICE_LOG"] = str(log_path)
    os.environ["FAKE_SERVICE_MODE"] = mode
    return TtsMatcha(
        manifest={"id": "tts_matcha", "type": "tts"},
        project_root=root,
        package_dir=package,
    )


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _load(self, payload: dict):
        package = self.root / "pkg"
        package.mkdir(parents=True, exist_ok=True)
        (package / "config.json").write_text(json.dumps(payload), encoding="utf-8")
        return load_config(package, project_root=self.root)

    def test_valid_config_passes(self):
        self.assertEqual(self._load(VALID_CONFIG)["schema_version"], 1)

    def test_rejects_wrong_schema_version(self):
        payload = json.loads(json.dumps(VALID_CONFIG))
        payload["schema_version"] = 2
        with self.assertRaises(MatchaConfigError):
            self._load(payload)

    def test_rejects_non_local_kind(self):
        payload = json.loads(json.dumps(VALID_CONFIG))
        payload["runtime"]["kind"] = "external"
        with self.assertRaises(MatchaConfigError):
            self._load(payload)

    def test_rejects_absolute_root(self):
        payload = json.loads(json.dumps(VALID_CONFIG))
        payload["runtime"]["root"] = "/tmp/matcha"
        with self.assertRaises(MatchaConfigError):
            self._load(payload)

    def test_rejects_binary_outside_root(self):
        payload = json.loads(json.dumps(VALID_CONFIG))
        payload["runtime"]["binary"] = "../outside/service"
        with self.assertRaises(MatchaConfigError):
            self._load(payload)

    def test_rejects_affinity_count_mismatch(self):
        payload = json.loads(json.dumps(VALID_CONFIG))
        payload["runtime"]["ep_affinity"] = "8;9;10"
        with self.assertRaises(MatchaConfigError):
            self._load(payload)

    def test_rejects_duplicate_affinity_cores(self):
        payload = json.loads(json.dumps(VALID_CONFIG))
        payload["runtime"]["ep_affinity"] = "8;8"
        with self.assertRaises(MatchaConfigError):
            self._load(payload)

    def test_rejects_bad_affinity_token(self):
        payload = json.loads(json.dumps(VALID_CONFIG))
        payload["runtime"]["ep_affinity"] = "8,9"
        with self.assertRaises(MatchaConfigError):
            self._load(payload)

    def test_rejects_chunk_target_above_max(self):
        payload = json.loads(json.dumps(VALID_CONFIG))
        payload["generation"]["chunk_target"] = 200
        with self.assertRaises(MatchaConfigError):
            self._load(payload)

    def test_rejects_nonzero_speaker(self):
        payload = json.loads(json.dumps(VALID_CONFIG))
        payload["voice"]["speaker_id"] = 1
        with self.assertRaises(MatchaConfigError):
            self._load(payload)

    def test_rejects_empty_warmup_text(self):
        payload = json.loads(json.dumps(VALID_CONFIG))
        payload["startup"]["warmup_text"] = "  "
        with self.assertRaises(MatchaConfigError):
            self._load(payload)


class MergeWavTests(unittest.TestCase):
    def test_merges_same_format_frames(self):
        merged = merge_wav_frames([_make_wav(100), _make_wav(200)])
        with wave.open(BytesIO(merged)) as reader:
            self.assertEqual(reader.getnframes(), 300)
            self.assertEqual(reader.getframerate(), 16000)

    def test_rejects_empty_list(self):
        with self.assertRaises(TtsServiceError):
            merge_wav_frames([])

    def test_rejects_garbage_frame(self):
        with self.assertRaises(TtsServiceError):
            merge_wav_frames([b"not a wav"])


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _make(self, **kwargs) -> TtsMatcha:
        provider = _make_provider(self.root, **kwargs)
        self.addCleanup(provider.shutdown)
        return provider

    def test_is_a_tts_provider(self):
        provider = self._make()
        self.assertIsInstance(provider, TtsProvider)
        self.assertEqual(provider.id, "tts_matcha")

    def test_prewarm_then_synthesize(self):
        provider = self._make()
        provider.prewarm()
        audio, headers = provider.synthesize({"text": "你好|世界", "speed": 1.0})
        with wave.open(BytesIO(audio)) as reader:
            self.assertEqual(reader.getnframes(), 3200)
            self.assertEqual(reader.getframerate(), 16000)
        self.assertEqual(headers["X-Dice-TTS-Frames"], "2")

    def test_stream_emits_one_frame_per_sentence(self):
        provider = self._make()
        frames: list[bytes] = []
        provider.stream({"text": "第一句|第二句|第三句", "speed": 1.0}, frames.append)
        self.assertEqual(len(frames), 3)
        for frame in frames:
            self.assertTrue(frame.startswith(b"RIFF"))

    def test_request_line_carries_speed_and_sanitized_text(self):
        provider = self._make()
        provider.synthesize({"text": "你好\t换行\n测试", "speed": 1.5})
        logged = (self.root / "requests.log").read_text(encoding="utf-8").strip()
        request_id, speed, text = logged.split("\t")
        self.assertEqual(speed, "1.5")
        self.assertNotIn("\t", text)
        self.assertNotIn("\n", text)

    def test_voice_zero_and_default_accepted(self):
        provider = self._make()
        provider.synthesize({"text": "你好", "voice": "0"})
        provider.synthesize({"text": "你好", "voice": "default"})
        with self.assertRaises(TtsValidationError):
            provider.synthesize({"text": "你好", "voice": "alice"})

    def test_request_error_becomes_service_error(self):
        provider = self._make(mode="request-error")
        with self.assertRaises(TtsServiceError) as ctx:
            provider.synthesize({"text": "坏句子"})
        self.assertIn("sentence 1 synthesis failed", str(ctx.exception))

    def test_startup_error_refuses_prewarm(self):
        provider = self._make(mode="startup-error")
        with self.assertRaises(TtsServiceError):
            provider.prewarm()

    def test_ready_timeout_refuses_prewarm(self):
        provider = self._make(mode="no-ready")
        with self.assertRaises(TtsServiceError):
            provider.prewarm()

    def test_crash_mid_request_is_reported(self):
        provider = self._make(mode="die-mid-request")
        with self.assertRaises(TtsServiceError):
            provider.stream({"text": "第一句|第二句"}, lambda _frame: None)

    def test_dead_service_restarts_on_next_request(self):
        provider = self._make(mode="die-mid-request")
        with self.assertRaises(TtsServiceError):
            provider.stream({"text": "第二句触发退出"}, lambda _frame: None)
        # The next request spawns a fresh child and succeeds.
        frames: list[bytes] = []
        provider.stream({"text": "再来一次"}, frames.append)
        self.assertEqual(len(frames), 1)

    def test_shutdown_stops_service(self):
        provider = self._make()
        provider.prewarm()
        health = provider.health()
        self.assertTrue(health["ok"])
        self.assertTrue(health["running"])
        provider.shutdown()
        health = provider.health()
        self.assertFalse(health["running"])
        self.assertFalse(health["ok"])

    def test_health_reports_missing_assets_without_spawning(self):
        provider = self._make()
        (self.root / "fake_matcha_service.py").unlink()
        health = provider.health()
        self.assertIn("problems", health)
        self.assertTrue(any("binary missing" in problem for problem in health["problems"]))
        self.assertFalse(health["ok"])
        self.assertFalse(health["running"])

    def test_lazy_first_request_spawns_and_synthesizes(self):
        provider = self._make()
        frames: list[bytes] = []
        provider.stream({"text": "直接开始"}, frames.append)
        self.assertEqual(len(frames), 1)
        self.assertTrue(provider.health()["running"])


if __name__ == "__main__":
    unittest.main()
