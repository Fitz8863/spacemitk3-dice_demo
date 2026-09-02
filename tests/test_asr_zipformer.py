from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.asr import AsrProvider, AsrSessionError  # noqa: E402
from core.components import Component, ComponentRegistry, _validate_manifest  # noqa: E402
import components.asr_zipformer.provider as asr_provider_module  # noqa: E402
from components.asr_zipformer.provider import (  # noqa: E402
    AsrConfigError,
    ZipformerAsrProvider,
    _AsrSession,
    load_config,
)


VALID_CONFIG = {
    "schema_version": 1,
    "runtime": {
        "binary": "asr/zipformer-streaming/build/stream_asr",
        "working_dir": "asr/zipformer-streaming",
        "model_dir": "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
        "capture": {"device": "default", "sample_rate": 16000, "channels": 1, "format": "S16_LE"},
        "vad": {"enabled": True, "rms": 400, "pause_ms": 600, "max_ms": 8000},
        "cpu_affinity": "",
        "start_timeout_seconds": 15,
        "terminate_grace_seconds": 5,
    },
}


def _write_config(directory: Path, payload: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return directory


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _load(self, payload: dict):
        package_dir = _write_config(self.root / "pkg", payload)
        return load_config(package_dir, project_root=self.root)

    def test_valid_config_passes(self):
        payload = self._load(VALID_CONFIG)
        self.assertEqual(payload["schema_version"], 1)

    def test_rejects_wrong_schema_version(self):
        payload = dict(VALID_CONFIG, schema_version=2)
        with self.assertRaises(AsrConfigError):
            self._load(payload)

    def test_rejects_absolute_binary(self):
        payload = json.loads(json.dumps(VALID_CONFIG))
        payload["runtime"]["binary"] = "/usr/bin/stream_asr"
        with self.assertRaises(AsrConfigError):
            self._load(payload)

    def test_rejects_traversal_working_dir(self):
        payload = json.loads(json.dumps(VALID_CONFIG))
        payload["runtime"]["working_dir"] = "asr/../../etc"
        with self.assertRaises(AsrConfigError):
            self._load(payload)

    def test_rejects_wrong_sample_rate(self):
        payload = json.loads(json.dumps(VALID_CONFIG))
        payload["runtime"]["capture"]["sample_rate"] = 8000
        with self.assertRaises(AsrConfigError):
            self._load(payload)

    def test_rejects_bad_cpu_affinity(self):
        payload = json.loads(json.dumps(VALID_CONFIG))
        payload["runtime"]["cpu_affinity"] = "0..3"
        with self.assertRaises(AsrConfigError):
            self._load(payload)

    def test_rejects_absolute_model_dir(self):
        payload = json.loads(json.dumps(VALID_CONFIG))
        payload["runtime"]["model_dir"] = "/models/zipformer"
        with self.assertRaises(AsrConfigError):
            self._load(payload)

    def test_rejects_non_positive_vad(self):
        payload = json.loads(json.dumps(VALID_CONFIG))
        payload["runtime"]["vad"]["rms"] = 0
        with self.assertRaises(AsrConfigError):
            self._load(payload)


class FakeProcess:
    """Minimal Popen double: pipes are BytesIO, terminate ends the stream."""

    def __init__(self, stdout_data: bytes = b"", stderr_data: bytes = b"") -> None:
        self.stdout = io.BytesIO(stdout_data)
        self.stderr = io.BytesIO(stderr_data)
        self._rc: int | None = None
        self.terminate_calls = 0
        self._on_terminate = None

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self._on_terminate is not None:
            self._on_terminate()
        if self._rc is None:
            self._rc = 0

    def kill(self) -> None:
        self._rc = -9

    def finish(self, rc: int = 0) -> None:
        if self._rc is None:
            self._rc = rc

    def wait(self, timeout=None):
        if self._rc is None:
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
        return self._rc

    def poll(self):
        return self._rc


class WiredFakePipeline:
    """Fake popen whose capture termination finishes the asr process."""

    def __init__(self, asr_stdout: bytes, asr_stderr: bytes, calls: list) -> None:
        self._asr_stdout = asr_stdout
        self._asr_stderr = asr_stderr
        self._calls = calls
        self._capture: FakeProcess | None = None

    def __call__(self, argv, **kwargs):
        self._calls.append((list(argv), kwargs))
        if kwargs.get("stdin") is None:
            capture = FakeProcess()
            self._capture = capture
            return capture
        asr = FakeProcess(self._asr_stdout, self._asr_stderr)
        if self._capture is not None:
            self._capture._on_terminate = asr.finish
        return asr


ASR_JSONL_OUTPUT = (
    b'{"type":"partial","text":"\xe7\xa1\xae"}\n'
    b'{"type":"sentence","text":"\xe7\xa1\xae\xe8\xae\xa4"}\n'
    b"not-a-json-line\n"
    b'{"type":"final","text":"\xe5\xbc\x80\xe5\xa7\x8b"}\n'
    b'{"type":"stats","audio_seconds":1.0,"infer_seconds":0.3,"rtf":0.3,"chunks":3,"tokens":2}\n'
)
ASR_STDERR_OUTPUT = "[stream_asr] encoder: fake.onnx (SpaceMIT EP)\n".encode("utf-8")


class SessionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.working_dir = Path(self._tmp.name)
        self.sentences: list[str] = []
        self.logs: list[str] = []

    def _make_session(self, calls):
        popen = WiredFakePipeline(ASR_JSONL_OUTPUT, ASR_STDERR_OUTPUT, calls)
        return _AsrSession(
            capture_argv=["arecord", "-D", "default"],
            asr_argv=["stream_asr", "--pcm", "--jsonl"],
            working_dir=self.working_dir,
            grace_seconds=2,
            on_sentence=self.sentences.append,
            on_log=self.logs.append,
            popen=popen,
        )

    def test_session_streams_sentences_and_ignores_noise(self):
        calls: list = []
        session = self._make_session(calls)
        session.start()
        self.assertTrue(session.wait_ready(2))
        session.stop()
        self.assertEqual(self.sentences, ["确认", "开始"])
        joined_logs = "\n".join(self.logs)
        self.assertIn("non-JSON", joined_logs)
        self.assertIn("stats rtf=0.3", joined_logs)
        self.assertIn("[stream_asr] encoder", joined_logs)
        self.assertFalse(session.running)
        # capture is stopped first so stream_asr can flush and exit on its own
        capture_argv = next(argv for argv, kwargs in calls if kwargs.get("stdin") is None)
        self.assertEqual(capture_argv[:2], ["arecord", "-D"])

    def test_stop_is_idempotent(self):
        calls: list = []
        session = self._make_session(calls)
        session.start()
        session.stop()
        session.stop()
        self.assertFalse(session.running)


class FakeSession:
    instances: list["FakeSession"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self._running = True
        self.stopped = False
        FakeSession.instances.append(self)

    def start(self) -> None:
        pass

    @property
    def running(self) -> bool:
        return self._running

    def wait_ready(self, timeout_seconds: float) -> bool:
        return True

    def stop(self) -> None:
        self.stopped = True
        self._running = False


class DeadFakeSession(FakeSession):
    """A session whose ASR process died before loading the model."""

    @property
    def running(self) -> bool:
        return False

    def wait_ready(self, timeout_seconds: float) -> bool:
        return False


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _make_board_tree(self) -> None:
        binary = self.root / "asr/zipformer-streaming/build/stream_asr"
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(b"#!/bin/sh\n")
        binary.chmod(0o755)
        model = self.root / (
            "asr/zipformer-streaming/"
            "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
        )
        model.mkdir(parents=True, exist_ok=True)

    def test_health_reports_missing_runtime(self):
        provider = ZipformerAsrProvider(project_root=self.root)
        health = provider.health()
        self.assertFalse(health["ok"])
        self.assertTrue(health["problems"])

    def test_health_ok_with_binary_and_model(self):
        self._make_board_tree()
        provider = ZipformerAsrProvider(project_root=self.root)
        health = provider.health()
        self.assertTrue(health["ok"])
        self.assertNotIn("problems", health)

    @patch.object(asr_provider_module, "_AsrSession", FakeSession)
    def test_start_session_builds_argv_and_enforces_single_session(self):
        FakeSession.instances = []
        provider = ZipformerAsrProvider(project_root=self.root)
        session = provider.start_session(lambda _text: None)
        self.assertIsInstance(session, FakeSession)
        asr_argv = session.kwargs["asr_argv"]
        self.assertEqual(asr_argv[0], str(provider._binary))
        self.assertIn("--pcm", asr_argv)
        self.assertIn("--jsonl", asr_argv)
        self.assertIn("--model-dir", asr_argv)
        capture_argv = session.kwargs["capture_argv"]
        self.assertEqual(capture_argv[0], "arecord")
        self.assertIn("default", capture_argv)
        with self.assertRaises(AsrSessionError):
            provider.start_session(lambda _text: None)

    @patch.object(asr_provider_module, "_AsrSession", FakeSession)
    def test_stop_session_allows_restart(self):
        FakeSession.instances = []
        provider = ZipformerAsrProvider(project_root=self.root)
        session = provider.start_session(lambda _text: None)
        provider.stop_session(session)
        self.assertTrue(session.stopped)
        provider.start_session(lambda _text: None)  # no exception

    @patch.object(asr_provider_module, "_AsrSession", FakeSession)
    def test_cpu_affinity_wraps_argv_with_taskset(self):
        FakeSession.instances = []
        provider = ZipformerAsrProvider(project_root=self.root)
        provider._config["runtime"]["cpu_affinity"] = "0,3"
        session = provider.start_session(lambda _text: None)
        self.assertEqual(session.kwargs["asr_argv"][:3], ["taskset", "-c", "0,3"])

    @patch.object(asr_provider_module, "_AsrSession", FakeSession)
    def test_vad_disabled_flag(self):
        FakeSession.instances = []
        provider = ZipformerAsrProvider(project_root=self.root)
        provider._config["runtime"]["vad"]["enabled"] = False
        session = provider.start_session(lambda _text: None)
        self.assertIn("--no-vad", session.kwargs["asr_argv"])
        self.assertNotIn("--vad-rms", session.kwargs["asr_argv"])

    @patch.object(asr_provider_module, "_AsrSession", FakeSession)
    def test_stop_ignores_foreign_handles(self):
        FakeSession.instances = []
        provider = ZipformerAsrProvider(project_root=self.root)
        provider.start_session(lambda _text: None)
        provider.stop_session("not-a-session")  # no error, session untouched
        self.assertTrue(FakeSession.instances[0].running)

    @patch.object(asr_provider_module, "_AsrSession", DeadFakeSession)
    def test_start_session_raises_when_process_dies_during_startup(self):
        provider = ZipformerAsrProvider(project_root=self.root)
        with self.assertRaises(AsrSessionError):
            provider.start_session(lambda _text: None)


class RegistryContractTests(unittest.TestCase):
    def test_asr_type_requires_asr_provider_interface(self):
        class NotAnAsr(Component):
            id = "asr_fake"
            type = "asr"

        registry = ComponentRegistry()
        with self.assertRaises(ValueError):
            registry.register(NotAnAsr(), {"id": "asr_fake", "type": "asr"})

    def test_real_provider_registers_under_asr_type(self):
        registry = ComponentRegistry()
        provider = ZipformerAsrProvider()
        registry.register(provider, {
            "id": "asr_zipformer",
            "type": "asr",
            "entry": "provider.py:ZipformerAsrProvider",
        })
        self.assertEqual(registry.provider_ids("asr"), ["asr_zipformer"])
        self.assertEqual(registry.require("asr_zipformer", expected_type="asr"), provider)

    def test_packaged_manifest_is_valid(self):
        manifest_path = ROOT / "backend/components/asr_zipformer/manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_manifest(manifest_path, manifest)


if __name__ == "__main__":
    unittest.main()
