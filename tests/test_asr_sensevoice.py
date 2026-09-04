from __future__ import annotations

import collections
import io
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.asr import AsrProvider, AsrSessionError  # noqa: E402
from core.components import Component, ComponentRegistry, _validate_manifest  # noqa: E402
import components.asr_sensevoice.provider as asr_provider_module  # noqa: E402
from components.asr_sensevoice.provider import (  # noqa: E402
    AsrConfigError,
    SensevoiceAsrProvider,
    _AsrEngine,
    load_config,
)


VALID_CONFIG = {
    "schema_version": 1,
    "runtime": {
        "binary": "asr/sensevoice-x100/build/bin/asr_pipe_demo",
        "working_dir": "asr/sensevoice-x100",
        "model_dir": "asr/sensevoice-x100/model",
        "language": "auto",
        "core_arch": "x100",
        "capture": {"device": "default", "sample_rate": 16000, "channels": 1, "format": "S16_LE"},
        "vad": {"enabled": True, "rms": 400, "pause_ms": 600, "max_ms": 8000},
        "cpu_affinity": "",
        "start_timeout_seconds": 30,
        "terminate_grace_seconds": 5,
    },
}


def _write_config(directory: Path, payload: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return directory


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _mutated(**overrides) -> dict:
    payload = json.loads(json.dumps(VALID_CONFIG))
    runtime = payload["runtime"]
    for dotted, value in overrides.items():
        section, sep, field = dotted.partition("__")
        if sep:
            runtime[section][field] = value
        else:
            runtime[dotted] = value
    return payload


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
        with self.assertRaises(AsrConfigError):
            self._load(dict(VALID_CONFIG, schema_version=2))

    def test_rejects_absolute_binary(self):
        with self.assertRaises(AsrConfigError):
            self._load(_mutated(binary="/usr/bin/asr_pipe_demo"))

    def test_rejects_absolute_model_dir(self):
        with self.assertRaises(AsrConfigError):
            self._load(_mutated(model_dir="/models/sensevoice"))

    def test_rejects_traversal_model_dir(self):
        with self.assertRaises(AsrConfigError):
            self._load(_mutated(model_dir="asr/sensevoice-x100/../../models"))

    def test_rejects_wrong_sample_rate(self):
        with self.assertRaises(AsrConfigError):
            self._load(_mutated(capture__sample_rate=8000))

    def test_rejects_bad_cpu_affinity(self):
        with self.assertRaises(AsrConfigError):
            self._load(_mutated(cpu_affinity="0..3"))

    def test_rejects_non_positive_vad(self):
        with self.assertRaises(AsrConfigError):
            self._load(_mutated(vad__rms=0))

    def test_rejects_disabled_vad(self):
        """The pipe demo's non-VAD mode is timed flushing; command listening
        is VAD-segmented only, so a disabled VAD is a config error here."""
        with self.assertRaises(AsrConfigError):
            self._load(_mutated(vad__enabled=False))

    def test_rejects_unknown_language(self):
        with self.assertRaises(AsrConfigError):
            self._load(_mutated(language="fr"))

    def test_rejects_unknown_core_arch(self):
        with self.assertRaises(AsrConfigError):
            self._load(_mutated(core_arch="z80"))

    def test_rejects_non_sensevoice_engine(self):
        with self.assertRaises(AsrConfigError):
            self._load(_mutated(engine="zipformer"))


class GatedStdout:
    """readline-compatible stream whose lines arrive only when pushed.

    Lets tests control exactly when the (fully buffered fake) engine emits a
    sentence or the ready event, so routing swaps and prewarm semantics can
    be asserted without racing the reader thread.  Closing emulates process
    death: the reader sees EOF.
    """

    def __init__(self) -> None:
        self._lines: collections.deque[bytes] = collections.deque()
        self._closed = False
        self._condition = threading.Condition()

    def push(self, line: bytes) -> None:
        with self._condition:
            self._lines.append(line)
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def __iter__(self):
        return self

    def __next__(self) -> bytes:
        line = self.readline()
        if line == b"":
            raise StopIteration
        return line

    def readline(self) -> bytes:
        with self._condition:
            while not self._lines and not self._closed:
                self._condition.wait()
            if self._lines:
                return self._lines.popleft()
            return b""


class FakeProcess:
    """Minimal Popen double: pipes are BytesIO (or a gated stream), death
    closes them so the reader threads see EOF like a real pipe."""

    def __init__(
        self,
        stdout_data: bytes = b"",
        stderr_data: bytes = b"",
        *,
        gated_stdout: bool = False,
    ) -> None:
        self.stdout = GatedStdout() if gated_stdout else io.BytesIO(stdout_data)
        self.stderr = io.BytesIO(stderr_data)
        self._rc: int | None = None
        self.terminate_calls = 0
        self._on_terminate = None

    def _die(self, rc: int) -> None:
        if self._rc is None:
            self._rc = rc
        for stream in (self.stdout, self.stderr):
            try:
                stream.close()
            except Exception:
                pass

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self._on_terminate is not None:
            self._on_terminate()
        self._die(0)

    def kill(self) -> None:
        self._die(-9)

    def finish(self, rc: int = 0) -> None:
        self._die(rc)

    def wait(self, timeout=None):
        if self._rc is None:
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
        return self._rc

    def poll(self):
        return self._rc


class WiredFakePipeline:
    """Fake popen whose capture termination finishes the asr process."""

    def __init__(
        self,
        asr_stdout: bytes,
        asr_stderr: bytes,
        calls: list,
        *,
        asr_dies_at_spawn: bool = False,
        gated: bool = False,
    ) -> None:
        self._asr_stdout = asr_stdout
        self._asr_stderr = asr_stderr
        self._calls = calls
        self._asr_dies = asr_dies_at_spawn
        self._gated = gated
        self._capture: FakeProcess | None = None
        self.asr_processes: list[FakeProcess] = []

    def __call__(self, argv, **kwargs):
        self._calls.append((list(argv), kwargs))
        if kwargs.get("stdin") is None:
            capture = FakeProcess()
            self._capture = capture
            return capture
        asr = FakeProcess(
            self._asr_stdout, self._asr_stderr, gated_stdout=self._gated
        )
        if self._asr_dies:
            asr.finish(1)
        self.asr_processes.append(asr)
        if self._capture is not None:
            self._capture._on_terminate = asr.finish
        return asr

    def asr_spawn_count(self) -> int:
        return len(self.asr_processes)

    def push_line(self, line: bytes) -> None:
        """Deliver one stdout line to the most recently spawned engine."""
        self.asr_processes[-1].stdout.push(line)


ASR_JSONL_OUTPUT = (
    b'{"type":"ready"}\n'
    b'{"type":"partial","text":"\xe7\xa1\xae"}\n'
    b'{"type":"sentence","text":"\xe7\xa1\xae\xe8\xae\xa4","audio_ms":1400,"proc_ms":600,"rtf":0.438}\n'
    b"not-a-json-line\n"
    b'{"type":"final","text":"\xe5\xbc\x80\xe5\xa7\x8b","audio_ms":2000,"proc_ms":810,"rtf":0.405}\n'
)
ASR_STDERR_OUTPUT = "[ModelDownloader] All models available in fake/model\n".encode("utf-8")


class EngineTests(unittest.TestCase):
    """The resident engine: ready semantics, routing swaps, supervision."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.working_dir = Path(self._tmp.name)
        self.logs: list[str] = []

    def _make_engine(
        self,
        calls,
        *,
        resurrect_min_lifetime=30.0,
        asr_dies_at_spawn=False,
        gated=False,
        start_timeout_seconds=5.0,
    ):
        popen = WiredFakePipeline(
            ASR_JSONL_OUTPUT,
            ASR_STDERR_OUTPUT,
            calls,
            asr_dies_at_spawn=asr_dies_at_spawn,
            gated=gated,
        )
        engine = _AsrEngine(
            capture_argv=["arecord", "-D", "default"],
            asr_argv=["asr_pipe_demo", "--vad", "--jsonl"],
            working_dir=self.working_dir,
            grace_seconds=2,
            start_timeout_seconds=start_timeout_seconds,
            on_log=self.logs.append,
            popen=popen,
            resurrect_min_lifetime=resurrect_min_lifetime,
        )
        return engine, popen

    def test_attached_routing_receives_sentences_and_diagnostics(self):
        calls: list = []
        engine, _popen = self._make_engine(calls)
        sentences: list[str] = []
        engine.attach(sentences.append)
        engine.prewarm()
        self.assertTrue(_wait_until(lambda: len(sentences) >= 2))
        self.assertEqual(sentences, ["确认", "开始"])
        joined_logs = "\n".join(self.logs)
        self.assertIn("non-JSON", joined_logs)
        self.assertIn("sentence rtf=0.438 audio_ms=1400", joined_logs)
        self.assertIn("[ModelDownloader]", joined_logs)
        engine.stop()
        # capture is spawned first and stopped first (engine drains on EOF)
        capture_argv = next(argv for argv, kwargs in calls if kwargs.get("stdin") is None)
        self.assertEqual(capture_argv[:2], ["arecord", "-D"])

    def test_prewarm_waits_for_the_ready_event_not_stderr(self):
        """Readiness is the engine's explicit ready JSONL event; stderr
        lines (model downloader chatter) must not release prewarm early."""
        calls: list = []
        engine, popen = self._make_engine(calls, gated=True, start_timeout_seconds=3.0)
        outcome: list = []

        def run_prewarm():
            try:
                engine.prewarm()
                outcome.append("ok")
            except AsrSessionError as exc:
                outcome.append(f"error: {exc}")

        thread = threading.Thread(target=run_prewarm, daemon=True)
        thread.start()
        time.sleep(0.3)
        # stderr chatter already arrived (spawned immediately) but prewarm
        # is still waiting: no ready event yet.
        self.assertIn("[ModelDownloader]", "\n".join(self.logs))
        self.assertTrue(thread.is_alive())
        popen.push_line(b'{"type":"ready"}\n')
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(outcome, ["ok"])
        engine.stop()

    def test_prewarm_times_out_without_ready_event(self):
        calls: list = []
        engine, _popen = self._make_engine(
            calls, gated=True, start_timeout_seconds=0.5
        )
        with self.assertRaises(AsrSessionError):
            engine.prewarm()

    def test_attach_replaces_the_previous_routing(self):
        calls: list = []
        engine, popen = self._make_engine(calls, gated=True)
        first: list[str] = []
        second: list[str] = []
        engine.attach(first.append)
        engine.attach(second.append)
        engine.prewarm()
        popen.push_line(b'{"type":"sentence","text":"\xe7\xa1\xae\xe8\xae\xa4"}\n')
        self.assertTrue(_wait_until(lambda: len(second) >= 1))
        self.assertEqual(first, [])
        engine.stop()

    def test_detach_stops_dispatch(self):
        calls: list = []
        engine, popen = self._make_engine(calls, gated=True)
        sentences: list[str] = []
        handle = engine.attach(sentences.append)
        engine.detach(handle)
        engine.prewarm()
        popen.push_line(b'{"type":"sentence","text":"\xe7\xa1\xae\xe8\xae\xa4"}\n')
        time.sleep(0.2)
        self.assertEqual(sentences, [])
        engine.stop()

    def test_prewarm_is_idempotent_while_alive(self):
        calls: list = []
        engine, popen = self._make_engine(calls)
        engine.attach(lambda _text: None)
        engine.prewarm()
        engine.prewarm()
        self.assertEqual(popen.asr_spawn_count(), 1)
        engine.stop()

    def test_prewarm_raises_when_engine_exits_during_startup(self):
        calls: list = []
        engine, popen = self._make_engine(calls, asr_dies_at_spawn=True)
        with self.assertRaises(AsrSessionError):
            engine.prewarm()
        # The failed startup suppresses background spawns: no retry loop.
        time.sleep(0.2)
        self.assertEqual(popen.asr_spawn_count(), 1)
        # An explicit intent (attach) is what recovers the engine.
        engine.attach(lambda _text: None)
        self.assertTrue(_wait_until(lambda: popen.asr_spawn_count() >= 2))
        engine.stop()

    def test_unexpected_death_respawns_and_rebinds_routing(self):
        calls: list = []
        engine, popen = self._make_engine(calls, gated=True, resurrect_min_lifetime=0.0)
        sentences: list[str] = []
        engine.attach(sentences.append)
        engine.prewarm()
        popen.push_line(b'{"type":"sentence","text":"\xe7\xa1\xae\xe8\xae\xa4"}\n')
        self.assertTrue(_wait_until(lambda: len(sentences) >= 1))
        # Kill the live asr process: supervision respawns and rebinds the
        # same routing, so the fresh stream delivers the sentences again.
        popen.asr_processes[0].finish(1)
        self.assertTrue(_wait_until(lambda: popen.asr_spawn_count() >= 2))
        popen.push_line(b'{"type":"final","text":"\xe5\xbc\x80\xe5\xa7\x8b"}\n')
        self.assertTrue(_wait_until(lambda: len(sentences) >= 2))
        self.assertTrue(engine.alive)
        engine.stop()

    def test_early_death_waits_for_the_next_attach(self):
        calls: list = []
        engine, popen = self._make_engine(calls, gated=True, resurrect_min_lifetime=30.0)
        sentences: list[str] = []
        engine.attach(sentences.append)
        engine.prewarm()
        popen.push_line(b'{"type":"sentence","text":"\xe7\xa1\xae\xe8\xae\xa4"}\n')
        self.assertTrue(_wait_until(lambda: len(sentences) >= 1))
        popen.asr_processes[0].finish(1)
        self.assertTrue(_wait_until(lambda: not engine.alive))
        # The engine only lived for milliseconds: no auto-respawn, no loop.
        time.sleep(0.3)
        self.assertEqual(popen.asr_spawn_count(), 1)
        # The next attach (routing switch) is what recovers the engine.
        engine.attach(lambda _text: None)
        self.assertTrue(_wait_until(lambda: popen.asr_spawn_count() >= 2))
        engine.stop()

    def test_stop_disables_supervision(self):
        calls: list = []
        engine, popen = self._make_engine(calls, resurrect_min_lifetime=0.0)
        engine.attach(lambda _text: None)
        engine.prewarm()
        engine.stop()
        time.sleep(0.2)
        self.assertEqual(popen.asr_spawn_count(), 1)


class FakeEngine:
    """Engine double for provider-level tests (argv building, lifecycle)."""

    instances: list["FakeEngine"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.routings: list[dict] = []
        self.current: dict | None = None
        self.prewarm_calls = 0
        self.stop_calls = 0
        self.alive = False
        self.raise_on_prewarm = False
        FakeEngine.instances.append(self)

    def attach(self, on_sentence, on_log=None) -> dict:
        routing = {"on_sentence": on_sentence, "on_log": on_log}
        self.routings.append(routing)
        self.current = routing
        return routing

    def detach(self, routing) -> None:
        if self.current is routing:
            self.current = None

    def prewarm(self) -> None:
        self.prewarm_calls += 1
        if self.raise_on_prewarm:
            raise AsrSessionError("load failed")
        self.alive = True

    def stop(self) -> None:
        self.stop_calls += 1
        self.alive = False


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _make_board_tree(self) -> None:
        binary = self.root / "asr/sensevoice-x100/build/bin/asr_pipe_demo"
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(b"#!/bin/sh\n")
        binary.chmod(0o755)
        model = self.root / "asr/sensevoice-x100/model"
        model.mkdir(parents=True, exist_ok=True)
        (model / "model_quant_optimized.onnx").write_bytes(b"model")

    def test_health_reports_missing_runtime(self):
        provider = SensevoiceAsrProvider(project_root=self.root)
        health = provider.health()
        self.assertFalse(health["ok"])
        self.assertTrue(any("binary missing" in p for p in health["problems"]))
        self.assertTrue(any("model missing" in p for p in health["problems"]))
        self.assertFalse(health["running"])

    def test_health_ok_with_binary_and_model(self):
        self._make_board_tree()
        provider = SensevoiceAsrProvider(project_root=self.root)
        health = provider.health()
        self.assertTrue(health["ok"])
        self.assertNotIn("problems", health)
        self.assertFalse(health["running"])

    @patch.object(asr_provider_module, "_AsrEngine", FakeEngine)
    def test_start_session_builds_argv_and_attaches(self):
        FakeEngine.instances = []
        self._make_board_tree()
        provider = SensevoiceAsrProvider(project_root=self.root)
        provider.start_session(lambda _text: None)
        engine = FakeEngine.instances[0]
        asr_argv = engine.kwargs["asr_argv"]
        self.assertEqual(asr_argv[0], str(provider._binary))
        self.assertIn("--vad", asr_argv)
        self.assertIn("--jsonl", asr_argv)
        self.assertEqual(asr_argv[asr_argv.index("--model-dir") + 1], str(provider._model_dir))
        self.assertEqual(asr_argv[asr_argv.index("--language") + 1], "auto")
        self.assertEqual(asr_argv[asr_argv.index("--core-arch") + 1], "x100")
        self.assertEqual(asr_argv[asr_argv.index("--vad-thresh") + 1], "400")
        self.assertEqual(asr_argv[asr_argv.index("--pause") + 1], "0.6")
        self.assertEqual(asr_argv[asr_argv.index("--max-utt") + 1], "8.0")
        capture_argv = engine.kwargs["capture_argv"]
        self.assertEqual(capture_argv[0], "arecord")
        self.assertIn("default", capture_argv)
        # A second session replaces the routing instead of raising: sessions
        # are logical attaches on the resident engine.
        handle = provider.start_session(lambda _text: None)
        self.assertIs(engine.current, handle)

    @patch.object(asr_provider_module, "_AsrEngine", FakeEngine)
    def test_stop_session_detaches_current_routing(self):
        FakeEngine.instances = []
        self._make_board_tree()
        provider = SensevoiceAsrProvider(project_root=self.root)
        handle = provider.start_session(lambda _text: None)
        engine = FakeEngine.instances[0]
        provider.stop_session(handle)
        self.assertIsNone(engine.current)
        provider.start_session(lambda _text: None)  # re-attach: no exception
        self.assertIsNotNone(engine.current)

    @patch.object(asr_provider_module, "_AsrEngine", FakeEngine)
    def test_cpu_affinity_wraps_argv_with_taskset(self):
        FakeEngine.instances = []
        self._make_board_tree()
        provider = SensevoiceAsrProvider(project_root=self.root)
        provider._config["runtime"]["cpu_affinity"] = "0,3"
        provider.start_session(lambda _text: None)
        self.assertEqual(
            FakeEngine.instances[0].kwargs["asr_argv"][:3], ["taskset", "-c", "0,3"]
        )

    @patch.object(asr_provider_module, "_AsrEngine", FakeEngine)
    def test_stop_session_ignores_foreign_handles(self):
        FakeEngine.instances = []
        self._make_board_tree()
        provider = SensevoiceAsrProvider(project_root=self.root)
        provider.start_session(lambda _text: None)
        engine = FakeEngine.instances[0]
        provider.stop_session("not-a-session")  # no error, routing untouched
        self.assertIsNotNone(engine.current)

    @patch.object(asr_provider_module, "_AsrEngine", FakeEngine)
    def test_prewarm_spawns_and_reports_alive(self):
        FakeEngine.instances = []
        self._make_board_tree()
        provider = SensevoiceAsrProvider(project_root=self.root)
        provider.prewarm()
        engine = FakeEngine.instances[0]
        self.assertEqual(engine.prewarm_calls, 1)
        self.assertTrue(provider.health()["running"])
        provider.stop_session("anything")  # detach keeps the engine alive
        self.assertTrue(provider.health()["running"])

    @patch.object(asr_provider_module, "_AsrEngine", FakeEngine)
    def test_prewarm_failure_raises(self):
        FakeEngine.instances = []
        self._make_board_tree()
        provider = SensevoiceAsrProvider(project_root=self.root)
        provider._ensure_engine().raise_on_prewarm = True
        with self.assertRaises(AsrSessionError):
            provider.prewarm()

    @patch.object(asr_provider_module, "_AsrEngine", FakeEngine)
    def test_shutdown_stops_engine(self):
        FakeEngine.instances = []
        self._make_board_tree()
        provider = SensevoiceAsrProvider(project_root=self.root)
        provider.prewarm()
        provider.shutdown()
        engine = FakeEngine.instances[0]
        self.assertEqual(engine.stop_calls, 1)
        self.assertFalse(provider.health()["running"])
        provider.stop_session("anything")  # engine gone: harmless no-op


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
        provider = SensevoiceAsrProvider()
        registry.register(provider, {
            "id": "asr_sensevoice",
            "type": "asr",
            "entry": "provider.py:SensevoiceAsrProvider",
        })
        self.assertEqual(registry.provider_ids("asr"), ["asr_sensevoice"])
        self.assertEqual(registry.require("asr_sensevoice", expected_type="asr"), provider)

    def test_packaged_manifest_is_valid(self):
        manifest_path = ROOT / "backend/components/asr_sensevoice/manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_manifest(manifest_path, manifest)


if __name__ == "__main__":
    unittest.main()
