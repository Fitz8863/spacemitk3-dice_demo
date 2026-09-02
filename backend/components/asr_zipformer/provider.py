"""Zipformer streaming ASR provider (board-local subprocess session).

Bridges the dice arena to ``asr/zipformer-streaming`` (see that directory's
AGENTS.md for the JSONL contract).  One listening session spawns an
``arecord | stream_asr --pcm --jsonl`` pipeline: the reader thread turns
``sentence``/``final`` events into ``on_sentence`` callbacks, everything else
becomes diagnostics.  Sessions are created on demand by the round engine, so
this package has no lifecycle scripts and nothing to pre-start.
"""
from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable

from core.asr import AsrProvider, AsrSessionError

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_CAPTURE_FORMATS = {"S16_LE"}


class AsrConfigError(ValueError):
    """Raised when asr_zipformer/config.json is malformed."""


def _resolve_repo_path(value: Any, field: str, project_root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise AsrConfigError(f"{field} must be a non-empty repository-relative string")
    raw = Path(value)
    if raw.is_absolute() or any(part in {"", ".", ".."} for part in raw.parts):
        raise AsrConfigError(f"{field} must stay inside the project")
    root = project_root.resolve()
    candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise AsrConfigError(f"{field} escapes the project") from exc
    return candidate


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise AsrConfigError(f"{field} must be an integer >= 1")
    return value


def load_config(package_dir: Path, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Validate asr_zipformer/config.json without touching the filesystem.

    Existence of the binary/model is deliberately NOT checked here: the
    registry instantiates providers on any machine, while ``health()`` and
    session startup report board-local readiness.
    """
    path = Path(package_dir) / "config.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AsrConfigError("config.json is missing") from exc
    except json.JSONDecodeError as exc:
        raise AsrConfigError(f"config.json is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise AsrConfigError("config schema_version must be 1")

    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        raise AsrConfigError("runtime must be an object")
    _resolve_repo_path(runtime.get("binary"), "runtime.binary", project_root)
    _resolve_repo_path(runtime.get("working_dir"), "runtime.working_dir", project_root)
    model_dir = runtime.get("model_dir")
    if (
        not isinstance(model_dir, str)
        or not model_dir.strip()
        or Path(model_dir).is_absolute()
        or any(part in {"", ".", ".."} for part in Path(model_dir).parts)
    ):
        raise AsrConfigError("runtime.model_dir must be a working-dir-relative directory name")

    capture = runtime.get("capture", {})
    if not isinstance(capture, dict):
        raise AsrConfigError("runtime.capture must be an object")
    device = capture.get("device", "default")
    if not isinstance(device, str) or not device.strip():
        raise AsrConfigError("capture.device must be a non-empty string")
    if capture.get("sample_rate", 16000) != 16000:
        raise AsrConfigError("capture.sample_rate must be 16000 (engine contract)")
    if capture.get("channels", 1) != 1:
        raise AsrConfigError("capture.channels must be 1 (engine contract)")
    if capture.get("format", "S16_LE") not in _CAPTURE_FORMATS:
        raise AsrConfigError(f"capture.format must be one of {sorted(_CAPTURE_FORMATS)}")

    vad = runtime.get("vad", {})
    if not isinstance(vad, dict):
        raise AsrConfigError("runtime.vad must be an object")
    if not isinstance(vad.get("enabled", True), bool):
        raise AsrConfigError("vad.enabled must be boolean")
    _positive_int(vad.get("rms", 400), "vad.rms")
    _positive_int(vad.get("pause_ms", 600), "vad.pause_ms")
    _positive_int(vad.get("max_ms", 8000), "vad.max_ms")

    affinity = str(runtime.get("cpu_affinity", "") or "")
    if affinity:
        cores = [part.strip() for part in affinity.split(",")]
        if not cores or not all(core.isdigit() and core for core in cores):
            raise AsrConfigError("runtime.cpu_affinity must be a comma-separated core list like '0,3'")

    _positive_int(runtime.get("start_timeout_seconds", 15), "runtime.start_timeout_seconds")
    _positive_int(runtime.get("terminate_grace_seconds", 5), "runtime.terminate_grace_seconds")
    return payload


class _AsrSession:
    """One ``arecord | stream_asr --pcm --jsonl`` pipeline plus reader threads."""

    def __init__(
        self,
        *,
        capture_argv: list[str],
        asr_argv: list[str],
        working_dir: Path,
        grace_seconds: int,
        on_sentence: Callable[[str], None],
        on_log: Callable[[str], None],
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    ) -> None:
        self._capture_argv = capture_argv
        self._asr_argv = asr_argv
        self._working_dir = working_dir
        self._grace_seconds = grace_seconds
        self._on_sentence = on_sentence
        self._on_log = on_log
        self._popen = popen
        self._capture: subprocess.Popen | None = None
        self._asr: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop_lock = threading.Lock()
        self._stopped = False

    @property
    def running(self) -> bool:
        return self._reader is not None and self._reader.is_alive()

    def start(self) -> None:
        self._capture = self._popen(
            self._capture_argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        assert self._capture.stdout is not None
        self._asr = self._popen(
            self._asr_argv,
            stdin=self._capture.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self._working_dir),
            start_new_session=True,
        )
        # The parent must drop its copy so only stream_asr reads arecord.
        self._capture.stdout.close()
        self._reader = threading.Thread(target=self._read_events, name="asr-jsonl", daemon=True)
        self._stderr_reader = threading.Thread(target=self._read_stderr, name="asr-stderr", daemon=True)
        self._reader.start()
        self._stderr_reader.start()

    def wait_ready(self, timeout_seconds: float) -> bool:
        """Wait until the engine printed its first stderr line (model loaded)."""
        return self._ready.wait(timeout_seconds)

    def _read_stderr(self) -> None:
        assert self._asr is not None and self._asr.stderr is not None
        try:
            for raw in self._asr.stderr:
                line = raw.decode("utf-8", errors="replace").strip()
                if line:
                    self._ready.set()
                    self._on_log(f"{line}")
        except Exception as exc:  # diagnostics only; never kill the backend
            self._on_log(f"stderr reader error: {exc}")
        finally:
            self._ready.set()

    def _read_events(self) -> None:
        assert self._asr is not None and self._asr.stdout is not None
        try:
            for raw in self._asr.stdout:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    self._on_log(f"non-JSON output ignored: {line[:200]}")
                    continue
                if not isinstance(event, dict):
                    continue
                kind = event.get("type")
                if kind in ("sentence", "final"):
                    text = str(event.get("text") or "")
                    if text:
                        try:
                            self._on_sentence(text)
                        except Exception as exc:
                            self._on_log(f"sentence callback error: {exc}")
                elif kind == "stats":
                    rtf = event.get("rtf")
                    self._on_log(f"stats rtf={rtf}")
                # "partial" events are intentionally dropped: too chatty for
                # logs and not used for intent matching.
        except Exception as exc:  # see _read_stderr
            self._on_log(f"event reader error: {exc}")
        finally:
            capture_rc = self._returncode(self._capture)
            asr_rc = self._returncode(self._asr)
            if not self._stopped:
                self._on_log(f"session ended unexpectedly (asr rc={asr_rc}, capture rc={capture_rc})")

    @staticmethod
    def _returncode(process: subprocess.Popen | None) -> Any:
        if process is None:
            return None
        try:
            return process.wait(timeout=0)
        except subprocess.TimeoutExpired:
            return "running"

    def stop(self) -> None:
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True
        # Stop the capture first: stream_asr then sees stdin EOF, flushes its
        # tail text and exits on its own, so the reader thread drains cleanly.
        if self._capture is not None and self._capture.poll() is None:
            try:
                self._capture.terminate()
            except OSError:
                pass
        if self._asr is not None:
            try:
                self._asr.wait(timeout=self._grace_seconds)
            except subprocess.TimeoutExpired:
                try:
                    self._asr.terminate()
                    self._asr.wait(timeout=2)
                except (subprocess.TimeoutExpired, OSError):
                    try:
                        self._asr.kill()
                    except OSError:
                        pass
        if self._capture is not None:
            try:
                self._capture.wait(timeout=2)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    self._capture.kill()
                except OSError:
                    pass
        for thread in (self._reader, self._stderr_reader):
            if thread is not None:
                thread.join(timeout=5)


class ZipformerAsrProvider(AsrProvider):
    """Streaming ASR via the board-local zipformer runtime subprocess."""

    id = "asr_zipformer"
    type = "asr"

    def __init__(
        self,
        manifest: dict[str, Any] | None = None,
        *,
        project_root: Path | None = None,
    ) -> None:
        super().__init__(manifest)
        root = project_root or PROJECT_ROOT
        self._config = load_config(Path(__file__).parent, project_root=root)
        runtime = self._config["runtime"]
        self._binary = _resolve_repo_path(runtime["binary"], "runtime.binary", root)
        self._working_dir = _resolve_repo_path(runtime["working_dir"], "runtime.working_dir", root)
        self._model_dir = self._working_dir / str(runtime["model_dir"])
        self._lock = threading.Lock()
        self._session: _AsrSession | None = None

    def health(self) -> dict[str, Any]:
        problems: list[str] = []
        if not self._binary.is_file():
            problems.append(f"binary missing: {self._binary}")
        elif not self._binary.stat().st_mode & 0o111:
            problems.append(f"binary not executable: {self._binary}")
        if not self._model_dir.is_dir():
            problems.append(
                "model dir missing; download per asr/zipformer-streaming/README.md"
            )
        health = {"id": self.id, "type": self.type, "ok": not problems}
        if problems:
            health["problems"] = problems
        return health

    def start_session(
        self,
        on_sentence: Callable[[str], None],
        *,
        on_log: Callable[[str], None] | None = None,
    ) -> Any:
        on_log = on_log if callable(on_log) else (lambda _message: None)
        runtime = self._config["runtime"]
        capture = runtime.get("capture", {})
        capture_argv = [
            "arecord",
            "-D",
            str(capture.get("device", "default")),
            "-q",
            "-f",
            str(capture.get("format", "S16_LE")),
            "-r",
            "16000",
            "-c",
            "1",
            "-t",
            "raw",
        ]
        vad = runtime.get("vad", {})
        asr_argv = [
            str(self._binary),
            "--pcm",
            "--jsonl",
            "--model-dir",
            str(self._model_dir),
        ]
        if vad.get("enabled", True):
            asr_argv += [
                "--vad-rms",
                str(int(vad.get("rms", 400))),
                "--vad-pause-ms",
                str(int(vad.get("pause_ms", 600))),
                "--vad-max-ms",
                str(int(vad.get("max_ms", 8000))),
            ]
        else:
            asr_argv.append("--no-vad")
        affinity = str(runtime.get("cpu_affinity", "") or "").strip()
        if affinity:
            asr_argv = ["taskset", "-c", affinity, *asr_argv]

        session = _AsrSession(
            capture_argv=capture_argv,
            asr_argv=asr_argv,
            working_dir=self._working_dir,
            grace_seconds=int(runtime.get("terminate_grace_seconds", 5)),
            on_sentence=on_sentence,
            on_log=on_log,
        )
        with self._lock:
            if self._session is not None and self._session.running:
                raise AsrSessionError("another ASR session is already listening")
            session.start()
            self._session = session
        start_timeout = float(runtime.get("start_timeout_seconds", 15))
        if not session.wait_ready(start_timeout):
            if not session.running:
                raise AsrSessionError("ASR process exited during startup (see asr log)")
            on_log(
                f"model load exceeded {start_timeout:g}s; continuing anyway"
            )
        return session

    def stop_session(self, handle: Any) -> None:
        if not isinstance(handle, _AsrSession):
            return
        with self._lock:
            if self._session is handle:
                self._session = None
        handle.stop()
