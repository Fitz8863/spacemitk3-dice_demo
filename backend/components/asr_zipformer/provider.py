"""Zipformer streaming ASR provider (board-local resident engine).

The engine pair (``arecord | stream_asr --pcm --jsonl``) is resident: it is
spawned once by the backend's startup prewarm (or lazily on the first
attach) and stays up for the whole process lifetime.  A "session" is a
logical routing attached to the engine — switching between round intent
listening and standby wake-word listening is an instant callback swap, so
the ~2.6 s zipformer model load is paid exactly once per process instead of
at every round/standby transition.

Failure behaviour: a crashed engine is logged loudly; it respawns in the
background and re-binds the attached routing only when it had been running
long enough to count as healthy (``resurrect_min_lifetime``), so a binary
that crash-loops at startup cannot spin respawn cycles — it recovers at the
next attach (round/standby switch) instead.  ``prewarm()`` is the one
blocking entry point (startup, fail-fast); ``shutdown()`` releases the
engine and the microphone.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from core.asr import AsrProvider, AsrSessionError

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_CAPTURE_FORMATS = {"S16_LE"}

# An engine that died sooner than this after spawning is treated as
# unhealthy (crash-loop risk): no automatic respawn, recover on next attach.
_RESURRECT_MIN_LIFETIME_SECONDS = 30.0


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
    try:
        payload = json.loads((package_dir / "config.json").read_text(encoding="utf-8"))
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


class _Routing:
    """One attached callback pair; its identity is the detach token."""

    __slots__ = ("on_sentence", "on_log")

    def __init__(
        self,
        on_sentence: Callable[[str], None],
        on_log: Callable[[str], None] | None,
    ) -> None:
        self.on_sentence = on_sentence
        self.on_log = on_log


class _AsrEngine:
    """Resident ``arecord | stream_asr`` pair with one swappable routing.

    Spawning is serialized (``_spawn_lock``) and only ever replaces a dead
    engine.  Routing attach/detach is an instant callback swap under
    ``_lock``; sentences already in flight may land on either side of a
    switch, which is fine — it is one utterance.
    """

    def __init__(
        self,
        *,
        capture_argv: list[str],
        asr_argv: list[str],
        working_dir: Path,
        grace_seconds: int,
        start_timeout_seconds: float,
        on_log: Callable[[str], None],
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
        resurrect_min_lifetime: float = _RESURRECT_MIN_LIFETIME_SECONDS,
    ) -> None:
        self._capture_argv = capture_argv
        self._asr_argv = asr_argv
        self._working_dir = working_dir
        self._grace_seconds = grace_seconds
        self._start_timeout = start_timeout_seconds
        self._on_log = on_log
        self._popen = popen
        self._resurrect_min_lifetime = resurrect_min_lifetime
        self._lock = threading.Lock()
        self._spawn_lock = threading.Lock()
        self._routing: _Routing | None = None
        self._capture: subprocess.Popen | None = None
        self._asr: subprocess.Popen | None = None
        self._ready = threading.Event()
        self._events_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._spawned_at = 0.0
        self._stopped = False
        self._supervision_enabled = True
        # Set by a failed prewarm: blocks background spawns until the next
        # explicit attach/prewarm, so a queued "respawn" cannot undo the
        # abort (a failed startup must leave the engine down).
        self._spawn_suppressed = False

    def _log(self, message: str) -> None:
        self._on_log(message)

    @property
    def alive(self) -> bool:
        with self._lock:
            asr = self._asr
        return asr is not None and asr.poll() is None

    # ---- routing --------------------------------------------------------

    def attach(
        self,
        on_sentence: Callable[[str], None],
        on_log: Callable[[str], None] | None = None,
    ) -> _Routing:
        """Bind a new routing, replacing any previous one.  Never blocks."""
        routing = _Routing(on_sentence, on_log)
        with self._lock:
            self._routing = routing
            # An attach is an explicit intent to hear things: it lifts a
            # suppression left behind by a failed prewarm.
            self._spawn_suppressed = False
            asr = self._asr
        if asr is None or asr.poll() is not None:
            self._log("ASR engine is down; respawning in the background")
            threading.Thread(
                target=self._spawn_if_dead, name="asr-engine-spawn", daemon=True
            ).start()
        return routing

    def detach(self, routing: Any) -> None:
        """Remove the routing if it is the current one.  Idempotent."""
        with self._lock:
            if routing is not None and self._routing is routing:
                self._routing = None

    # ---- lifecycle ------------------------------------------------------

    def prewarm(self) -> None:
        """Spawn and wait for the model (blocking; the one slow entry point)."""
        with self._spawn_lock:
            with self._lock:
                self._spawn_suppressed = False
                stopped = self._stopped
            if stopped:
                raise AsrSessionError("ASR engine was shut down")
            if self.alive:
                return
            try:
                self._spawn_processes()
            except OSError as exc:
                raise AsrSessionError(f"failed to spawn the ASR engine: {exc}") from exc
            if not self._ready.wait(self._start_timeout) or not self.alive:
                self._abort_processes()
                raise AsrSessionError(
                    f"ASR model load did not complete within {self._start_timeout:g}s "
                    "(engine exited or stalled; see the asr log)"
                )

    def stop(self) -> None:
        """Deliberate full teardown: kill the pair; the object is done."""
        self._teardown(permanent=True)

    def _spawn_if_dead(self) -> None:
        with self._spawn_lock:
            with self._lock:
                if self._stopped or self._spawn_suppressed:
                    return
            if self.alive:
                return
            try:
                self._spawn_processes()
            except OSError as exc:
                self._log(f"ASR engine respawn failed: {exc}")

    def _spawn_processes(self) -> None:
        capture = self._popen(
            self._capture_argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            asr = self._popen(
                self._asr_argv,
                stdin=capture.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self._working_dir),
                start_new_session=True,
            )
        except BaseException:
            # The capture half must not outlive a failed engine spawn.
            try:
                capture.terminate()
                capture.wait(timeout=2)
            except Exception:
                pass
            raise
        finally:
            # The parent must drop its copy so only stream_asr reads arecord.
            capture.stdout.close()
        ready = threading.Event()
        events_thread = threading.Thread(
            target=self._read_events, args=(asr, ready), name="asr-jsonl", daemon=True
        )
        stderr_thread = threading.Thread(
            target=self._read_stderr, args=(asr, ready), name="asr-stderr", daemon=True
        )
        with self._lock:
            self._capture = capture
            self._asr = asr
            self._ready = ready
            self._spawned_at = time.monotonic()
            self._events_thread = events_thread
            self._stderr_thread = stderr_thread
            # Publish and start under the lock: a concurrent teardown must
            # never observe an assigned-but-unstarted thread (join raises).
            events_thread.start()
            stderr_thread.start()

    def _abort_processes(self) -> None:
        """Tear a failed spawn down; the engine stays down until the next
        explicit intent (attach/prewarm) — a queued background spawn must
        not undo the abort."""
        with self._lock:
            self._spawn_suppressed = True
        self._teardown(permanent=False)

    # ---- reader threads -------------------------------------------------

    def _read_events(self, asr: subprocess.Popen, ready: threading.Event) -> None:
        try:
            for raw in asr.stdout:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    self._log(f"non-JSON output ignored: {line[:200]}")
                    continue
                if not isinstance(event, dict):
                    continue
                kind = event.get("type")
                if kind in ("sentence", "final"):
                    text = str(event.get("text") or "")
                    with self._lock:
                        routing = self._routing
                    if routing is not None and text:
                        try:
                            routing.on_sentence(text)
                        except Exception as exc:
                            self._log(f"sentence callback error: {exc}")
                elif kind == "stats":
                    self._log(f"stats rtf={event.get('rtf')}")
                # "partial" events are intentionally dropped: too chatty for
                # logs and not used for intent matching.
        except Exception as exc:  # diagnostics only; never kill the backend
            self._log(f"event reader error: {exc}")
        finally:
            with self._lock:
                has_routing = self._routing is not None
                stopped = self._stopped
                supervision = self._supervision_enabled
                capture = self._capture
            ready.set()
            capture_rc = self._returncode(capture)
            asr_rc = self._returncode(asr)
            if stopped:
                return
            if not supervision or not has_routing:
                self._log(f"ASR engine ended (asr rc={asr_rc}, capture rc={capture_rc})")
                return
            lifetime = time.monotonic() - self._spawned_at
            if lifetime < self._resurrect_min_lifetime:
                self._log(
                    f"ASR engine died after only {lifetime:.1f}s "
                    f"(asr rc={asr_rc}); no auto-respawn — recovering at the "
                    "next attach"
                )
                return
            self._log(
                f"ASR engine died unexpectedly (asr rc={asr_rc}); "
                "respawning in the background"
            )
            threading.Thread(
                target=self._spawn_if_dead, name="asr-engine-respawn", daemon=True
            ).start()

    def _read_stderr(self, asr: subprocess.Popen, ready: threading.Event) -> None:
        try:
            for raw in asr.stderr:
                line = raw.decode("utf-8", errors="replace").strip()
                if line:
                    ready.set()
                    self._log(line)
        except Exception as exc:
            self._log(f"stderr reader error: {exc}")
        finally:
            ready.set()

    @staticmethod
    def _returncode(process: subprocess.Popen | None) -> Any:
        if process is None:
            return None
        try:
            return process.wait(timeout=0)
        except subprocess.TimeoutExpired:
            return "running"

    # ---- teardown -------------------------------------------------------

    def _teardown(self, *, permanent: bool) -> None:
        with self._lock:
            self._routing = None
            if permanent:
                self._stopped = True
            else:
                self._supervision_enabled = False
            capture, asr = self._capture, self._asr
            events_thread, stderr_thread = self._events_thread, self._stderr_thread
        # Stop the capture first: stream_asr then sees stdin EOF, flushes its
        # tail text and exits on its own, so the reader threads drain cleanly.
        if capture is not None and capture.poll() is None:
            try:
                capture.terminate()
            except OSError:
                pass
        if asr is not None:
            try:
                asr.wait(timeout=self._grace_seconds)
            except subprocess.TimeoutExpired:
                try:
                    asr.terminate()
                    asr.wait(timeout=2)
                except (subprocess.TimeoutExpired, OSError):
                    try:
                        asr.kill()
                    except OSError:
                        pass
        if capture is not None:
            try:
                capture.wait(timeout=2)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    capture.kill()
                except OSError:
                    pass
        for thread in (events_thread, stderr_thread):
            if thread is not None:
                thread.join(timeout=5)
        with self._lock:
            self._capture = None
            self._asr = None
            self._events_thread = None
            self._stderr_thread = None
            if not permanent:
                self._supervision_enabled = True


class ZipformerAsrProvider(AsrProvider):
    """Streaming ASR via the board-local zipformer resident engine."""

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
        self._engine: _AsrEngine | None = None
        self._engine_lock = threading.Lock()

    def _log(self, message: str) -> None:
        print(f"[asr_zipformer] {message}", flush=True)

    def _build_engine(self) -> _AsrEngine:
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
        return _AsrEngine(
            capture_argv=capture_argv,
            asr_argv=asr_argv,
            working_dir=self._working_dir,
            grace_seconds=int(runtime.get("terminate_grace_seconds", 5)),
            start_timeout_seconds=float(runtime.get("start_timeout_seconds", 15)),
            on_log=self._log,
        )

    def _ensure_engine(self) -> _AsrEngine:
        with self._engine_lock:
            if self._engine is None:
                self._engine = self._build_engine()
            return self._engine

    # ---- lifecycle ------------------------------------------------------

    def prewarm(self) -> None:
        """Spawn and warm the engine (backend startup calls this, blocking)."""
        self._ensure_engine().prewarm()

    def shutdown(self) -> None:
        with self._engine_lock:
            engine = self._engine
            self._engine = None
        if engine is not None:
            engine.stop()

    # ---- sessions (logical routings on the resident engine) --------------

    def start_session(
        self,
        on_sentence: Callable[[str], None],
        *,
        on_log: Callable[[str], None] | None = None,
    ) -> Any:
        return self._ensure_engine().attach(on_sentence, on_log)

    def stop_session(self, handle: Any) -> None:
        with self._engine_lock:
            engine = self._engine
        if engine is not None:
            engine.detach(handle)

    # ---- health ---------------------------------------------------------

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
        with self._engine_lock:
            engine = self._engine
        running = engine is not None and engine.alive
        health = {"id": self.id, "type": self.type, "ok": not problems, "running": running}
        if problems:
            health["problems"] = problems
        return health
