"""SenseVoice ASR provider (board-local resident engine, sentence-segmented).

SenseVoice is a NON-streaming recognizer: it transcribes complete
utterances.  The resident engine pair is ``arecord | asr_pipe_demo --vad
--jsonl``: the pipe demo energy-VAD-segments the mic stream into utterances,
recognizes each one with SenseVoice, and emits JSON Lines events on stdout
(``ready`` after model load + warmup, ``sentence`` per finalized utterance).
To the arena this is indistinguishable from the streaming zipformer engine:
a "session" is a logical routing attached to the resident pair, swapped
instantly between round intent listening and standby wake-word listening.

The inference threads run on the X100 general cores (CPU 0-7) by default
(``core_arch: "x100"``), keeping the A100 EP cores free for the TTS/YOLO
engines; idle cost is near zero because nothing is decoded while nobody
speaks (unlike a streaming decoder whose VAD loop runs continuously).

Engine lifecycle mirrors ``asr_zipformer``: ``prewarm()`` is the one
blocking entry point (spawn + wait for the ``ready`` event, fail-fast),
a crashed engine respawns in the background once it had proven healthy,
and ``shutdown()`` releases the engine and the microphone.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from core.asr import AsrProvider, AsrSessionError

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_CAPTURE_FORMATS = {"S16_LE"}
_LANGUAGES = {"zh", "en", "ja", "ko", "yue", "auto"}
_CORE_ARCHS = {"x100", "a100", "auto"}

# An engine that died sooner than this after spawning is treated as
# unhealthy (crash-loop risk): no automatic respawn, recover on next attach.
_RESURRECT_MIN_LIFETIME_SECONDS = 30.0


class AsrConfigError(ValueError):
    """Raised when asr_sensevoice/config.json is malformed."""


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


def _parse_affinity(value: str) -> tuple[str, list[int]]:
    """Split a cpu_affinity value into ("", []), ("x100", cores) or
    ("a100", cores).  Raises AsrConfigError on malformed or mixed lists."""
    cores = [part.strip() for part in value.split(",")]
    if not cores or not all(core.isdigit() and core for core in cores):
        raise AsrConfigError(
            "runtime.cpu_affinity must be a comma-separated core list like '6,7', or 'none'"
        )
    numbers = [int(core) for core in cores]
    if any(not (0 <= core <= 15) for core in numbers):
        raise AsrConfigError("runtime.cpu_affinity cores must be within 0-15")
    x100 = [core for core in numbers if core <= 7]
    a100 = [core for core in numbers if core >= 8]
    if x100 and a100:
        raise AsrConfigError(
            "runtime.cpu_affinity must stay within one cluster: X100 (0-7) or A100 (8-15)"
        )
    if a100:
        if not (1 <= len(a100) <= 2):
            raise AsrConfigError(
                "A100 affinity allows 1 or 2 cores (EP inference threads = cores)"
            )
        return "a100", numbers
    return "x100", numbers


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
    _resolve_repo_path(runtime.get("model_dir"), "runtime.model_dir", project_root)
    if runtime.get("engine", "sensevoice") != "sensevoice":
        raise AsrConfigError("runtime.engine must be 'sensevoice' (other engines have their own components)")
    if runtime.get("language", "auto") not in _LANGUAGES:
        raise AsrConfigError(f"runtime.language must be one of {sorted(_LANGUAGES)}")
    core_arch = runtime.get("core_arch", "x100")
    if core_arch not in _CORE_ARCHS:
        raise AsrConfigError(f"runtime.core_arch must be one of {sorted(_CORE_ARCHS)}")
    _positive_int(runtime.get("num_threads", 2), "runtime.num_threads")

    affinity = str(runtime.get("cpu_affinity", "") or "").strip()
    if affinity and affinity != "none":
        cluster, _cores = _parse_affinity(affinity)
        if cluster == "a100" and core_arch == "x100":
            raise AsrConfigError(
                "cpu_affinity selects A100 cores but core_arch is x100; set "
                "core_arch to a100 (or auto) or pick X100 cores (0-7)"
            )

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

    # The pipe demo's non-VAD mode is timed flushing (one recognition every
    # N seconds of audio regardless of speech); it is unusable for command
    # listening, so this component is VAD-segmented, always.
    vad = runtime.get("vad", {})
    if not isinstance(vad, dict):
        raise AsrConfigError("runtime.vad must be an object")
    if vad.get("enabled", True) is not True:
        raise AsrConfigError("vad.enabled must be true (sentence segmentation is this engine's contract)")
    _positive_int(vad.get("rms", 400), "vad.rms")
    _positive_int(vad.get("pause_ms", 600), "vad.pause_ms")
    _positive_int(vad.get("max_ms", 8000), "vad.max_ms")

    _positive_int(runtime.get("start_timeout_seconds", 30), "runtime.start_timeout_seconds")
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
    """Resident ``arecord | asr_pipe_demo`` pair with one swappable routing.

    Mirrors the zipformer engine with one protocol difference: readiness is
    the engine's explicit ``{"type":"ready"}`` JSONL event (emitted after
    model load AND warmup).  stderr lines must NOT mark the engine ready —
    the model downloader prints to stderr well before the model is loaded.
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
        spawn_env: dict[str, str] | None = None,
    ) -> None:
        self._capture_argv = capture_argv
        self._asr_argv = asr_argv
        self._working_dir = working_dir
        self._grace_seconds = grace_seconds
        self._start_timeout = start_timeout_seconds
        self._on_log = on_log
        self._popen = popen
        self._resurrect_min_lifetime = resurrect_min_lifetime
        # Extra environment for the engine process only (inherited
        # os.environ plus these overrides) — used to hand the EP its
        # precise A100 thread-affinity list.
        self._spawn_env = dict(spawn_env) if spawn_env else None
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
        """Spawn and wait for the ``ready`` event (blocking; the one slow
        entry point — model load plus warmup synthesis)."""
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
                **(
                    {"env": {**os.environ, **self._spawn_env}}
                    if self._spawn_env
                    else {}
                ),
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
            # The parent must drop its copy so only the engine reads arecord.
            capture.stdout.close()
        ready = threading.Event()
        events_thread = threading.Thread(
            target=self._read_events, args=(asr, ready), name="asr-jsonl", daemon=True
        )
        stderr_thread = threading.Thread(
            target=self._read_stderr, args=(asr,), name="asr-stderr", daemon=True
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
                if kind == "ready":
                    ready.set()
                elif kind in ("sentence", "final"):
                    text = str(event.get("text") or "")
                    if event.get("rtf") is not None:
                        self._log(
                            f"sentence rtf={event.get('rtf')} "
                            f"audio_ms={event.get('audio_ms')}"
                        )
                    with self._lock:
                        routing = self._routing
                    if routing is not None and text:
                        try:
                            routing.on_sentence(text)
                        except Exception as exc:
                            self._log(f"sentence callback error: {exc}")
                # "partial" events are intentionally dropped: not used for
                # intent matching (and SenseVoice never emits them).
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

    def _read_stderr(self, asr: subprocess.Popen) -> None:
        # Diagnostics only — deliberately NOT marking the engine ready: the
        # model downloader prints here long before the model is loaded.
        try:
            for raw in asr.stderr:
                line = raw.decode("utf-8", errors="replace").strip()
                if line:
                    self._log(line)
        except Exception as exc:
            self._log(f"stderr reader error: {exc}")

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
        # Stop the capture first: the pipe engine then sees stdin EOF,
        # flushes its tail sentence and exits on its own, so the reader
        # threads drain cleanly.
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


class SensevoiceAsrProvider(AsrProvider):
    """Sentence-segmented ASR via the board-local SenseVoice pipe engine."""

    id = "asr_sensevoice"
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
        self._model_dir = _resolve_repo_path(runtime["model_dir"], "runtime.model_dir", root)
        self._engine: _AsrEngine | None = None
        self._engine_lock = threading.Lock()

    def _log(self, message: str) -> None:
        print(f"[asr_sensevoice] {message}", flush=True)

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
            "--vad",
            "--jsonl",
            "--model-dir",
            str(self._model_dir),
            "--language",
            str(runtime.get("language", "auto")),
            "--vad-thresh",
            str(int(vad.get("rms", 400))),
            "--pause",
            str(int(vad.get("pause_ms", 600)) / 1000.0),
            "--max-utt",
            str(int(vad.get("max_ms", 8000)) / 1000.0),
        ]
        # Core placement (mirrors the engine repo's run_mic_asr.sh model):
        #   X100 list (0-7)  -> taskset the whole process onto those cores
        #   A100 list (8-15, 1-2 cores) -> hand the EP a precise
        #       one-thread-per-core affinity via the environment (taskset
        #       cannot reach the A100 cluster); thread count = core count
        #   none             -> no pinning; EP threads float per core_arch
        threads = int(runtime.get("num_threads", 2))
        spawn_env: dict[str, str] | None = None
        affinity = str(runtime.get("cpu_affinity", "") or "").strip()
        if affinity and affinity != "none":
            cluster, cores = _parse_affinity(affinity)
            if cluster == "a100":
                spawn_env = {
                    "SPACEMIT_EP_INTRA_THREAD_AFFINITY": ";".join(str(core) for core in cores)
                }
                asr_argv += ["--core-arch", "a100", "--threads", str(len(cores))]
            else:
                asr_argv = ["taskset", "-c", affinity, *asr_argv]
                asr_argv += ["--core-arch", str(runtime.get("core_arch", "x100"))]
                asr_argv += ["--threads", str(threads)]
        else:
            asr_argv += ["--core-arch", str(runtime.get("core_arch", "x100"))]
            asr_argv += ["--threads", str(threads)]
        return _AsrEngine(
            capture_argv=capture_argv,
            asr_argv=asr_argv,
            working_dir=self._working_dir,
            grace_seconds=int(runtime.get("terminate_grace_seconds", 5)),
            start_timeout_seconds=float(runtime.get("start_timeout_seconds", 30)),
            on_log=self._log,
            spawn_env=spawn_env,
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
        if not (self._model_dir / "model_quant_optimized.onnx").is_file():
            problems.append(
                "model missing under runtime.model_dir; rebuild per "
                "asr/sensevoice/.gitignore"
            )
        with self._engine_lock:
            engine = self._engine
        running = engine is not None and engine.alive
        health = {"id": self.id, "type": self.type, "ok": not problems, "running": running}
        if problems:
            health["problems"] = problems
        return health
