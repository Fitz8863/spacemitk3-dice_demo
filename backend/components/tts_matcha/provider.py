"""Dice Arena adapter for the board-local Matcha-TTS service subprocess.

The engine is ``tts/matcha-tts/build-cpp/matcha_tts_service``: one resident
child speaking a line protocol on stdio (see ``tts/matcha-tts/README.md``).
The provider owns the child's whole lifecycle — it spawns on first use (or on
the backend's startup prewarm), serializes one request at a time, restarts a
crashed child on the next request, and stops it on backend shutdown.  There
are no lifecycle scripts and no HTTP hop.

Local-engine contract: the arena config pins exactly one local TTS engine per
process run, and that engine must be warm before the backend serves.  Hence
``prewarm()`` (called from server startup, blocking, fatal on failure) and
``shutdown()`` (called from the backend's runtime shutdown seam).
"""
from __future__ import annotations

import base64
import json
import os
import queue
import subprocess
import threading
import time
import uuid
import wave
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from core.errors import TtsServiceError, TtsValidationError
from core.tts import TtsProvider
from core.tts_config import TtsConfigError, load_component_config

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_STDERR_TAIL_LINES = 30


class MatchaConfigError(ValueError):
    """Raised when tts_matcha/config.json is malformed."""


def _resolve_repo_path(value: Any, field: str, project_root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise MatchaConfigError(f"{field} must be a non-empty repository-relative string")
    raw = Path(value)
    if raw.is_absolute() or any(part in {"", ".", ".."} for part in raw.parts):
        raise MatchaConfigError(f"{field} must stay inside the project")
    root = project_root.resolve()
    candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise MatchaConfigError(f"{field} escapes the project") from exc
    return candidate


def _positive_number(value: Any, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value <= 0
    ):
        raise MatchaConfigError(f"{field} must be a positive number")
    return float(value)


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise MatchaConfigError(f"{field} must be an integer >= 1")
    return value


def _validate_affinity(value: Any, expected_threads: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MatchaConfigError("runtime.ep_affinity must be a non-empty string like '8;9'")
    cores = [part.strip() for part in value.split(";")]
    if (
        not cores
        or any(not core.isdigit() for core in cores)
        or len(cores) != len(set(cores))
        or len(cores) != expected_threads
    ):
        raise MatchaConfigError(
            "runtime.ep_affinity must be a semicolon-separated unique core list "
            f"with exactly runtime.ep_threads ({expected_threads}) entries"
        )
    return ";".join(cores)


def load_config(package_dir: Path, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Load and validate tts_matcha/config.json.

    Only path *syntax* is validated here so a development checkout without
    the board assets still registers the component; ``health()`` reports the
    missing files instead.
    """
    try:
        config = load_component_config(package_dir)
    except TtsConfigError as exc:
        raise MatchaConfigError(str(exc)) from exc
    if config.get("schema_version") != 1:
        raise MatchaConfigError("config schema_version must be 1")
    runtime = config.get("runtime", {})
    if runtime.get("kind", "local") != "local":
        raise MatchaConfigError("tts_matcha runtime.kind must be local (subprocess engine)")

    root = _resolve_repo_path(runtime.get("root"), "runtime.root", project_root)

    def root_relative(value: Any, field: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise MatchaConfigError(f"{field} must be a non-empty root-relative string")
        raw = Path(value)
        if raw.is_absolute() or any(part in {"", ".", ".."} for part in raw.parts):
            raise MatchaConfigError(f"{field} must stay inside runtime.root")
        candidate = (root / raw).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise MatchaConfigError(f"{field} escapes runtime.root") from exc
        return candidate

    root_relative(runtime.get("binary"), "runtime.binary")
    root_relative(runtime.get("model_dir"), "runtime.model_dir")
    root_relative(runtime.get("sherpa_lib_dir"), "runtime.sherpa_lib_dir")

    ep_threads = _positive_int(runtime.get("ep_threads", 2), "runtime.ep_threads")
    if not isinstance(runtime.get("enable_affinity", True), bool):
        raise MatchaConfigError("runtime.enable_affinity must be boolean")
    if runtime.get("enable_affinity", True):
        _validate_affinity(runtime.get("ep_affinity", "8;9"), ep_threads)

    _positive_number(runtime.get("start_timeout_seconds", 60), "runtime.start_timeout_seconds")
    _positive_number(runtime.get("terminate_grace_seconds", 5), "runtime.terminate_grace_seconds")
    _positive_number(runtime.get("request_timeout_seconds", 120), "runtime.request_timeout_seconds")

    startup = config.get("startup", {})
    warmup_text = startup.get("warmup_text", "你好。")
    if not isinstance(warmup_text, str) or not warmup_text.strip():
        raise MatchaConfigError("startup.warmup_text must be a non-empty string")

    generation = config.get("generation", {})
    chunk_target = _positive_int(generation.get("chunk_target", 40), "generation.chunk_target")
    chunk_max = _positive_int(generation.get("chunk_max", 90), "generation.chunk_max")
    if chunk_target > chunk_max:
        raise MatchaConfigError("generation.chunk_target cannot exceed generation.chunk_max")

    voice = config.get("voice", {})
    if voice.get("speaker_id", 0) != 0:
        # matcha-icefall-zh-en is a single-speaker model; sid 0 is the only
        # speaker its metadata exposes.  The voice parameter stays addressable
        # (voice "0") so a multi-speaker model can slot in later.
        raise MatchaConfigError("voice.speaker_id must be 0 (single-speaker model)")
    return config


def _sanitize_text(text: str) -> str:
    """Make text safe for the tab-separated request protocol."""
    return " ".join(text.split())


def merge_wav_frames(frames: list[bytes]) -> bytes:
    """Concatenate same-format WAV frames into one WAV."""
    if not frames:
        raise TtsServiceError("tts_matcha produced no audio frames")
    payload = bytearray()
    params = None
    for index, frame in enumerate(frames):
        try:
            with wave.open(BytesIO(frame)) as reader:
                item = reader.getparams()
                payload.extend(reader.readframes(reader.getnframes()))
        except (wave.Error, EOFError) as exc:
            raise TtsServiceError(f"tts_matcha frame {index} is not a valid WAV") from exc
        if params is None:
            params = item
        elif (item.nchannels, item.sampwidth, item.framerate) != (
            params.nchannels, params.sampwidth, params.framerate
        ):
            raise TtsServiceError("tts_matcha frames disagree on audio format")
    assert params is not None
    buffer = BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(params.nchannels)
        writer.setsampwidth(params.sampwidth)
        writer.setframerate(params.framerate)
        writer.writeframes(bytes(payload))
    return buffer.getvalue()


class _MatchaService:
    """One resident ``matcha_tts_service`` subprocess plus reader threads."""

    def __init__(
        self,
        *,
        argv: list[str],
        env: dict[str, str],
        cwd: Path,
        ready_timeout_seconds: float,
        on_log: Callable[[str], None],
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    ) -> None:
        self._on_log = on_log
        self._ready_timeout = ready_timeout_seconds
        self.ready_detail: dict[str, Any] = {}
        self._events: queue.Queue[dict[str, Any]] = queue.Queue()
        self._ready = threading.Event()
        self._stderr_tail: list[str] = []
        self._request_lock = threading.Lock()
        self._process = popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        self._reader = threading.Thread(
            target=self._read_events, name="tts-matcha-events", daemon=True
        )
        self._stderr_reader = threading.Thread(
            target=self._read_stderr, name="tts-matcha-stderr", daemon=True
        )
        self._reader.start()
        self._stderr_reader.start()

    @property
    def alive(self) -> bool:
        return self._process.poll() is None

    @property
    def returncode(self) -> int | None:
        return self._process.poll()

    @property
    def stderr_tail(self) -> str:
        return " | ".join(self._stderr_tail[-_STDERR_TAIL_LINES:])

    def wait_ready(self) -> dict[str, Any]:
        """Block until the engine warmed up (or failed/timeout)."""
        if not self._ready.wait(self._ready_timeout):
            raise TtsServiceError(
                f"tts_matcha service did not become ready within "
                f"{self._ready_timeout:g}s (load+warmup too slow or stuck); "
                f"stderr: {self.stderr_tail}"
            )
        if not self.ready_detail:
            raise TtsServiceError(
                f"tts_matcha service exited during startup "
                f"(rc={self._process.poll()}); stderr: {self.stderr_tail}"
            )
        return dict(self.ready_detail)

    def request(
        self,
        text: str,
        speed: float,
        write_frame: Callable[[bytes], None],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Run one synthesis request; stream sentence frames through."""
        request_id = uuid.uuid4().hex[:12]
        with self._request_lock:
            if not self.alive:
                raise TtsServiceError(
                    f"tts_matcha service is not running; stderr: {self.stderr_tail}"
                )
            assert self._process.stdin is not None
            try:
                self._process.stdin.write(f"{request_id}\t{speed:g}\t{text}\n")
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise TtsServiceError(
                    f"tts_matcha service pipe broke: {exc}; stderr: {self.stderr_tail}"
                ) from exc

            deadline = time.monotonic() + timeout_seconds
            frames = 0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TtsServiceError(
                        f"tts_matcha request timed out after {timeout_seconds:g}s"
                    )
                try:
                    event = self._events.get(timeout=min(remaining, 1.0))
                except queue.Empty:
                    if not self.alive:
                        raise TtsServiceError(
                            "tts_matcha service exited mid-request "
                            f"(rc={self._process.poll()}); stderr: {self.stderr_tail}"
                        ) from None
                    continue

                kind = event.get("event")
                if event.get("id") not in (request_id, ""):
                    continue  # stray line from a superseded request
                if kind == "audio":
                    audio = base64.b64decode(str(event.get("wav_b64") or ""))
                    if not audio.startswith(b"RIFF"):
                        raise TtsServiceError("tts_matcha audio frame is not a WAV")
                    write_frame(audio)
                    frames += 1
                elif kind == "done":
                    outcome = dict(event)
                    outcome["frames"] = frames
                    return outcome
                elif kind == "error":
                    raise TtsServiceError(
                        f"tts_matcha synthesis failed: {event.get('message')}"
                    )
                # "sentence" events are diagnostics only.

    def stop(self, grace_seconds: float) -> None:
        """EOF the child and reap it; escalate to terminate/kill if needed."""
        try:
            if self._process.stdin is not None:
                self._process.stdin.close()
        except OSError:
            pass
        try:
            self._process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            try:
                self._process.terminate()
                self._process.wait(timeout=2)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    self._process.kill()
                except OSError:
                    pass
        for stream in (self._process.stdout, self._process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        self._reader.join(timeout=5)
        self._stderr_reader.join(timeout=5)

    # ---- reader threads -------------------------------------------------

    def _read_events(self) -> None:
        assert self._process.stdout is not None
        try:
            for raw in self._process.stdout:
                line = raw.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    self._on_log(f"tts_matcha non-JSON output ignored: {line[:200]}")
                    continue
                if not isinstance(event, dict):
                    continue
                if event.get("event") == "ready":
                    self.ready_detail = event
                    self._on_log(f"tts_matcha ready: {line}")
                    self._ready.set()
                self._events.put(event)
        except Exception as exc:  # diagnostics only; requesters see process death
            self._on_log(f"tts_matcha event reader error: {exc}")
        finally:
            # Unblocks wait_ready; a process that died before "ready" is
            # reported through the empty ready_detail.
            self._ready.set()

    def _read_stderr(self) -> None:
        assert self._process.stderr is not None
        try:
            for raw in self._process.stderr:
                line = raw.rstrip()
                if not line:
                    continue
                self._stderr_tail.append(line)
                if len(self._stderr_tail) > _STDERR_TAIL_LINES * 4:
                    del self._stderr_tail[: _STDERR_TAIL_LINES]
                self._on_log(f"tts_matcha stderr: {line}")
        except Exception as exc:
            self._on_log(f"tts_matcha stderr reader error: {exc}")


class TtsMatcha(TtsProvider):
    """Adapt the resident Matcha service subprocess to the TTS seam."""

    id = "tts_matcha"
    type = "tts"
    name = "Matcha-TTS (Sherpa-ONNX, SpacemiT EP)"
    version = "1.0"

    def __init__(
        self,
        manifest: dict[str, Any] | None = None,
        *,
        project_root: Path | None = None,
        package_dir: Path | None = None,
    ) -> None:
        super().__init__(manifest)
        self._root = (project_root or PROJECT_ROOT).resolve()
        # ``package_dir`` lets tests point the provider at a fixture config;
        # the registry always instantiates with the real package directory.
        self._config = load_config(package_dir or Path(__file__).parent, project_root=self._root)
        runtime = self._config["runtime"]
        engine_root = _resolve_repo_path(runtime["root"], "runtime.root", self._root)

        self._binary = engine_root / str(runtime["binary"])
        self._model_dir = engine_root / str(runtime["model_dir"])
        self._sherpa_lib_dir = engine_root / str(runtime["sherpa_lib_dir"])
        self._ep_threads = int(runtime.get("ep_threads", 2))
        self._ep_affinity = str(runtime.get("ep_affinity", "8;9"))
        self._enable_affinity = bool(runtime.get("enable_affinity", True))
        self._start_timeout = float(runtime.get("start_timeout_seconds", 60))
        self._grace = float(runtime.get("terminate_grace_seconds", 5))
        self._request_timeout = float(runtime.get("request_timeout_seconds", 120))
        self._warmup_text = str(self._config.get("startup", {}).get("warmup_text", "你好。"))
        generation = self._config.get("generation", {})
        self._chunk_target = int(generation.get("chunk_target", 40))
        self._chunk_max = int(generation.get("chunk_max", 90))
        self._lock = threading.Lock()
        self._service: _MatchaService | None = None

    # ---- lifecycle ------------------------------------------------------

    def _service_argv(self) -> list[str]:
        argv = [
            str(self._binary),
            "--model-dir", str(self._model_dir),
            "--ep-threads", str(self._ep_threads),
            "--warmup-text", self._warmup_text,
            "--chunk-target", str(self._chunk_target),
            "--chunk-max", str(self._chunk_max),
        ]
        if self._enable_affinity:
            argv += ["--ep-affinity", self._ep_affinity]
        else:
            argv.append("--no-affinity")
        return argv

    def _service_env(self) -> dict[str, str]:
        env = dict(os.environ)
        lib = str(self._sherpa_lib_dir)
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{lib}:{existing}" if existing else lib
        return env

    def _ensure_started(self) -> _MatchaService:
        """Return a live service, (re)spawning when absent or crashed."""
        with self._lock:
            service = self._service
            if service is not None and service.alive:
                return service
            if service is not None:
                self._log(f"tts_matcha service died (rc={service.returncode}); restarting")
                service.stop(self._grace)
            try:
                service = _MatchaService(
                    argv=self._service_argv(),
                    env=self._service_env(),
                    cwd=self._binary.parent,
                    ready_timeout_seconds=self._start_timeout,
                    on_log=self._log,
                )
            except OSError as exc:
                raise TtsServiceError(
                    f"failed to start tts_matcha service ({self._binary}): {exc}"
                ) from exc
            self._service = service
            # Startup (load + mandatory warmup) is part of the local-engine
            # contract, so wait for readiness here; a failure raises.
            service.wait_ready()
            return service

    def prewarm(self) -> None:
        """Start and warm the engine (server startup calls this, blocking)."""
        self._ensure_started()

    def shutdown(self) -> None:
        with self._lock:
            service = self._service
            self._service = None
        if service is not None:
            service.stop(self._grace)

    def _log(self, message: str) -> None:
        print(f"[tts_matcha] {message}", flush=True)

    # ---- health ---------------------------------------------------------

    def health(self) -> dict[str, Any]:
        problems: list[str] = []
        if not self._binary.is_file():
            problems.append(f"binary missing: {self._binary} (build per tts/matcha-tts/README.md)")
        elif not self._binary.stat().st_mode & 0o111:
            problems.append(f"binary not executable: {self._binary}")
        if not self._model_dir.is_dir():
            problems.append(f"model dir missing: {self._model_dir} (board asset)")
        if not (self._sherpa_lib_dir / "libsherpa-onnx-c-api.so").is_file():
            problems.append(f"sherpa runtime missing: {self._sherpa_lib_dir} (board asset)")

        with self._lock:
            service = self._service
        running = service is not None and service.alive
        health: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "ok": not problems and running,
            "engine": "matcha-icefall-zh-en (sherpa-onnx, spacemit EP)",
            "voice": "0",
            "running": running,
            "supports_voice_clone": False,
            "supports_speed": True,
            "supports_stream": True,
        }
        if problems:
            health["problems"] = problems
            return health
        if running and service is not None and service.ready_detail:
            health["sample_rate"] = service.ready_detail.get("sample_rate")
            health["ep_threads"] = service.ready_detail.get("ep_threads")
            health["ep_affinity"] = service.ready_detail.get("ep_affinity")
        return health

    # ---- synthesis ------------------------------------------------------

    def _validate_payload(self, payload: dict[str, Any]) -> tuple[str, float]:
        text, voice, speed = self.validate(payload)
        # The model has exactly one speaker; the voice parameter stays
        # addressable as its id ("0"), with "default" as the friendly alias.
        if voice not in ("default", "0"):
            raise TtsValidationError(
                "tts_matcha is a single-speaker model; only voice '0' (or 'default') is supported"
            )
        sanitized = _sanitize_text(text)
        if not sanitized:
            raise TtsValidationError("text is required")
        return sanitized, speed

    def synthesize(self, payload: dict[str, Any]) -> tuple[bytes, dict[str, str]]:
        text, speed = self._validate_payload(payload)
        frames: list[bytes] = []
        summary = self._run_request(text, speed, frames.append)
        return merge_wav_frames(frames), {
            "X-Dice-TTS-Frames": str(summary.get("frames", len(frames))),
        }

    def stream(self, payload: dict[str, Any], write_frame: Callable[[bytes], None]) -> None:
        text, speed = self._validate_payload(payload)
        self._run_request(text, speed, write_frame)

    def _run_request(
        self,
        text: str,
        speed: float,
        write_frame: Callable[[bytes], None],
    ) -> dict[str, Any]:
        service = self._ensure_started()
        outcome = service.request(text, speed, write_frame, self._request_timeout)
        self._log(
            f"synthesized sentences={outcome.get('sentences')} "
            f"audio={outcome.get('audio_seconds')}s in {outcome.get('elapsed_seconds')}s"
        )
        return outcome
