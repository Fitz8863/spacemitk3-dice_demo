"""Small runtime adapters shared by the provider and snapshot tests.

The full resident-process orchestration is implemented in the provider task.
This module owns only the security-sensitive lifetime of runtime snapshots.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence, Iterator
import json
import os
import subprocess
import threading

from components.vision_yolov8_adjudicator.llm import OpenAICompatibleVisionVerifier, VerificationResult


class SnapshotError(ValueError):
    """Raised when a runtime snapshot reference is invalid or unsafe."""


class YoloRuntimeProcess:
    """Adapter for a resident YOLO process using the vision-control-v1 pipes.

    The camera process deliberately keeps diagnostics on stdout/stderr while
    structured events and commands use dedicated inherited file descriptors.
    This prevents a verbose OpenCV/GStreamer log from corrupting the event
    protocol and, more importantly, lets a prewarmed process accept multiple
    adjudication rounds without being restarted.
    """

    def __init__(self, binary: str | Path | None = None, working_dir: str | Path | None = None) -> None:
        self.binary = str(binary or "yolov8_camera")
        self.working_dir = str(working_dir) if working_dir else None
        self._process: subprocess.Popen[str] | None = None
        self._control_write: int | None = None
        self._event_stream: Any | None = None
        self._event_read: int | None = None
        self._stdout_thread: threading.Thread | None = None

    def start(
        self,
        profile: Mapping[str, Any],
        view_id: str = "default",
        prewarm: bool = True,
        *,
        snapshot_dir: str | Path | None = None,
    ) -> None:
        """Start one resident runtime and wire its command/event pipes.

        ``snapshot_dir`` is supplied by the provider per adjudication job so
        that runtime evidence cannot escape into a shared ``/tmp`` directory.
        It is optional for backwards compatibility with manually launched
        runtimes; production provider calls always provide it.
        """
        self.stop()
        runtime = profile.get("runtime", {}) if isinstance(profile, Mapping) else {}
        binary = runtime.get("binary") if isinstance(runtime, Mapping) else None
        if binary: self.binary = str(binary)
        workdir = runtime.get("working_dir") if isinstance(runtime, Mapping) else None
        if workdir: self.working_dir = str(workdir)
        # Component deployment defaults live next to this package.  A game
        # profile may override them for tests or a custom local runtime.
        if not binary:
            try:
                from components.vision_yolov8_adjudicator.profile import load_component_config

                component = load_component_config(Path(__file__).parent)
                defaults = component.get("runtime", {})
                if not self.binary or self.binary == "yolov8_camera":
                    configured = defaults.get("binary") if isinstance(defaults, Mapping) else None
                    if configured:
                        root = Path(__file__).resolve().parents[3]
                        self.binary = str((root / str(configured)).resolve())
                if not self.working_dir and isinstance(defaults, Mapping) and defaults.get("working_dir"):
                    root = Path(__file__).resolve().parents[3]
                    self.working_dir = str((root / str(defaults["working_dir"])).resolve())
            except Exception:
                # A fake/injected runtime binary need not have a component
                # config.  Let subprocess report its normal launch error.
                pass

        control_read, control_write = os.pipe()
        event_read, event_write = os.pipe()
        cmd = [self.binary, "--no-display", "--control-fd", str(control_read),
               "--event-fd", str(event_write), "--view-id", str(view_id)]
        if prewarm: cmd.append("--prewarm")
        if snapshot_dir is not None:
            Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
            cmd.extend(["--snapshot-dir", str(Path(snapshot_dir).resolve())])
        try:
            self._process = subprocess.Popen(
                cmd,
                cwd=self.working_dir,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                pass_fds=(control_read, event_write),
            )
        except Exception:
            for fd in (control_read, control_write, event_read, event_write):
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise
        # Parent only writes commands and reads events.  Child inherited the
        # opposite ends through pass_fds; close those ends immediately here.
        os.close(control_read)
        os.close(event_write)
        self._control_write = control_write
        self._event_read = event_read
        self._event_stream = os.fdopen(event_read, "r", encoding="utf-8", buffering=1)
        self._event_read = None

        # Drain diagnostics independently so the child cannot block on a
        # full stdout pipe while the provider is waiting for event-fd data.
        stdout = self._process.stdout
        if stdout is not None:
            def drain() -> None:
                try:
                    for _line in stdout:
                        pass
                except (OSError, ValueError):
                    pass

            self._stdout_thread = threading.Thread(target=drain, name="yolo-runtime-log", daemon=True)
            self._stdout_thread.start()

    def send(self, command: Mapping[str, Any]) -> None:
        if not self._process or self._process.poll() is not None or self._control_write is None:
            raise RuntimeError("YOLO runtime is not running")
        payload = (json.dumps(dict(command), ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            os.write(self._control_write, payload)
        except OSError as exc:
            raise RuntimeError("YOLO runtime control channel is closed") from exc

    def events(self) -> Iterator[dict[str, Any]]:
        stream = self._event_stream
        if not self._process or stream is None:
            return
        for line in stream:
            try:
                event = json.loads(line)
            except (TypeError, ValueError):
                continue
            if isinstance(event, dict):
                yield event

    def stop(self) -> None:
        process = self._process
        self._process = None
        for stream_name in ("_event_stream",):
            stream = getattr(self, stream_name)
            setattr(self, stream_name, None)
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        for fd_name in ("_control_write", "_event_read"):
            fd = getattr(self, fd_name)
            setattr(self, fd_name, None)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        if process is None:
            return
        try:
            process.terminate()
            process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError:
                pass


def _snapshot_path(observation: Mapping[str, Any], task_dir: Path) -> Path:
    snapshot = observation.get("snapshot")
    if not isinstance(snapshot, Mapping):
        raise SnapshotError("observation.snapshot must be an object")
    raw = snapshot.get("path")
    if not isinstance(raw, str) or not raw.strip():
        raise SnapshotError("snapshot.path must be a non-empty absolute path")
    candidate = Path(raw)
    if not candidate.is_absolute():
        raise SnapshotError("snapshot.path must be an absolute path")
    root = Path(task_dir).resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SnapshotError("snapshot.path must stay inside task directory") from exc
    if resolved.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise SnapshotError("snapshot.path must reference a JPEG or PNG file")
    if not resolved.is_file():
        raise SnapshotError("snapshot file does not exist")
    return resolved


def verify_snapshot(
    observation: Mapping[str, Any],
    *,
    task_dir: Path,
    verifier: OpenAICompatibleVisionVerifier,
    system_prompt: str,
    user_prompt: str,
    allowed_outcomes: Sequence[str],
    timeout_seconds: float,
    model: str | None = None,
) -> VerificationResult:
    """Read one stable snapshot, verify it, and always remove the file.

    Runtime-created snapshots are single-use evidence.  Validation occurs
    before reading, and cleanup is restricted to the resolved task directory.
    """

    path = _snapshot_path(observation, Path(task_dir))
    try:
        # Read before invoking the verifier so missing/invalid files fail at a
        # deterministic boundary.  The verifier receives the path to preserve
        # its injectable transport and existing API.
        path.read_bytes()
        return verifier.verify(
            image_path=path,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            allowed_outcomes=allowed_outcomes,
            timeout_seconds=timeout_seconds,
            model=model,
        )
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # Cleanup failure must not expose an arbitrary path or mask the
            # verifier result; callers can report the leaked snapshot.
            pass
