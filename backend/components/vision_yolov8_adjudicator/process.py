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


def build_rtsp_args(config: Mapping[str, Any], selected_video: Mapping[str, Any] | None) -> list[str]:
    """Build one RTSP publisher argument set with a profile-owned path."""
    rtsp = config.get("rtsp", {}) if isinstance(config, Mapping) else {}
    if not isinstance(rtsp, Mapping) or not rtsp.get("enabled", False):
        return []
    args = ["--rtsp"]
    for option, key in (("--rtsp-host", "host"), ("--rtsp-port", "port")):
        value = rtsp.get(key)
        if value is not None:
            args.extend([option, str(value)])
    video_path = selected_video.get("path") if isinstance(selected_video, Mapping) else None
    if not isinstance(video_path, str) or not video_path.strip():
        video_path = rtsp.get("path")
    if video_path is not None:
        args.extend(["--rtsp-path", str(video_path).rstrip("/") or "/"])
    return args


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
        self._on_log: Any = lambda _line: None
        self._runtime_exit_emitted = False
        self._lifecycle_lock = threading.Lock()

    def start(
        self,
        profile: Mapping[str, Any],
        view_id: str = "default",
        prewarm: bool = True,
        *,
        snapshot_dir: str | Path | None = None,
        on_log: Any | None = None,
    ) -> None:
        """Start one resident runtime and wire its command/event pipes.

        ``snapshot_dir`` is supplied by the provider per adjudication job so
        that runtime evidence cannot escape into a shared ``/tmp`` directory.
        It is optional for backwards compatibility with manually launched
        runtimes; production provider calls always provide it.
        """
        self.stop()
        self._on_log = on_log if callable(on_log) else (lambda _line: None)
        self._runtime_exit_emitted = False
        runtime = profile.get("runtime", {}) if isinstance(profile, Mapping) else {}
        binary = runtime.get("binary") if isinstance(runtime, Mapping) else None
        if binary: self.binary = str(binary)
        workdir = runtime.get("working_dir") if isinstance(runtime, Mapping) else None
        if workdir: self.working_dir = str(workdir)
        # Component deployment defaults live next to this package.  A game
        # profile may override them for tests or a custom local runtime.
        component_config: Mapping[str, Any] = {}
        if not binary:
            try:
                from components.vision_yolov8_adjudicator.profile import load_component_config

                component = load_component_config(Path(__file__).parent)
                component_config = component
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
        if not component_config:
            try:
                from components.vision_yolov8_adjudicator.profile import load_component_config
                component_config = load_component_config(Path(__file__).parent)
            except Exception:
                component_config = {}

        # Game profiles own the model and camera semantics.  Forward only
        # those validated, non-secret values as command-line overrides; the
        # component config remains the deployment default.  Resolving the
        # model against the repository root is important because the C++
        # process runs with its own working directory.
        project_root = Path(__file__).resolve().parents[3]
        runtime_overrides: list[str] = []
        vision = profile.get("vision", {}) if isinstance(profile, Mapping) else {}
        if isinstance(vision, Mapping):
            model = vision.get("model")
            if isinstance(model, str) and model.strip():
                model_path = Path(model)
                if not model_path.is_absolute():
                    model_path = (project_root / model_path).resolve()
                runtime_overrides.extend(["--model", str(model_path)])
            stable_frames = vision.get("stable_frames")
            if isinstance(stable_frames, int) and not isinstance(stable_frames, bool):
                runtime_overrides.extend(["--stable-frames", str(stable_frames)])
            confidence = vision.get("confidence", vision.get("conf"))
            if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
                runtime_overrides.extend(["--conf", str(confidence)])
            divider_detection = vision.get("divider_detection")
            if isinstance(divider_detection, bool):
                runtime_overrides.append(
                    "--divider-detection" if divider_detection else "--no-divider-detection"
                )

        # A multi-view profile selects the camera per view.  Single-view
        # profiles intentionally inherit the component/C++ camera default.
        selected_view: Mapping[str, Any] | None = None
        multi = profile.get("multi_view", {}) if isinstance(profile, Mapping) else {}
        if isinstance(multi, Mapping) and multi.get("enabled"):
            for candidate in multi.get("views", []):
                if isinstance(candidate, Mapping) and str(candidate.get("id")) == str(view_id):
                    selected_view = candidate
                    break
        if selected_view is not None:
            camera = selected_view.get("camera")
            if isinstance(camera, str) and camera.strip():
                if camera.strip().isdigit():
                    runtime_overrides.extend(["--camera", camera.strip()])
                else:
                    runtime_overrides.extend(["--device", camera.strip()])
            selected_video = selected_view.get("video")
        else:
            selected_video = profile.get("video") if isinstance(profile, Mapping) else None

        control_read, control_write = os.pipe()
        event_read, event_write = os.pipe()
        cmd = [self.binary, "--no-display", "--control-fd", str(control_read),
               "--event-fd", str(event_write), "--view-id", str(view_id), *runtime_overrides]
        cmd.extend(build_rtsp_args(component_config, selected_video))
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
                    for line in stdout:
                        line = line.rstrip()
                        if line:
                            try:
                                self._on_log(line)
                            except Exception:
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
        process = self._process
        if not process or stream is None:
            return
        for line in stream:
            try:
                event = json.loads(line)
            except (TypeError, ValueError):
                continue
            if isinstance(event, dict):
                yield event
        returncode = process.poll()
        if returncode is not None and not self._runtime_exit_emitted:
            self._runtime_exit_emitted = True
            yield {"event": "runtime_exit", "returncode": returncode}

    def stop(self) -> None:
        # Terminate the child before closing the event stream.  A provider
        # collector may be blocked in ``TextIOWrapper`` while waiting for the
        # child's next event; closing that wrapper first can wait forever on
        # its read lock.  Child exit closes the inherited event writer and
        # releases the collector via EOF.
        with self._lifecycle_lock:
            process = self._process
            self._process = None
            if process is not None:
                try:
                    if process.poll() is None:
                        process.terminate()
                    process.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        process.kill()
                    except OSError:
                        pass
                    try:
                        process.wait(timeout=1)
                    except (OSError, subprocess.TimeoutExpired):
                        pass

            stream = self._event_stream
            self._event_stream = None
            if stream is not None:
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass
            for fd_name in ("_control_write", "_event_read"):
                fd = getattr(self, fd_name)
                setattr(self, fd_name, None)
                if fd is not None:
                    try:
                        os.close(fd)
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
