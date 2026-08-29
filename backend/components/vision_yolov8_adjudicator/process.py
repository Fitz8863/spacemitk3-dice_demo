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

from .llm import OpenAICompatibleVisionVerifier, VerificationResult


class SnapshotError(ValueError):
    """Raised when a runtime snapshot reference is invalid or unsafe."""


class YoloRuntimeProcess:
    """Thin adapter for a resident YOLO process using JSONL stdin/stdout.

    Tests and deployments can inject a compatible runtime object into the
    provider; this class intentionally contains no game-specific logic.
    """

    def __init__(self, binary: str | Path | None = None, working_dir: str | Path | None = None) -> None:
        self.binary = str(binary or "yolov8_camera")
        self.working_dir = str(working_dir) if working_dir else None
        self._process: subprocess.Popen[str] | None = None

    def start(self, profile: Mapping[str, Any], view_id: str = "default", prewarm: bool = True) -> None:
        runtime = profile.get("runtime", {}) if isinstance(profile, Mapping) else {}
        binary = runtime.get("binary") if isinstance(runtime, Mapping) else None
        if binary: self.binary = str(binary)
        workdir = runtime.get("working_dir") if isinstance(runtime, Mapping) else None
        if workdir: self.working_dir = str(workdir)
        cmd = [self.binary, "--no-display"]
        if prewarm: cmd.append("--prewarm")
        self._process = subprocess.Popen(cmd, cwd=self.working_dir, stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, bufsize=1)

    def send(self, command: Mapping[str, Any]) -> None:
        if not self._process or not self._process.stdin:
            raise RuntimeError("YOLO runtime is not running")
        self._process.stdin.write(json.dumps(dict(command)) + "\n")
        self._process.stdin.flush()

    def events(self) -> Iterator[dict[str, Any]]:
        if not self._process or not self._process.stdout:
            return iter(())
        for line in self._process.stdout:
            try:
                event = json.loads(line)
            except (TypeError, ValueError):
                continue
            if isinstance(event, dict):
                yield event

    def stop(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()
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
