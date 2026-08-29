"""Small runtime adapters shared by the provider and snapshot tests.

The full resident-process orchestration is implemented in the provider task.
This module owns only the security-sensitive lifetime of runtime snapshots.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .llm import OpenAICompatibleVisionVerifier, VerificationResult


class SnapshotError(ValueError):
    """Raised when a runtime snapshot reference is invalid or unsafe."""


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
