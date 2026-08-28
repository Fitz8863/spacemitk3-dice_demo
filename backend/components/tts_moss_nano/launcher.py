"""Lifecycle hook for the local MOSS-TTS-Nano HTTP process."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from components.tts_moss_nano.settings import MossSettings, load_settings  # noqa: E402


PACKAGE_DIR = Path(__file__).resolve().parent
DAEMON = PACKAGE_DIR / "daemon.py"


def _paths(settings: MossSettings) -> tuple[Path, Path]:
    runtime_dir = Path(os.environ.get("DICE_RUNTIME_DIR", PROJECT_ROOT / ".runtime"))
    pid_file = Path(os.environ.get(
        "DICE_MOSS_TTS_PID_FILE", runtime_dir / f"moss-tts-{settings.port}.pid"
    ))
    log_file = Path(os.environ.get(
        "DICE_MOSS_TTS_LOG_FILE", runtime_dir / f"moss-tts-{settings.port}.log"
    ))
    return pid_file, log_file


def _cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]


def _is_expected(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        cwd = Path(f"/proc/{pid}/cwd").resolve()
        exe = Path(f"/proc/{pid}/exe").resolve()
    except (OSError, RuntimeError):
        return False
    command = _cmdline(pid)
    return (
        cwd == PROJECT_ROOT
        and exe.name.startswith("python")
        and any(Path(item).resolve() == DAEMON for item in command if item)
    )


def _discover() -> list[int]:
    found: list[int] = []
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(entry.name)
        except ValueError:
            continue
        if _is_expected(pid) and pid not in found:
            found.append(pid)
    return found


def _health(settings: MossSettings) -> dict[str, object] | None:
    try:
        with urllib.request.urlopen(f"{settings.base_url}/health", timeout=2) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def start() -> int:
    settings = load_settings()
    if not DAEMON.is_file():
        raise SystemExit(f"MOSS daemon is missing: {DAEMON}")
    if not settings.root.is_dir():
        raise SystemExit(f"MOSS-TTS root does not exist: {settings.root}")
    for directory in (settings.root / "python", settings.root / "src", settings.root / "lib"):
        if not directory.is_dir():
            raise SystemExit(f"MOSS-TTS runtime directory is missing: {directory}")
    model_manifest = settings.model_dir / "browser_poc_manifest.json"
    if not model_manifest.is_file():
        raise SystemExit(f"MOSS model manifest is missing: {model_manifest}")
    if settings.reference_audio is not None and not settings.reference_audio.is_file():
        raise SystemExit(f"MOSS reference audio is missing: {settings.reference_audio}")
    pid_file, log_file = _paths(settings)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    existing: list[int] = []
    if pid_file.is_file():
        try:
            candidate = int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            candidate = 0
        if _is_expected(candidate):
            existing.append(candidate)
        else:
            pid_file.unlink(missing_ok=True)
    existing.extend(pid for pid in _discover() if pid not in existing)
    if existing:
        health = _health(settings)
        if health and bool(health.get("ready")):
            pid_file.write_text(f"{existing[0]}\n", encoding="utf-8")
            print(f"MOSS-TTS bridge already running: pid={existing[0]} {settings.base_url}")
            return 0
        raise SystemExit(f"MOSS-TTS bridge process is not healthy; refusing to reuse it: {existing}")

    environment = os.environ.copy()
    python_bin = environment.get("DICE_PYTHON", sys.executable)
    python_entries = [str(settings.root / "python"), str(settings.root / "src")]
    if environment.get("PYTHONPATH"):
        python_entries.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = ":".join(python_entries)
    library_entries = [str(settings.root / "lib")]
    if environment.get("LD_LIBRARY_PATH"):
        library_entries.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = ":".join(library_entries)
    with log_file.open("ab") as output:
        process = subprocess.Popen(
            [python_bin, str(DAEMON)],
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_file.write_text(f"{process.pid}\n", encoding="utf-8")
    for _ in range(settings.start_timeout_seconds):
        if process.poll() is not None:
            pid_file.unlink(missing_ok=True)
            raise SystemExit(f"MOSS-TTS bridge exited during startup; see {log_file}")
        health = _health(settings)
        if health and bool(health.get("ready")):
            print(f"MOSS-TTS bridge started: pid={process.pid} {settings.base_url}")
            return 0
        time.sleep(1)
    raise SystemExit(f"MOSS-TTS bridge did not become ready within {settings.start_timeout_seconds}s; see {log_file}")


def stop() -> int:
    settings = load_settings()
    pid_file, _ = _paths(settings)
    candidates: list[int] = []
    if pid_file.is_file():
        try:
            candidate = int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            candidate = 0
        if _is_expected(candidate):
            candidates.append(candidate)
    candidates.extend(pid for pid in _discover() if pid not in candidates)
    if not candidates:
        pid_file.unlink(missing_ok=True)
        print("MOSS-TTS bridge is not running")
        return 0
    for pid in candidates:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and any(_is_expected(pid) for pid in candidates):
        time.sleep(0.1)
    for pid in candidates:
        if _is_expected(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    pid_file.unlink(missing_ok=True)
    print(f"MOSS-TTS bridge stopped: pid={' '.join(str(pid) for pid in candidates)}")
    return 0


def main() -> int:
    action = sys.argv[1] if len(sys.argv) > 1 else "start"
    if action == "start":
        return start()
    if action == "stop":
        return stop()
    raise SystemExit("usage: launcher.py {start|stop}")


if __name__ == "__main__":
    raise SystemExit(main())
