"""Load the board-local environment file before any component reads config.

The backend runs as ``python3 backend/server.py`` from the repo root, so each
module resolves the root relative to its own location.  Components read runtime
settings (TTS URL, LLM key, timeouts) from ``os.environ``; this module ensures
``.dice-arena.env`` is loaded once, idempotently, before that read.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root (main/)
ENV_FILE = ROOT / ".dice-arena.env"


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE lines without overwriting the process env."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def load_board_env() -> None:
    """Load the repo ``.dice-arena.env`` once (idempotent)."""
    load_env_file(ENV_FILE)
