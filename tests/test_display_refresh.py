"""Display updates at game entry must not change detection ownership or controls."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is required")
@pytest.mark.parametrize("script", [
    "tests/test_display_layout.cjs",
    "tests/js/display_refresh_regressions.mjs",
])
def test_game_entry_uses_current_display_configuration(script):
    result = subprocess.run(
        ["node", str(ROOT / script), str(ROOT)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
