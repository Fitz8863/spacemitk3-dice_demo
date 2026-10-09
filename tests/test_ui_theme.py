"""Run presentation behavior without a backend or robot connection."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which('node') is None, reason='Node is required')
def test_theme_defaults_and_switching():
    result = subprocess.run(
        ['node', str(ROOT / 'tests/js/theme_regressions.cjs'), str(ROOT / 'web/theme.js')],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
