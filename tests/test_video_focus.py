"""Exercise presentation transitions without starting a round or robot action."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which('node') is None, reason='Node is required')
def test_video_focus_preserves_stream_and_restores_normal_layout():
    result = subprocess.run(
        ['node', str(ROOT / 'tests/js/video_focus_regressions.cjs'), str(ROOT / 'web/video-focus.js')],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
