"""Run presentation behavior without a backend or robot connection."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which('node') is None, reason='Node is required')
def test_fixed_theme_migrates_old_preferences():
    result = subprocess.run(
        ['node', str(ROOT / 'tests/js/theme_regressions.cjs'), str(ROOT / 'web/theme.js')],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which('node') is None, reason='Node is required')
def test_hall_prefers_dice_without_starting_rounds_and_preserves_selection_keys():
    result = subprocess.run(
        ['node', str(ROOT / 'tests/js/hall_regressions.cjs'), str(ROOT / 'web/app.js')],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_removed_theme_has_no_selector_or_option():
    html = (ROOT / 'web/index.html').read_text()
    assert 'themeSelect' not in html
    assert 'storybook' not in html
    assert '萌趣游乐舱' not in html
    assert 'data-theme="dark-arena"' in html
