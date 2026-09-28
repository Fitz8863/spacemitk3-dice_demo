"""Screen orientation must not remap physical detections or result ownership."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from core.arena_config import ArenaConfigError, validate_arena_config, with_global_defaults
from core.games import public_game_manifest


@pytest.mark.parametrize('side,rotations', [('LEFT', [0, 270]), ('RIGHT', [180, 90])])
def test_both_games_share_display_without_changing_detection(side, rotations):
    arena = {'schema_version': 1, 'participants': {'player': 'LEFT', 'agent': 'RIGHT'},
             'display': {'player_side': side, 'video_rotation_deg': {'dice': 0, 'rps': 270}}}
    validate_arena_config(arena)
    before = copy.deepcopy(arena)
    for game, rotation in zip(('dice', 'rps'), rotations):
        manifest = {'id': game, 'vision_profile': {'inference_roi': {'y': 0.5}}}
        original = copy.deepcopy(manifest)
        merged = with_global_defaults(manifest, arena)
        assert merged['participants'] == arena['participants']
        assert merged['vision_profile'] == manifest['vision_profile']
        assert merged['display'] == {'player_side': side,
                                     'agent_side': 'RIGHT' if side == 'LEFT' else 'LEFT',
                                     'video_rotation_deg': rotation}
        assert public_game_manifest(merged)['display'] == merged['display']
        assert manifest == original
    assert arena == before


def test_absent_display_keeps_legacy_game_mapping():
    manifest = {'id': 'rps', 'participants': {'player': 'RIGHT', 'agent': 'LEFT'}}
    merged = with_global_defaults(manifest, {'participants': {'player': 'LEFT', 'agent': 'RIGHT'}})
    assert merged == manifest


@pytest.mark.parametrize('display', [False, {}, {'player_side': 'top'},
    {'player_side': 'LEFT', 'video_rotation_deg': []},
    {'player_side': 'LEFT', 'video_rotation_deg': {'rps': 45}},
    {'player_side': 'LEFT', 'video_rotation_deg': {'rps': True}},
    {'player_side': 'LEFT', 'video_rotation_deg': {'rps': '90'}}])
def test_invalid_display_rejected(display):
    with pytest.raises(ArenaConfigError):
        validate_arena_config({'schema_version': 1, 'display': display})
