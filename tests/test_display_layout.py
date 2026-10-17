"""Screen orientation must not remap physical detections or result ownership."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from core.arena_config import ArenaConfigError, validate_arena_config, with_global_defaults
from core.games import public_game_manifest
from games.dice.result import project_participant_result
from games.rps.result import project as project_rps


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


@pytest.mark.parametrize('side', ['LEFT', 'RIGHT'])
@pytest.mark.parametrize('legacy', [False, True])
def test_display_is_the_only_orientation_setting_for_both_games(side, legacy):
    arena = {'schema_version': 1, 'display': {'player_side': side}}
    manifests = [{'id': 'dice'}, {'id': 'rps'}]
    if legacy:
        # A stale setting must not reverse role ownership in the new mode.
        arena['participants'] = {'player': 'RIGHT', 'agent': 'LEFT'}
        for manifest in manifests:
            manifest['participants'] = {'player': 'RIGHT', 'agent': 'LEFT'}
    before = copy.deepcopy((arena, manifests))
    merged = [with_global_defaults(m, validate_arena_config(arena)) for m in manifests]
    for game in merged:
        assert game['participants'] == {'player': 'LEFT', 'agent': 'RIGHT'}
        assert game['display']['player_side'] == side
        assert game['display']['video_rotation_deg'] == (180 if side == 'RIGHT' else 0)
    dice = project_participant_result({
        'winner': 'RIGHT', 'left_values': [5, 1, 3, 2, 5],
        'right_values': [5, 6, 1, 5, 2],
    }, merged[0]['participants'])
    assert (dice['player_score'], dice['agent_score'], dice['winner_role']) == (16, 19, 'AGENT')
    rps = project_rps('石头', '剪刀', merged[1]['participants'], source='test')
    assert (rps['player_choice'], rps['agent_choice'], rps['winner_role']) == ('石头', '剪刀', 'PLAYER')
    assert (arena, manifests) == before


def test_display_does_not_change_other_games_physical_mapping():
    arena = {'display': {'player_side': 'RIGHT'},
             'participants': {'player': 'RIGHT', 'agent': 'LEFT'}}
    merged = with_global_defaults({'id': 'other_game'}, arena)
    assert merged['participants'] == arena['participants']


@pytest.mark.parametrize('display', [False, {}, {'player_side': 'top'},
    {'player_side': 'LEFT', 'video_rotation_deg': []},
    {'player_side': 'LEFT', 'video_rotation_deg': {'rps': 45}},
    {'player_side': 'LEFT', 'video_rotation_deg': {'rps': True}},
    {'player_side': 'LEFT', 'video_rotation_deg': {'rps': '90'}}])
def test_invalid_display_rejected(display):
    with pytest.raises(ArenaConfigError):
        validate_arena_config({'schema_version': 1, 'display': display})
