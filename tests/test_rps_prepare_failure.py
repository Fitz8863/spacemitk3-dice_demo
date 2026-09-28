"""Exercise real robot dispatch + round routing when preparation cannot move."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from games.rps import pipeline
from core.state_machine import GameRound
from test_rps_game import rps_manifest, wait_for


def dispatch(provider, slot):
    # Extract the production bridge without starting server hardware globals.
    tree = ast.parse((ROOT / 'backend/server.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_round_robot_fn')
    namespace = {'COMPONENTS': SimpleNamespace(require=lambda *a, **k: provider),
                 '_game_provider_id': lambda *a: 'robot_arm_nero',
                 'DiceArenaError': RuntimeError, 'time': time}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'robot_bridge', 'exec'), namespace)
    return namespace['_round_robot_fn']('rps', slot)


def test_prepare_timeout_routes_to_arm_diagnosis_without_chant_or_vision():
    provider = Mock()
    provider.prepare_throw.return_value = {'status': 'failed', 'reason': 'Controller endpoint did not reach target'}
    provider.reset_home.return_value = {'status': 'completed'}
    slot = {'status': 'completed', 'gesture': '布'}
    components = Mock()
    manifest = rps_manifest()

    def adjudicate(manifest, on_event, cancelled, log):
        return pipeline.run(log, cancelled, 1, components=components,
                            manifest=manifest, on_event=on_event, arm_throw=slot)

    round_ = GameRound(game_id='rps', manifest=manifest,
                       robot_fn=dispatch(provider, slot), adjudicate_fn=adjudicate,
                       log=lambda _: None)
    try:
        round_.start()
        round_.submit_intent('confirm')
        assert wait_for(lambda: round_.state == 'analysis_failed', timeout=3)
        result = round_.snapshot()['result']
        assert result['diagnosis']['reason'] == 'arm_throw_failed'
        assert 'Controller endpoint did not reach target' in result['diagnosis']['message']
        assert '绿色' in result['diagnosis']['message']
        assert slot['gesture'] is None
        components.require.assert_not_called()
        provider.throw_gesture.assert_not_called()
        assert not any(e.get('event') == 'speech' and e.get('await')
                       for e in round_.snapshot()['events'])
    finally:
        round_.cancel()


def test_throw_failure_during_observe_overrides_no_hand_diagnosis():
    slot = {'status': 'pending', 'gesture': None}
    provider = Mock()
    def observe(*a, **k):
        slot.update(status='failed', reason='出拳反馈超时')
        return {'diagnosed': True, 'diagnosis': {'reason': 'no_hand'}}
    provider.observe.side_effect = observe
    components = SimpleNamespace(require=lambda *a, **k: provider)
    outcome = pipeline.run(lambda _: None, lambda: False, 1, components=components,
                           manifest=rps_manifest(), on_event=lambda _: None, arm_throw=slot)
    assert outcome['diagnosis']['reason'] == 'arm_throw_failed'
    assert '出拳反馈超时' in outcome['diagnosis']['message']
