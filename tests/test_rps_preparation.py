"""Preparation completes before chant; throwing overlaps awaited chant."""
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from core.state_machine import GameRound
from test_state_machine import wait_for


def test_ready_motion_gates_chant_but_throw_does_not_wait_for_speech():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'backend/games/rps/manifest.json').read_text())
    prepared = threading.Event()
    throw_started = threading.Event()
    calls = []

    def robot(action, on_event, cancelled, log):
        calls.append(action['command'])
        if action['command'] == 'prepare_throw':
            prepared.wait(2)
        if action['command'] == 'throw_gesture':
            assert prepared.is_set()
            assert action['prepared'] is True
            throw_started.set()
        return {'status': 'completed', 'gesture': '石头'}

    round_ = GameRound(game_id='rps', manifest=manifest, robot_fn=robot,
                       adjudicate_fn=lambda *args: {}, log=lambda _: None)
    try:
        round_.start()
        round_.submit_intent('confirm')
        assert wait_for(lambda: 'prepare_throw' in calls)
        assert not any(e.get('event') == 'speech' and e.get('await')
                       for e in round_.snapshot()['events'])
        prepared.set()
        assert throw_started.wait(2)
        assert wait_for(lambda: any(e.get('event') == 'speech' and e.get('await')
                                   for e in round_.snapshot()['events']))
        assert round_.state == 'play'
        assert calls == ['prepare_throw', 'throw_gesture']
    finally:
        prepared.set()
        round_.cancel()


def test_display_rotation_does_not_change_human_inference_region():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'backend/games/rps/adjudicator_config.json').read_text())
    assert config['rotate']['enabled'] is False
    assert config['inference_rotate'] == {'enabled': True, 'direction': 'ccw', 'angle': 90}
    assert config['inference_roi'] == {'enabled': True, 'x': 0, 'y': 0.5, 'w': 1, 'h': 0.5}
    assert config['roi'] == {'enabled': True, 'x': 0, 'y': 0, 'w': 0.5, 'h': 1}
