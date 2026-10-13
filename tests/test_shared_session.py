"""Multi-client session tests; no providers, devices or audio are opened."""
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from core.shared_session import SharedSession
from core.errors import DiceArenaError


class Round:
    def __init__(self, game, sink):
        self.id = f'round-{game}'
        self.game_id = game
        self.state, self.status = 'rules', 'running'
        self.version = 1
        self.sink = sink
        self.inputs = []
        self.speech_active = False

    def snapshot(self):
        return {'round_id': self.id, 'game_id': self.game_id, 'state': self.state,
                'state_version': self.version, 'status': self.status, 'events': [], 'result': None}

    def submit_intent(self, intent, payload):
        self.inputs.append(intent)
        if intent == 'back':
            self.cancel()
        else:
            self.state = 'ready'
            self.version += 1
            self.sink(self, {'event': 'state_changed'})
        return self.snapshot()

    def cancel(self):
        self.status = 'cancelled'
        self.sink(self, {'event': 'round_complete'})


@pytest.fixture
def session():
    cfg = {'input_board_enable': True, 'standby': {'enabled': True, 'idle_seconds': 120, 'boot_standby': True}}
    created = []
    now = [0.0]
    def create(game, sink):
        result = Round(game, sink)
        created.append(result)
        return result
    s = SharedSession(config=lambda: cfg, games=lambda: ['dice', 'rps'], create_round=create,
                      clock=lambda: now[0], auto_start=False)
    yield s, cfg, created, now
    s.close()


def act(s, action, **payload):
    return s.submit({'action': action, **payload})


def test_same_selection_and_round_for_all_clients(session):
    s, _, made, _ = session
    act(s, 'select', game='rps')
    a = act(s, 'enter', request_id='A')
    b = act(s, 'enter', request_id='B')
    assert a['round_id'] == b['round_id'] == 'round-rps'
    assert len(made) == 1
    assert made[0].status == 'running'
    assert s.snapshot()['selected_game'] == 'rps'


def test_standby_confirm_respects_shared_cursor_and_exit_resets_once(session):
    s, _, made, _ = session
    act(s, 'navigate', direction='down')
    assert s.snapshot()['phase'] == 'select'
    assert s.snapshot()['selected_game'] == 'rps'
    act(s, 'enter')
    act(s, 'exit', round_id=made[0].id)
    assert s.snapshot()['phase'] == 'select'
    assert s.snapshot()['selected_game'] == 'dice'


def test_request_retry_and_late_state_cannot_advance_twice(session):
    s, _, made, _ = session
    act(s, 'enter')
    cmd = dict(round_id=made[0].id, state='rules', state_version=1, intent='confirm')
    act(s, 'intent', request_id='press-1', **cmd)
    act(s, 'intent', request_id='press-1', **cmd)
    with pytest.raises(DiceArenaError, match='stale'):
        act(s, 'intent', request_id='press-2', **cmd)
    assert made[0].inputs == ['confirm']


def test_invalid_selection_does_not_change_shared_cursor(session):
    s, _, _, _ = session
    with pytest.raises(DiceArenaError):
        act(s, 'select', game='disabled')
    assert s.snapshot()['selected_game'] == 'dice'


def test_presence_counts_and_45_second_grace(session):
    s, _, made, now = session
    s.connected(); s.connected()
    act(s, 'enter')
    s.disconnected()
    now[0] = 50
    s.tick()
    assert made[0].status == 'running'
    s.disconnected()
    now[0] = 94
    s.tick()
    assert made[0].status == 'running'
    s.connected()
    now[0] = 100
    s.tick()
    assert made[0].status == 'running'
    s.disconnected()
    now[0] = 146
    s.tick()
    assert made[0].status == 'cancelled'
    assert s.snapshot()['phase'] == 'select'


def test_global_activity_and_speech_reset_idle_timeout(session):
    s, _, made, now = session
    s.connected()
    act(s, 'enter')
    now[0] = 110
    act(s, 'activity')
    now[0] = 220
    s.tick()
    assert made[0].status == 'running'
    made[0].speech_active = True
    now[0] = 400
    s.tick()
    assert made[0].status == 'running'
    made[0].speech_active = False
    act(s, 'activity')
    now[0] = 521
    s.tick()
    assert made[0].status == 'cancelled'


def test_switch_applies_after_round_and_snapshot_restores_current_game(session):
    s, cfg, made, _ = session
    act(s, 'select', game='rps')
    act(s, 'enter')
    cfg['input_board_enable'] = False
    s.tick()
    assert s.snapshot()['shared'] is True
    assert s.snapshot()['round']['state'] == 'rules'
    act(s, 'exit', round_id=made[0].id)
    s.tick()
    assert s.snapshot()['shared'] is False


def test_worker_serializes_simultaneous_enter():
    made = []
    def create(game, sink):
        time.sleep(.02)
        r = Round(game, sink)
        made.append(r)
        return r
    s = SharedSession(config=lambda: {'input_board_enable': True}, games=lambda: ['dice'], create_round=create)
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda n: act(s, 'enter', request_id=str(n)), range(8)))
        assert len(made) == 1
        assert len({r['round_id'] for r in results}) == 1
    finally:
        s.close()
