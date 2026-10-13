"""One display session and one input queue for all board/web/ASR clients.

Providers stay outside this module. The injected round factory owns hardware;
all public snapshots are copies and no browser owns the round's lifetime.
"""
from __future__ import annotations

import copy
import queue
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import Future
from typing import Any, Callable

from core.errors import DiceArenaError, InvalidRequestError

_WAITING_STATES = {'rules', 'ready', 'result', 'analysis_failed', 'arm_failed'}


class SharedSession:
    def __init__(self, *, config: Callable, games: Callable, create_round: Callable,
                 on_hall: Callable | None = None, legacy_running: Callable | None = None,
                 board_source: Callable | None = None, clock: Callable = time.monotonic,
                 auto_start: bool = True) -> None:
        self._config, self._games, self._create = config, games, create_round
        self._on_hall = on_hall or (lambda phase: None)
        self._legacy_running = legacy_running or (lambda: False)
        self._board_source = board_source or (lambda: {})
        self._clock = clock
        self._condition = threading.Condition(threading.RLock())
        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._worker = None
        self._responses: OrderedDict = OrderedDict()
        self._id = uuid.uuid4().hex
        self._revision = 0
        self._round = None
        self._shared = bool(config().get('input_board_enable', False)) and not self._legacy_running()
        self._phase = 'standby' if self._standby().get('boot_standby', True) and self._standby().get('enabled', True) else 'select'
        self._selected = self._default_game()
        self._last_activity = clock()
        self._connections = 0
        self._detach_deadline = None
        self._last_error = ''
        self._hall_key = None
        self._last_board = None
        if auto_start:
            self._worker = threading.Thread(target=self._run, name='shared-session', daemon=True)
            self._worker.start()

    def _standby(self):
        return self._config().get('standby') or {}

    def _default_game(self):
        games = self._games()
        return 'dice' if 'dice' in games else next(iter(games), None)

    @property
    def closed(self):
        return self._stop.is_set()

    @property
    def enabled(self):
        with self._condition:
            return self._shared

    def owns(self, round_id):
        with self._condition:
            return self._round is not None and self._round.id == round_id

    def _snapshot_locked(self):
        return {'session_id': self._id, 'revision': self._revision, 'shared': self._shared,
                'phase': self._phase, 'selected_game': self._selected,
                'round_id': self._round.id if self._round else None,
                'game_id': self._round.game_id if self._round else None,
                'round': self._round.snapshot() if self._round else None,
                'board_input': self._board_source(), 'error': self._last_error,
                'server_time_ms': int(time.time() * 1000)}

    def snapshot(self):
        with self._condition:
            return self._snapshot_locked()

    def wait_for_update(self, revision, timeout=15):
        with self._condition:
            self._condition.wait_for(lambda: self._revision != revision or self._stop.is_set(), timeout)
            return self._snapshot_locked()

    def _changed(self):
        self._revision += 1
        self._condition.notify_all()

    def submit(self, command, *, wait=True):
        """Synchronous HTTP commands and fire-and-forget device callbacks share FIFO."""
        if self._stop.is_set():
            raise DiceArenaError('session is shutting down', 'SESSION_UNAVAILABLE', 503)
        future = Future()
        if self._worker is None:
            self._apply(dict(command), future)
        else:
            self._queue.put((dict(command), future))
        return future.result(timeout=60) if wait else None

    def notify(self, round_, event):
        # Called under the engine's lock. Never take the session lock here.
        self._queue.put(({'action': '_event', 'round': round_, 'event': dict(event)}, None))

    def connected(self):
        with self._condition:
            self._connections += 1
            self._detach_deadline = None

    def disconnected(self):
        with self._condition:
            self._connections = max(0, self._connections - 1)
            if not self._connections and self._detach_deadline is None:
                self._detach_deadline = self._clock() + 45

    def _apply(self, cmd, future):
        try:
            with self._condition:
                key = cmd.get('request_id')
                if key is not None and (not isinstance(key, str) or not key or len(key) > 200):
                    raise InvalidRequestError('request_id must be a non-empty string of at most 200 characters')
                if key and key in self._responses:
                    previous, result = self._responses[key]
                    if previous != cmd:
                        raise InvalidRequestError('request_id reused for a different action')
                    result = copy.deepcopy(result)
                else:
                    result = self._execute(cmd)
                    if key:
                        self._responses[key] = (cmd, copy.deepcopy(result))
                        while len(self._responses) > 256:
                            self._responses.popitem(last=False)
            if future:
                future.set_result(result)
        except Exception as exc:
            if future:
                future.set_exception(exc)
            else:
                print(f'[session] action failed: {exc}', flush=True)

    def _execute(self, cmd):
        action = cmd.get('action')
        if action == '_event':
            if self._round is cmd['round']:
                event = cmd['event']
                if event.get('event') in {'state_changed', 'audio_finished'}:
                    self._last_activity = self._clock()
                if event.get('event') == 'round_complete' and self._round.status != 'error':
                    self._return_hall()
                else:
                    self._changed()
            return None
        if not self._shared:
            raise DiceArenaError('shared session is disabled', 'SESSION_DISABLED', 409)
        if cmd.get('session_id') not in (None, self._id):
            raise DiceArenaError('stale session', 'SESSION_STALE', 409)
        if action == 'detach':
            return self._snapshot_locked()
        self._last_activity = self._clock()
        if action == 'activity':
            return self._snapshot_locked()
        if action == 'board':
            return self._board(cmd)
        if action == 'wake':
            if self._phase == 'standby':
                self._phase = 'select'
                self._changed()
        elif action in {'navigate', 'select'}:
            if self._round:
                raise DiceArenaError('selection is only available in the hall', 'SESSION_STALE', 409)
            games = self._games()
            if action == 'select':
                game = cmd.get('game')
                if game not in games:
                    raise InvalidRequestError('unknown or disabled game')
                self._selected = game
            elif games:
                if cmd.get('direction') not in {'up', 'down'}:
                    raise InvalidRequestError('direction must be up/down')
                index = games.index(self._selected) if self._selected in games else 0
                self._selected = games[(index + (1 if cmd['direction'] == 'down' else -1)) % len(games)]
            self._phase = 'select'
            self._changed()
        elif action == 'enter':
            if self._round and self._round.status == 'running':
                return self._snapshot_locked()
            game = cmd.get('game', self._selected)
            if game not in self._games():
                raise InvalidRequestError('unknown or disabled game')
            self._selected = game
            self._round = self._create(game, self.notify)
            self._phase, self._last_error = 'game', ''
            if not self._connections:
                self._detach_deadline = self._clock() + 45
            self._changed()
        elif action in {'intent', 'exit'}:
            round_ = self._round
            if round_ is None or cmd.get('round_id') != round_.id:
                raise DiceArenaError('stale round', 'SESSION_STALE', 409)
            snap = round_.snapshot()
            if cmd.get('state') not in (None, snap['state']) or cmd.get('state_version') not in (None, snap.get('state_version')):
                raise DiceArenaError('stale state', 'SESSION_STALE', 409)
            if action == 'exit':
                round_.cancel()
                self._return_hall()
            else:
                intent = cmd.get('intent')
                if not isinstance(intent, str) or not intent:
                    raise InvalidRequestError('intent is required')
                # Only the backend audio worker is allowed to release speech gates.
                if round_.status == 'error' and intent == 'new_round':
                    return self._execute({'action': 'enter'})
                if intent not in {'speech_started', 'speech_done'}:
                    round_.submit_intent(intent, cmd.get('payload') or {})
                    if round_.status in {'exited', 'cancelled'}:
                        self._return_hall()
                    else:
                        self._changed()
        else:
            raise InvalidRequestError('unknown session action')
        return self._snapshot_locked()

    def _board(self, cmd):
        # Capture context at key receipt, so buffered Enter cannot skip rules.
        current = self._round.id if self._round else None
        if cmd.get('round_id') != current:
            return self._snapshot_locked()
        key = cmd.get('key')
        if self._round:
            from core.board_input import resolve_intent
            if self._round.status == 'error':
                return self._execute({'action': 'enter' if key == 'confirm' else 'exit', 'round_id': current}) if key in {'confirm', 'back'} else self._snapshot_locked()
            intent = resolve_intent(self._round.game_id, self._round.state, key)
            if intent:
                return self._execute({'action': 'intent', 'round_id': current, 'intent': intent,
                                      'state': cmd.get('state'), 'state_version': cmd.get('state_version'),
                                      'payload': {'source': 'board'}})
        elif key in {'up', 'down'}:
            return self._execute({'action': 'navigate', 'direction': key})
        elif key == 'confirm':
            return self._execute({'action': 'enter'})
        elif self._phase == 'standby':
            return self._execute({'action': 'wake'})
        return self._snapshot_locked()

    def _return_hall(self):
        self._round = None
        self._phase = 'select'
        self._selected = self._default_game()
        self._last_activity = self._clock()
        self._detach_deadline = None
        self._changed()

    def tick(self):
        with self._condition:
            requested = bool(self._config().get('input_board_enable', False))
            if self._round is None and requested != self._shared and not self._legacy_running():
                self._shared = requested
                if not requested:
                    self._on_hall('disabled')
                self._hall_key = None
                self._changed()
            if not self._shared:
                return
            source = self._board_source()
            if source != self._last_board:
                self._last_board = dict(source)
                self._changed()
            now = self._clock()
            if self._round and self._round.status in {'cancelled', 'exited'}:
                self._return_hall()
            if self._round and not self._connections and self._detach_deadline is not None and now >= self._detach_deadline:
                self._round.cancel()
                self._return_hall()
            standby = self._standby()
            if standby.get('enabled', True) and now - self._last_activity >= float(standby.get('idle_seconds', 120)):
                if self._round and (self._round.state in _WAITING_STATES or self._round.status == 'error') and not self._round.speech_active:
                    self._round.cancel()
                    self._return_hall()
                elif self._phase == 'select':
                    self._phase = 'standby'
                    self._changed()
            hall_key = (self._phase, repr(self._config().get('game_select')), self._config().get('asr_enabled'))
            if self._round is None and self._hall_key != hall_key:
                self._on_hall(self._phase)
                self._hall_key = hall_key
            elif self._round:
                self._hall_key = None

    def _run(self):
        while not self._stop.is_set():
            try:
                cmd, future = self._queue.get(timeout=.2)
                self._apply(cmd, future)
            except queue.Empty:
                pass
            try:
                self.tick()
            except Exception as exc:
                print(f'[session] supervisor error: {exc}', flush=True)

    def close(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._worker and self._worker is not threading.current_thread():
            self._worker.join(timeout=2)
