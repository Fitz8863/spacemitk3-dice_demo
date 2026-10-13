"""Single-production speech fanout and cancellable default-sink board playback.

The engine's notification path only enqueues events. Synthesis, disk IO and
PipeWire run on separate workers, never on the input queue or engine lock.
"""
from __future__ import annotations

import io
import os
import queue
import select
import subprocess
import threading
import time
import wave
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

from core.tts_protocol import validate_wav


def wav_pcm(audio):
    validate_wav(audio)
    with wave.open(io.BytesIO(audio), 'rb') as wav:
        if wav.getcomptype() != 'NONE':
            raise ValueError('only uncompressed PCM WAV is supported')
        channels, width, rate = wav.getnchannels(), wav.getsampwidth(), wav.getframerate()
        if width not in (1, 2, 3, 4) or channels < 1 or rate < 1:
            raise ValueError('unsupported WAV format')
        return (channels, width, rate), wav.readframes(wav.getnframes())


def pcm_wav(fmt, data):
    out = io.BytesIO()
    with wave.open(out, 'wb') as wav:
        wav.setnchannels(fmt[0]); wav.setsampwidth(fmt[1]); wav.setframerate(fmt[2])
        wav.writeframes(data)
    return out.getvalue()


class PipeWirePlayer:
    """One raw PCM pipe per directive; default PipeWire sink stays user-owned."""
    def __init__(self, fmt):
        formats = {1: 'u8', 2: 's16', 3: 's24', 4: 's32'}
        env = dict(os.environ)
        env.setdefault('XDG_RUNTIME_DIR', f'/run/user/{os.getuid()}')
        self._process = subprocess.Popen(
            ['pw-cat', '--playback', '--raw', '--rate', str(fmt[2]), '--channels', str(fmt[0]),
             '--format', formats[fmt[1]], '--latency', '100ms', '-'],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)
        os.set_blocking(self._process.stdin.fileno(), False)

    def write(self, data, cancelled):
        view = memoryview(data)
        deadline = time.monotonic() + 10
        while view:
            if cancelled():
                raise InterruptedError('playback cancelled')
            if self._process.poll() is not None:
                raise OSError('default-sink player exited before playback finished')
            if time.monotonic() >= deadline:
                raise TimeoutError('default-sink player stopped accepting audio')
            _, ready, _ = select.select([], [self._process.stdin], [], .05)
            if ready:
                try:
                    count = os.write(self._process.stdin.fileno(), view[:65536])
                except BlockingIOError:
                    continue
                view = view[count:]
                deadline = time.monotonic() + 10

    def finish(self, cancelled, deadline):
        self._process.stdin.close()
        while self._process.poll() is None:
            if cancelled():
                raise InterruptedError('playback cancelled')
            if time.monotonic() > deadline:
                raise TimeoutError('default-sink playback drain timed out')
            time.sleep(.03)
        if self._process.returncode:
            reason = self._process.stderr.read(2000).decode(errors='replace')
            raise OSError(f'PipeWire playback failed: {reason.strip()}')

    def close(self):
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._process.kill(); self._process.wait(timeout=1)
        for stream in (self._process.stdin, self._process.stderr):
            try:
                stream.close()
            except OSError:
                pass


@dataclass
class SpeechClip:
    round: Any
    directive: dict
    frames: list = field(default_factory=list)  # (start_seconds, end_seconds, fmt, pcm)
    condition: Any = field(default_factory=threading.Condition)
    cancelled: Any = field(default_factory=threading.Event)
    done: bool = False
    error: str = ''
    duration: float = 0
    started: float | None = None
    finished: bool = False
    board_error: str = ''
    bytes: int = 0
    expires_at: float = 0


class SharedAudio:
    def __init__(self, *, produce: Callable, player_factory: Callable = PipeWirePlayer,
                 clock: Callable = time.monotonic):
        self._produce, self._player_factory, self._clock = produce, player_factory, clock
        self._lock = threading.RLock()
        self._clips: OrderedDict = OrderedDict()
        self._active: dict = {}
        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, name='shared-audio-events', daemon=True)
        self._worker.start()

    def enqueue(self, round_, event):
        self._queue.put((round_, dict(event)))

    def _valid(self, clip):
        snap = clip.round.snapshot()
        return (not self._stop.is_set() and not clip.cancelled.is_set()
                and snap['status'] == 'running'
                and snap.get('state_version') == clip.directive.get('state_version'))

    def ensure(self, round_, directive):
        key = (round_.id, directive['directive_id'])
        current_version = round_.snapshot().get('state_version')
        with self._lock:
            clip = self._clips.get(key)
            if clip is not None:
                return clip
            clip = SpeechClip(round_, dict(directive))
            if directive.get('state_version') != current_version or round_.snapshot()['status'] != 'running':
                # An old browser must never replace current speech with history.
                clip.cancelled.set()
                clip.done = clip.finished = True
                return clip
            self._clips[key] = clip
            previous = self._active.get(round_.id)
            if previous:
                previous.cancelled.set()
            self._active[round_.id] = clip
            threading.Thread(target=self._generate, args=(clip,), name='shared-audio-producer', daemon=True).start()
            threading.Thread(target=self._play, args=(clip,), name='board-audio-player', daemon=True).start()
            return clip

    def _generate(self, clip):
        try:
            def emit(audio):
                if not self._valid(clip):
                    raise InterruptedError('speech superseded')
                fmt, pcm = wav_pcm(audio)
                seconds = len(pcm) / (fmt[0] * fmt[1] * fmt[2])
                with clip.condition:
                    # Bound in-memory audio even if a provider misbehaves.
                    if clip.bytes + len(pcm) > 32 * 1024 * 1024:
                        raise ValueError('speech exceeds 32 MiB cache limit')
                    clip.frames.append((clip.duration, clip.duration + seconds, fmt, pcm))
                    clip.bytes += len(pcm)
                    clip.duration += seconds
                    clip.condition.notify_all()
            self._produce(clip.round, clip.directive, emit)
        except InterruptedError:
            clip.cancelled.set()
        except Exception as exc:
            clip.error = str(exc)
            self._error(clip, 'synthesis', str(exc))
        finally:
            with clip.condition:
                clip.done = True
                clip.condition.notify_all()

    def _error(self, clip, kind, message):
        if clip.round.snapshot()['status'] == 'running':
            clip.round.emit_observation({'event': 'audio_error', 'kind': kind,
                'directive_id': clip.directive['directive_id'], 'message': message})
        print(f'[audio:{clip.round.id[:8]}] {kind}: {message}', flush=True)

    def _start(self, clip):
        if clip.started is not None or not self._valid(clip):
            return
        clip.started = self._clock()
        clip.round.emit_observation({'event': 'audio_started',
            'directive_id': clip.directive['directive_id'], 'started_at_ms': int(time.time()*1000)})
        if clip.directive.get('notify_start'):
            clip.round.submit_intent('speech_started', {'directive_id': clip.directive['directive_id']})

    def _wait_until(self, deadline, clip):
        while self._valid(clip) and self._clock() < deadline:
            clip.cancelled.wait(min(.03, deadline - self._clock()))

    def _play(self, clip):
        player = None
        index = 0
        fmt = None
        try:
            while self._valid(clip):
                with clip.condition:
                    clip.condition.wait_for(lambda: len(clip.frames) > index or clip.done or clip.cancelled.is_set(), .1)
                    frame = clip.frames[index] if index < len(clip.frames) else None
                    done = clip.done
                if frame is None:
                    if done:
                        break
                    continue
                _, _, current_fmt, pcm = frame
                if fmt is not None and fmt != current_fmt:
                    raise ValueError('speech stream changed its PCM format')
                fmt = current_fmt
                if not clip.board_error:
                    try:
                        player = player or self._player_factory(fmt)
                        # A small first write establishes playback before feeding
                        # the rest of a long WAV (which may block on pipe capacity).
                        first = max(1, fmt[2] // 100) * fmt[0] * fmt[1]
                        player.write(pcm[:first], lambda: not self._valid(clip))
                        self._start(clip)
                        player.write(pcm[first:], lambda: not self._valid(clip))
                    except InterruptedError:
                        break
                    except Exception as exc:
                        clip.board_error = str(exc)
                        self._error(clip, 'board_playback', str(exc))
                        if player:
                            player.close(); player = None
                        # User-selected policy: continue on the same audio clock.
                        self._start(clip)
                index += 1
            if self._valid(clip) and clip.started is not None:
                deadline = clip.started + max(clip.duration, float(clip.directive.get('playback_seconds') or 0))
                if player:
                    try:
                        player.finish(lambda: not self._valid(clip), max(deadline, self._clock()) + 10)
                    except InterruptedError:
                        pass
                    except Exception as exc:
                        clip.board_error = str(exc)
                        self._error(clip, 'board_playback', str(exc))
                self._wait_until(deadline, clip)
        except Exception as exc:
            self._error(clip, 'board_playback', str(exc))
        finally:
            if player:
                player.close()
            # Backend is the single acknowledgement owner, including synthesis
            # failure and cancellation; duplicate done is engine-idempotent.
            try:
                if clip.round.snapshot()['status'] == 'running':
                    clip.round.submit_intent('speech_done', {'directive_id': clip.directive['directive_id']})
            except Exception:
                pass
            clip.finished = True
            clip.expires_at = self._clock() + 45
            with clip.condition:
                clip.condition.notify_all()
            clip.round.emit_observation({'event': 'audio_finished', 'directive_id': clip.directive['directive_id']})

    def current(self, round_):
        with self._lock:
            clip = self._active.get(round_.id)
        if not clip or clip.finished or not self._valid(clip):
            return None
        return {**clip.directive, 'position_seconds': max(0, self._clock() - clip.started) if clip.started is not None else 0,
                'board_error': clip.board_error}

    def frames(self, clip, *, live=False):
        """All consumers share generation. Live consumers skip already played PCM."""
        position = max(0, self._clock() - clip.started) if live and clip.started is not None else 0
        index = 0
        while not self._stop.is_set():
            with clip.condition:
                clip.condition.wait_for(lambda: index < len(clip.frames) or clip.done or clip.cancelled.is_set(), .2)
                available = list(clip.frames[index:])
                index += len(available)
                done, error = clip.done, clip.error
            if clip.cancelled.is_set() or not self._valid(clip):
                return
            for start, end, fmt, pcm in available:
                if end <= position:
                    continue
                offset = max(0, int((position-start) * fmt[2])) * fmt[0] * fmt[1]
                yield pcm_wav(fmt, pcm[offset:])
            if done:
                if error:
                    raise RuntimeError(error)
                return

    def health(self):
        with self._lock:
            return {'worker_running': self._worker.is_alive(), 'cached_directives': len(self._clips),
                    'active': [{'round_id': c.round.id, 'directive_id': c.directive['directive_id'],
                                'board_error': c.board_error} for c in self._active.values() if not c.finished]}

    def _run(self):
        while not self._stop.is_set():
            try:
                round_, event = self._queue.get(timeout=.2)
                if event.get('event') == 'speech':
                    self.ensure(round_, event)
                elif event.get('event') in {'state_changed', 'round_complete'}:
                    with self._lock:
                        clip = self._active.get(round_.id)
                    if clip and not self._valid(clip):
                        clip.cancelled.set()
            except queue.Empty:
                pass
            except Exception as exc:
                print(f'[audio] event handling failed: {exc}', flush=True)
            with self._lock:
                for key, clip in list(self._clips.items()):
                    if clip.finished and (clip.expires_at <= self._clock() or len(self._clips) > 64 or sum(c.bytes for c in self._clips.values()) > 96*1024*1024):
                        self._clips.pop(key, None)
                        if self._active.get(clip.round.id) is clip:
                            self._active.pop(clip.round.id, None)

    def close(self):
        self._stop.set()
        with self._lock:
            for clip in self._clips.values():
                clip.cancelled.set()
        self._worker.join(timeout=2)
