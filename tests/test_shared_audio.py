"""Speech fanout with real PCM framing and fake default-sink playback."""
import io
import sys
import threading
import time
import wave
from pathlib import Path

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from core.shared_audio import SharedAudio, pcm_wav, wav_pcm
from core.state_machine import GameRound

FMT=(1,2,8000)
WAV=pcm_wav(FMT,b'\0\0'*400)


def wait(check):
    end=time.monotonic()+3
    while time.monotonic()<end:
        if check(): return
        time.sleep(.005)
    assert check(), 'audio task did not finish'


class Round:
    id='round-1'
    game_id='rps'
    def __init__(self):
        self.version=1;self.status='running';self.events=[];self.inputs=[]
    def snapshot(self):
        return {'state_version':self.version,'status':self.status}
    def emit_observation(self,e): self.events.append(e)
    def submit_intent(self,name,payload): self.inputs.append((name,payload['directive_id']))


class Player:
    created=[]
    def __init__(self,fmt):
        self.fmt=fmt;self.data=bytearray();self.closed=False
        self.created.append(self)
    def write(self,data,cancelled):
        assert not cancelled();self.data.extend(data)
    def finish(self,cancelled,deadline): assert not cancelled()
    def close(self): self.closed=True


def directive(name='clip-1',version=1):
    return {'event':'speech','directive_id':name,'mode':'audio','notify_start':True,
            'playback_seconds':.05,'state_version':version}


def test_board_and_two_browsers_share_one_producer_and_ack_owner():
    count=[];r=Round()
    def produce(round_, d, emit):
        count.append(d['directive_id']);emit(WAV)
    manager=SharedAudio(produce=produce,player_factory=Player)
    try:
        a=manager.ensure(r,directive())
        b=manager.ensure(r,directive())
        assert a is b
        one=b''.join(manager.frames(a));two=b''.join(manager.frames(b))
        assert one==two==WAV
        wait(lambda:a.finished)
        assert count==['clip-1']
        assert r.inputs==[('speech_started','clip-1'),('speech_done','clip-1')]
        assert Player.created[-1].data==wav_pcm(WAV)[1]
        assert manager.current(r) is None
    finally:manager.close()


def test_board_playback_failure_broadcasts_but_remote_audio_and_clock_continue():
    r=Round()
    def broken(fmt):raise OSError('no default sink')
    m=SharedAudio(produce=lambda r,d,emit:emit(WAV),player_factory=broken)
    try:
        clip=m.ensure(r,directive())
        assert b''.join(m.frames(clip))==WAV
        wait(lambda:clip.finished)
        assert r.inputs==[('speech_started','clip-1'),('speech_done','clip-1')]
        assert any(e['event']=='audio_error' and e['kind']=='board_playback' for e in r.events)
        assert clip.started is not None and time.monotonic()-clip.started>=.045
    finally:m.close()


def test_live_join_skips_already_played_pcm():
    r=Round();release=threading.Event();now=[0.]
    def produce(r,d,emit):
        emit(WAV);release.wait(2)
    m=SharedAudio(produce=produce,player_factory=Player,clock=lambda:now[0])
    try:
        clip=m.ensure(r,directive())
        wait(lambda:clip.started is not None)
        now[0]=.025
        frames=m.frames(clip,live=True)
        frame=next(frames)
        assert len(wav_pcm(frame)[1])==400
        now[0]=1.;release.set()
        wait(lambda:clip.finished)
    finally:release.set();now[0]=10;m.close()


def test_cancel_and_supersede_stop_old_audio_without_double_synthesis():
    r=Round();release=threading.Event();calls=[]
    def produce(r,d,emit):
        calls.append(d['directive_id']);emit(WAV)
        if d['directive_id']=='old': release.wait(1)
    m=SharedAudio(produce=produce,player_factory=Player)
    try:
        old=m.ensure(r,directive('old'));wait(lambda:old.started is not None)
        new=m.ensure(r,directive('new'))
        assert old.cancelled.is_set()
        release.set();wait(lambda:old.finished and new.finished)
        assert calls==['old','new']
        assert sum(n=='speech_started' and did=='new' for n,did in r.inputs)==1
    finally:release.set();m.close()


def test_synthesis_error_releases_await_without_fake_playback_start():
    r=Round()
    def fail(r,d,emit):raise ValueError('bad audio')
    m=SharedAudio(produce=fail,player_factory=Player)
    try:
        clip=m.ensure(r,directive());wait(lambda:clip.finished)
        assert r.inputs==[('speech_done','clip-1')]
        assert any(e['event']=='audio_error' and e['kind']=='synthesis' for e in r.events)
        with pytest.raises(RuntimeError,match='bad audio'):list(m.frames(clip))
    finally:m.close()


def test_real_awaited_start_action_runs_once_from_backend_not_consumers():
    robots=[]
    machine={'schema_version':1,'initial':'play','states':{
        'play':{'on_enter':[{'action':'speech','mode':'audio','audio':'audio/call.wav','await':True,
                'playback_seconds':.05,'on_start':[{'action':'robot','command':'throw_gesture'}]}],
                'duration':.01,'on_expire':{'to':'result'}},
        'result':{'on_intent':{'back':{'exit':True}}}}}
    m=SharedAudio(produce=lambda r,d,emit:emit(WAV),player_factory=Player)
    round_=GameRound(game_id='rps',manifest={'state_machine':machine},
        robot_fn=lambda *args:robots.append(args[0]['command']) or {'status':'completed'},
        event_sink=lambda e:m.enqueue(round_,e))
    try:
        round_.start()
        wait(lambda:round_.state=='result')
        assert robots==['throw_gesture']
        assert len([e for e in round_.snapshot()['events'] if e['event']=='speech_started'])==1
    finally:round_.cancel();m.close()


def test_late_historical_speech_cannot_replace_current_clip():
    r=Round();r.version=2
    m=SharedAudio(produce=lambda r,d,emit:emit(WAV),player_factory=Player)
    try:
        current=m.ensure(r,directive('current',2))
        historical=m.ensure(r,directive('historical',1))
        assert historical.cancelled.is_set()
        assert not current.cancelled.is_set()
        wait(lambda:current.finished)
    finally:m.close()
