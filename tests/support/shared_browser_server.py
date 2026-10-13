"""Local browser fixture: real HTTP/SSE/game engine; no device/provider IO.

Run with Python, --port N. Robot, camera, ASR, speaker and audio generation are
explicitly replaced. This script must never import or execute server.main().
"""
from __future__ import annotations
import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backend'))
import server
from core.shared_audio import SharedAudio,pcm_wav
from core.shared_session import SharedSession
from core.state_machine import GameRound

config={'input_board_enable':True,'asr_enabled':False,
        'participants':{'player':'LEFT','agent':'RIGHT'}, 'voice':'default','speed':1.0,
        'standby':{'enabled':True,'idle_seconds':120,'boot_standby':True}}
server.get_arena_config=lambda:config
server._board_input_source=lambda:{'enabled':True,'available':False}
server._board_input_status=lambda:{'enabled':True,'running':False,'devices':[]}
robot_calls=[]
productions=[]
players=[]

class Player:
    def __init__(self,fmt):self.data=0;players.append(self)
    def write(self,data,cancelled):self.data+=len(data)
    def finish(self,cancelled,deadline):pass
    def close(self):pass

def produce(round_,directive,emit):
    productions.append([round_.id,directive['directive_id']])
    # Audible sine so browser playback can be checked without a real speaker.
    import math,struct
    pcm=b''.join(struct.pack('<h',int(1500*math.sin(n*2*math.pi*440/8000))) for n in range(32000))
    emit(pcm_wav((1,2,8000),pcm))

audio=SharedAudio(produce=produce,player_factory=Player)
server._SHARED_AUDIO=audio

def adjudicate(manifest,on_event,cancelled,log):
    if manifest['id']=='dice':
        return {'verified':True,'player_side':'LEFT','agent_side':'RIGHT','winner_role':'PLAYER',
                'player_score':25,'agent_score':10,'player_values':[5]*5,'agent_values':[2]*5}
    return {'verified':True,'player_side':'LEFT','agent_side':'RIGHT','winner_role':'PLAYER',
            'player_choice':'rock','agent_choice':'scissors','player_score':1,'agent_score':0}

def robot(action,on_event,cancelled,log):
    robot_calls.append(action['command'])
    on_event({'event':'robot','command':action['command'],'phase':'completed'})
    return {'status':'completed','gesture':'scissors'}

def create(game,sink):
    manifest=copy.deepcopy(server.with_global_defaults(server.get_games().get(game),config))
    for state in manifest['state_machine']['states'].values():
        if 'duration' in state:state['duration']=min(state['duration'],.15)
        for action in state.get('on_enter',[]):
            if action.get('playback_seconds'):action['playback_seconds']=.1
            for nested in action.get('on_start',[]):nested['delay_seconds']=.05
    def notify(event):
        sink(round_,event);audio.enqueue(round_,event)
    round_=GameRound(game_id=game,manifest=manifest,robot_fn=robot,
        adjudicate_fn=adjudicate,event_sink=notify)
    round_.shared=True
    server.rounds[round_.id]=round_
    round_.start()
    return round_

session=SharedSession(config=lambda:config,games=lambda:['dice','rps'],create_round=create,
                      board_source=server._board_input_source)
server._SESSION=session

class Handler(server.Handler):
    def _is_board_local_client(self):
        # Emulate a remote browser; no actual board keyboard runs in this fixture.
        return False
    def do_GET(self):
        if self.path=='/test/metrics':
            self.send_json({'robot_calls':robot_calls,'productions':productions,
                            'board_players':len(players),'round_ids':list(server.rounds)})
            return
        return super().do_GET()
    def do_POST(self):
        if self.path=='/test/reset':
            self.read_json()
            if session.snapshot()['round_id']:
                session.submit({'action':'exit','round_id':session.snapshot()['round_id']})
            with session._condition:
                session._phase='standby'
                session._changed()
            self.send_json(session.snapshot());return
        if self.path=='/test/board':
            command=self.read_json();snap=session.snapshot();r=snap.get('round') or {}
            session.submit({'action':'board','key':command['key'],'round_id':snap['round_id'],
                            'state':r.get('state'),'state_version':r.get('state_version')})
            self.send_json(session.snapshot());return
        return super().do_POST()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=18681)
    args=parser.parse_args()
    httpd=server.ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    try:httpd.serve_forever()
    finally:session.close();audio.close();httpd.server_close()
