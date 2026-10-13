"""Shared HTTP compatibility uses an ephemeral server and fake round factory."""
import json
import sys
import threading
from http.client import HTTPConnection
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
import server
from core.shared_session import SharedSession
from test_shared_session import Round


@pytest.fixture
def http_session(monkeypatch):
    made=[]
    def create(game, sink):
        r=Round(game, sink)
        r.shared=True
        made.append(r)
        server.rounds[r.id]=r
        return r
    s=SharedSession(config=lambda: {'input_board_enable': True}, games=lambda: ['dice','rps'], create_round=create)
    monkeypatch.setattr(server, '_SESSION', s)
    monkeypatch.setattr(server, 'rounds', {})
    monkeypatch.setattr(server, 'get_arena_config', lambda: {'input_board_enable': True, 'asr_enabled': False})
    httpd=server.ThreadingHTTPServer(('127.0.0.1',0), server.Handler)
    t=threading.Thread(target=httpd.serve_forever,daemon=True);t.start()
    yield httpd.server_address[1],s,made
    s.close();httpd.shutdown();httpd.server_close();t.join(2)


def request(port, method, path, payload=None):
    c=HTTPConnection('127.0.0.1',port,timeout=5)
    c.request(method,path,body=json.dumps(payload) if payload is not None else None,
              headers={'Content-Type':'application/json'})
    r=c.getresponse();body=json.loads(r.read());c.close()
    return r.status,body


def test_http_selection_then_parallel_legacy_enter_reuses_round(http_session):
    port,s,made=http_session
    status,b=request(port,'POST','/api/session/actions',{'action':'select','game':'rps','request_id':'select'})
    assert status==200 and b['selected_game']=='rps'
    status,a=request(port,'POST','/api/game/rounds',{'game':'rps'})
    status,b=request(port,'POST','/api/game/rounds',{'game':'dice'})
    assert status==201 and a['round_id']==b['round_id']
    assert len(made)==1 and made[0].status=='running'
    _,snap=request(port,'GET','/api/session')
    assert snap['round_id']==a['round_id'] and snap['local_client'] is True


def test_old_listen_endpoints_cannot_clear_events_or_stop_asr(http_session,monkeypatch):
    port,_,_=http_session
    bus=server._AsrEventBus();bus.push({'status':'board_navigate','game_id':'rps'})
    monkeypatch.setattr(server,'_SELECT_BUS',bus)
    monkeypatch.setattr(server.ASR_BRIDGE,'stop',lambda:pytest.fail('a webpage stopped shared ASR'))
    for path in ('/api/asr/select','/api/asr/standby'):
        for listen in (True,False):
            assert request(port,'POST',path,{'listen':listen})[0]==200
    assert bus.poll()[0]['game_id']=='rps'


def test_legacy_web_speech_ack_cannot_advance_shared_round(http_session):
    port,_,made=http_session
    _,snap=request(port,'POST','/api/session/actions',{'action':'enter'})
    rid=snap['round_id']
    assert request(port,'POST',f'/api/game/rounds/{rid}/intents',{'intent':'speech_done','directive_id':'x'})[0]==200
    assert made[0].inputs==[]
    status,snap=request(port,'POST','/api/session/actions',{'action':'intent','round_id':rid,'state':'rules','state_version':1,'intent':'confirm'})
    assert status==200 and snap['round']['state']=='ready'
    status,_=request(port,'POST','/api/session/actions',{'action':'intent','round_id':rid,'state':'rules','state_version':1,'intent':'confirm'})
    assert status==409


def test_session_stream_first_frame_is_current_snapshot(http_session):
    port,s,_=http_session
    s.submit({'action':'select','game':'rps'})
    c=HTTPConnection('127.0.0.1',port,timeout=3)
    c.request('GET','/api/session/stream');r=c.getresponse()
    assert r.status==200
    assert r.fp.readline().strip()==b'event: snapshot'
    data=json.loads(r.fp.readline().removeprefix(b'data: '))
    assert data['selected_game']=='rps'
    c.close()
