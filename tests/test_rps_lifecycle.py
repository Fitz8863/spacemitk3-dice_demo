"""Failure and cleanup paths must not launch a throw or leak the camera."""
from pathlib import Path
import subprocess
import sys
import json
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from core.state_machine import GameRound
from core.state_schema import validate_state_machine, StateMachineError
from test_state_machine import wait_for


def test_audio_failure_does_not_start_robot():
    manifest=json.loads((ROOT/'backend/games/rps/manifest.json').read_text())
    calls=[]
    def robot(action,*args):
        calls.append(action['command'])
        return {'status':'completed'}
    round_=GameRound(game_id='rps',manifest=manifest,robot_fn=robot,log=lambda _:None)
    try:
        round_.start();round_.submit_intent('confirm')
        assert wait_for(lambda: round_._awaiting_directive is not None)
        round_.submit_intent('speech_done',{'directive_id':round_._awaiting_directive})
        assert wait_for(lambda: round_.status=='error')
        assert calls==['prepare_throw']
        assert '口令未开始播放' in round_.error
    finally:round_.cancel()


def test_stop_script_scopes_cleanup_and_includes_both_detectors():
    text=(ROOT/'scripts/stop_web.sh').read_text()
    function=text[text.index('runtime_children() {'):text.index('kill_runtime_children() {')]
    script='''ROOT_DIR=/srv/arena
pgrep() { case "$2" in yolov8_camera) echo 81;; yolov10_camera) echo 101; echo 102;; esac; }
readlink() { case "$2" in /proc/81/exe) echo /srv/arena/vision/yolov8_camera;; /proc/101/exe) echo /srv/arena/vision/yolov10_camera;; /proc/102/exe) echo /srv/other/yolov10_camera;; esac; }
'''+function+'\nruntime_children\n'
    result=subprocess.run(['bash','-c',script],capture_output=True,text=True,check=True)
    assert result.stdout.split()==['81','101']


@pytest.mark.parametrize('change',[{'mode':'tts_local'}, {'await':False}, {'on_start':[{'action':'adjudicate'}]}, {'playback_seconds':-1}])
def test_invalid_audio_start_hooks_rejected(change):
    manifest=json.loads((ROOT/'backend/games/rps/manifest.json').read_text())
    manifest['state_machine']['states']['play']['on_enter'][0].update(change)
    with pytest.raises(StateMachineError):validate_state_machine(manifest['state_machine'], 'rps')


def test_browser_rps_replay_and_audio_clock():
    import shutil
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is needed for the browser module harness')
    result = subprocess.run(
        [node, str(ROOT / 'tests/js/rps_round_regressions.mjs'), str(ROOT)],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
