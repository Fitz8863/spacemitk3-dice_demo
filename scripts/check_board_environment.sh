#!/usr/bin/env bash
# Import/link/inference checks only: no camera, microphone, CAN or playback.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WITH_ASR=0
WITH_MATCHA=0
for option in "$@"; do
    case "$option" in
        --with-asr) WITH_ASR=1 ;;
        --with-matcha) WITH_MATCHA=1 ;;
        *) echo "check: unknown option $option" >&2; exit 2 ;;
    esac
done
for tool in python3 pw-cat pw-cli v4l2-ctl gst-inspect-1.0 curl ip ldd; do
    command -v "$tool" >/dev/null || { echo "check: missing $tool; run scripts/setup_board.sh deps --yes" >&2; exit 1; }
done
PYTHONNOUSERSITE=1 DICE_PYTHON=/usr/bin/python3 \
    bash "$ROOT_DIR/scripts/robot_dependencies.sh" check
python3 "$ROOT_DIR/scripts/board_assets.py" moss
MOSS_ROOT="$(python3 "$ROOT_DIR/scripts/board_assets.py" moss --path)"
# Mirror MOSS's launcher, without exporting the robot's locked loader paths
# into the parent shell or the TTS process and without allocating model sessions.
PYTHONPATH="$MOSS_ROOT/python:$MOSS_ROOT/src:$MOSS_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
LD_LIBRARY_PATH="$MOSS_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
PYTHONNOUSERSITE=1 python3 - "$ROOT_DIR" <<'PY'
import importlib, json, sys
from pathlib import Path
import soundfile
from ort_cpu_runtime import _resolve_ort_providers
root = Path(sys.argv[1])
for name in ('numpy', 'scipy.signal', 'sentencepiece', 'soundfile',
             'spacemit_ort', 'onnx_tts_runtime'):
    module = importlib.import_module(name)
    print('MOSS import:', name, getattr(module, '__file__', 'built-in'))
config = json.loads((root/'backend/components/tts_moss_nano/config.json').read_text())
_resolve_ort_providers(config.get('execution_provider', {}).get('name', 'spacemit'))
if config.get('voice', {}).get('mode') == 'clone':
    reference = root/config['runtime']['root']/config['voice']['reference_audio']
    assert soundfile.info(reference).frames > 0, f'Empty voice reference: {reference}'
print('MOSS imports, EP and voice WAV OK; no synthesis or playback started.')
PY
python3 - "$ROOT_DIR" "$WITH_ASR" "$WITH_MATCHA" <<'PY'
import json, os, subprocess, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root/'backend'))
from components.vision_yolov8_objdetect.process import locked_recognition_environment

def check_linkage(binary, environment=None):
    result = subprocess.run(['ldd', str(binary)], env=environment,
                            capture_output=True, text=True, timeout=30)
    if result.returncode or 'not found' in result.stdout + result.stderr:
        raise RuntimeError(f'Missing shared libraries for {binary}:\n{result.stdout}{result.stderr}')

for game, provider in (('dice', 'vision_yolov8_objdetect'), ('rps', 'vision_yolov10_objdetect')):
    config = json.loads((root/f'backend/components/{provider}/config.json').read_text())
    binary = root/config['runtime']['binary']
    # Match the provider's actual process environment, including its lock switch.
    environment = locked_recognition_environment(config)
    check_linkage(binary, environment)
    flags = ['--yolov8'] if game == 'dice' else []
    subprocess.run([str(binary), '--config', str(root/f'backend/games/{game}/adjudicator_config.json'),
                    *flags, '--self-test', '--no-display', '--no-rtsp'],
                   env=environment, check=True, timeout=60)

arena = json.loads((root/'backend/config.json').read_text())
if sys.argv[2] == '1' or arena.get('asr_enabled', False):
    subprocess.run([sys.executable, str(root/'scripts/board_assets.py'), 'sensevoice'], check=True)
    config = json.loads((root/'backend/components/asr_sensevoice/config.json').read_text())
    check_linkage(root/config['runtime']['binary'])
    print('SenseVoice assets and linkage OK; microphone not opened.')
if sys.argv[3] == '1':
    subprocess.run([sys.executable, str(root/'scripts/board_assets.py'), 'matcha'], check=True)
    config = json.loads((root/'backend/components/tts_matcha/config.json').read_text())
    base = root/config['runtime']['root']
    environment = dict(os.environ, LD_LIBRARY_PATH=str(base/config['runtime']['sherpa_lib_dir']) + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
    check_linkage(base/config['runtime']['binary'], environment)
    print('Matcha assets and linkage OK; no synthesis started.')
PY
# Connecting to PipeWire's metadata is read-only. Missing hardware/session is
# a site setup issue, and shared speech failure already allows gameplay to proceed.
if pw-cli info 0 >/dev/null 2>&1; then
    echo "check: PipeWire user session reachable (no audio played)"
else
    echo "check: [警告] PipeWire 用户会话不可达；请从登录用户会话启动服务并确认默认声卡" >&2
fi
echo "check: software dependencies OK; camera/CAN/calibration require site setup"
