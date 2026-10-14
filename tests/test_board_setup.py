"""Run the customer installer against isolated, incomplete delivery trees."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
MOSS_MODEL = 'models/MOSS-TTS-Nano-100M-ONNX-xslim-dynq'
CODEC_MODEL = 'models/MOSS-Audio-Tokenizer-Nano-ONNX'


def write(path, data=b'payload'):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        data = data.encode()
    path.write_bytes(data)


def moss_payload():
    files = {
        f'{MOSS_MODEL}/browser_poc_manifest.json': json.dumps({'model_files': {
            'tts_meta': 'tts_browser_onnx_meta.json',
            'codec_meta': f'../MOSS-Audio-Tokenizer-Nano-ONNX/codec_browser_onnx_meta.json',
            'tokenizer_model': 'tokenizer.model',
        }}).encode(),
        f'{MOSS_MODEL}/tts_browser_onnx_meta.json': json.dumps({'files': {
            'prefill': 'prefill.onnx', 'decode_step': 'decode.onnx',
            'local_fixed_sampled_frame': 'frame.onnx',
        }, 'external_data_files': {}}).encode(),
        f'{CODEC_MODEL}/codec_browser_onnx_meta.json': json.dumps({'files': {
            'encode': 'encode.onnx', 'decode_full': 'decode.onnx',
        }, 'external_data_files': {'encode.onnx': ['encode.data']}}).encode(),
    }
    for name in (
        f'{MOSS_MODEL}/tokenizer.model', f'{MOSS_MODEL}/prefill.onnx',
        f'{MOSS_MODEL}/decode.onnx', f'{MOSS_MODEL}/frame.onnx',
        f'{CODEC_MODEL}/encode.onnx', f'{CODEC_MODEL}/decode.onnx',
        f'{CODEC_MODEL}/encode.data', 'python/spacemit_ort/__init__.py',
        'python/scipy/__init__.py', 'python/sentencepiece/__init__.py',
        'python/soundfile.py', 'python/cffi/__init__.py',
        'python/_cffi_backend.cpython-314-riscv64-linux-gnu.so',
        'python/sentencepiece/_sentencepiece.cpython-314-riscv64-linux-gnu.so',
        'python/spacemit_ort/libspacemit_ep.so.2.0.6',
        'lib/libspacemit_ep.so', 'lib/libspert.so.1',
        'voice/warm.wav', 'assets/reference.wav',
    ):
        files[name] = b'payload'
    return files


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / 'customer checkout'
    (root / 'scripts').mkdir(parents=True)
    shutil.copy(ROOT / 'scripts/setup_board.sh', root / 'scripts/setup_board.sh')
    helper = ROOT / 'scripts/board_assets.py'
    if helper.is_file():
        shutil.copy(helper, root / 'scripts/board_assets.py')
    write(root / 'backend/config.json', json.dumps({'input_board_enable': True}))
    write(root / 'backend/components/tts_moss_nano/config.json', json.dumps({
        'runtime': {'root': 'tts/moss-tts-nano', 'model_dir': MOSS_MODEL},
        'voice': {'mode': 'clone', 'reference_audio': 'voice/warm.wav'},
    }))
    # A real git clone has these files and creates the model directory, while
    # all three model weights and the Python/lib/voice/assets trees are absent.
    files = moss_payload()
    for name in ('browser_poc_manifest.json', 'tts_browser_onnx_meta.json', 'tokenizer.model'):
        write(root / 'tts/moss-tts-nano' / MOSS_MODEL / name,
              files[f'{MOSS_MODEL}/{name}'])
    return root


def cache_archive(root, files):
    path = root / '.runtime/downloads/moss-tts-nano-board-assets.tar.gz'
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, 'w:gz') as archive:
        for name, data in files.items():
            item = tarfile.TarInfo(name)
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
    write(Path(str(path) + '.sha256'), hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + path.name + '\n')
    return path


def install(root, *steps, env=None):
    return subprocess.run(['bash', str(root / 'scripts/setup_board.sh'), *steps],
                          env=env or os.environ.copy(), text=True,
                          capture_output=True, timeout=30)


def test_fresh_clone_provisions_weights_and_runtime_despite_existing_model_directory(checkout):
    cache_archive(checkout, moss_payload())
    result = install(checkout, 'models')
    assert result.returncode == 0, result.stdout + result.stderr
    for relative in moss_payload():
        assert (checkout / 'tts/moss-tts-nano' / relative).is_file(), relative


def test_complete_moss_delivery_is_reused_without_downloading_or_overwriting(checkout):
    for name, data in moss_payload().items():
        write(checkout / 'tts/moss-tts-nano' / name, data)
    voice = checkout / 'tts/moss-tts-nano/voice/warm.wav'
    voice.write_bytes(b'customer voice')
    result = install(checkout, 'models')
    assert result.returncode == 0, result.stdout + result.stderr
    assert voice.read_bytes() == b'customer voice'
    assert not (checkout / '.runtime/downloads').exists()


def test_extraction_with_missing_codec_external_data_does_not_report_success(checkout):
    files = moss_payload()
    files.pop(f'{CODEC_MODEL}/encode.data')
    cache_archive(checkout, files)
    result = install(checkout, 'models')
    assert result.returncode != 0, result.stdout + result.stderr
    assert 'encode.data' in result.stdout + result.stderr


def test_checksum_failure_stops_before_any_archive_is_extracted(checkout):
    path = cache_archive(checkout, moss_payload())
    write(Path(str(path) + '.sha256'), '0' * 64 + '\n')
    result = install(checkout, 'models')
    assert result.returncode != 0, result.stdout + result.stderr
    assert not (checkout / 'tts/moss-tts-nano' / MOSS_MODEL / 'prefill.onnx').exists()


def test_empty_model_file_is_repaired_from_the_asset_archive(checkout):
    files = moss_payload()
    for name, data in files.items():
        write(checkout / 'tts/moss-tts-nano' / name, data)
    write(checkout / 'tts/moss-tts-nano' / MOSS_MODEL / 'prefill.onnx', b'')
    cache_archive(checkout, files)
    result = install(checkout, 'models')
    assert result.returncode == 0, result.stdout + result.stderr
    assert (checkout / 'tts/moss-tts-nano' / MOSS_MODEL / 'prefill.onnx').stat().st_size > 0


def fake_command(root, name, source):
    path = root / 'fake-bin' / name
    write(path, '#!/usr/bin/env bash\n' + source + '\n')
    path.chmod(0o755)
    return path


def test_robot_stage_reuses_verified_dependencies_without_sudo(checkout):
    write(checkout / 'scripts/robot_dependencies.sh', '#!/bin/bash\n[[ \"$1\" == check ]]\n')
    fake_command(checkout, 'sudo', 'exit 99')
    env = dict(os.environ, PATH=str(checkout / 'fake-bin') + ':' + os.environ['PATH'])
    result = install(checkout, 'robot', env=env)
    assert result.returncode == 0, result.stdout + result.stderr


def test_robot_stage_bootstraps_missing_dependencies_then_checks_them(checkout):
    marker = checkout / 'robot-installed'
    write(checkout / 'scripts/robot_dependencies.sh',
          '#!/bin/bash\nif [[ "$1" == install ]]; then touch "' + str(marker) + '"; else test -f "' + str(marker) + '"; fi\n')
    fake_command(checkout, 'sudo', '"$@"')
    env = dict(os.environ, PATH=str(checkout / 'fake-bin') + ':' + os.environ['PATH'])
    result = install(checkout, 'robot', '--yes', env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert marker.is_file()


def test_robot_bootstrap_failure_is_not_reported_as_installed(checkout):
    write(checkout / 'scripts/robot_dependencies.sh', '#!/bin/bash\nexit 42\n')
    fake_command(checkout, 'sudo', '"$@"')
    env = dict(os.environ, PATH=str(checkout / 'fake-bin') + ':' + os.environ['PATH'])
    result = install(checkout, 'robot', '--yes', env=env)
    assert result.returncode != 0


def test_failed_imports_after_bootstrap_are_not_reported_as_installed(checkout):
    write(checkout / 'scripts/robot_dependencies.sh', '#!/bin/bash\n[[ "$1" == install ]]\n')
    fake_command(checkout, 'sudo', '"$@"')
    env = dict(os.environ, PATH=str(checkout / 'fake-bin') + ':' + os.environ['PATH'])
    result = install(checkout, 'robot', '--yes', env=env)
    assert result.returncode != 0


def test_missing_published_checksum_stops_before_extraction(checkout):
    path = cache_archive(checkout, moss_payload())
    Path(str(path) + '.sha256').unlink()
    fake_command(checkout, 'curl', 'exit 22')
    env = dict(os.environ, PATH=str(checkout / 'fake-bin') + ':' + os.environ['PATH'])
    result = install(checkout, 'models', env=env)
    assert result.returncode != 0
    assert not (checkout / 'tts/moss-tts-nano' / MOSS_MODEL / 'prefill.onnx').exists()


def test_interrupted_download_resumes_part_then_verifies_before_extraction(checkout):
    path = cache_archive(checkout, moss_payload())
    remote = checkout / 'remote-asset.tar.gz'
    path.rename(remote)
    partial = Path(str(path) + '.part')
    partial.write_bytes(remote.read_bytes()[:20])
    fake_command(checkout, 'curl', '''
[[ " $* " == *" -C - "* ]] || exit 31
while [[ $# -gt 0 ]]; do
    if [[ "$1" == -o ]]; then target="$2"; shift 2; else shift; fi
done
[[ "$target" == *.part && -f "$target" ]] || exit 32
cp "$FAKE_REMOTE_ASSET" "$target"
''')
    env = dict(os.environ, PATH=str(checkout / 'fake-bin') + ':' + os.environ['PATH'],
               FAKE_REMOTE_ASSET=str(remote))
    result = install(checkout, 'models', env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not partial.exists()
    assert path.read_bytes() == remote.read_bytes()
    assert (checkout / 'tts/moss-tts-nano' / MOSS_MODEL / 'prefill.onnx').is_file()


def test_incomplete_download_never_gets_final_archive_name(checkout):
    path = cache_archive(checkout, moss_payload())
    path.unlink()
    fake_command(checkout, 'curl', '''
while [[ $# -gt 0 ]]; do
    if [[ "$1" == -o ]]; then target="$2"; shift 2; else shift; fi
done
printf 'incomplete' > "$target"
exit 28
''')
    env = dict(os.environ, PATH=str(checkout / 'fake-bin') + ':' + os.environ['PATH'])
    result = install(checkout, 'models', env=env)
    assert result.returncode != 0
    assert not path.exists()
    assert Path(str(path) + '.part').read_bytes() == b'incomplete'


def test_malformed_model_metadata_is_not_considered_complete(checkout):
    files = moss_payload()
    files[f'{MOSS_MODEL}/tts_browser_onnx_meta.json'] = b'{}'
    cache_archive(checkout, files)
    result = install(checkout, 'models')
    assert result.returncode != 0
    assert 'asset check failed' in result.stderr


def test_dependency_packages_are_installed_noninteractively_with_yes(checkout):
    fake_command(checkout, 'uname', 'printf "riscv64\\n"')
    fake_command(checkout, 'dpkg-query', 'printf "config-files"')
    fake_command(checkout, 'gst-inspect-1.0', 'exit 0')
    fake_command(checkout, 'apt-get', 'printf "%s\\n" "$*" >> "$FAKE_APT_LOG"')
    fake_command(checkout, 'sudo', '"$@"')
    log = checkout / 'apt.log'
    env = dict(os.environ, PATH=str(checkout / 'fake-bin') + ':' + os.environ['PATH'],
               FAKE_APT_LOG=str(log))
    result = install(checkout, 'deps', '--yes', env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    commands = log.read_text().splitlines()
    assert commands[0] == 'update'
    for dependency in ('python3-spacemit-ort', 'spacemit-tcm', 'pipewire-bin',
                       'pipewire', 'wireplumber', 'python3-numpy', 'iproute2'):
        assert dependency in commands[1].split()


def test_noninteractive_missing_dependencies_require_yes_before_system_changes(checkout):
    if os.geteuid() == 0:
        pytest.skip('root explicitly opts into automatic package installation')
    fake_command(checkout, 'uname', 'printf "riscv64\\n"')
    fake_command(checkout, 'dpkg-query', 'exit 1')
    fake_command(checkout, 'apt-get', 'touch "$FAKE_APT_LOG"')
    fake_command(checkout, 'sudo', '"$@"')
    log = checkout / 'apt.log'
    env = dict(os.environ, PATH=str(checkout / 'fake-bin') + ':' + os.environ['PATH'],
               FAKE_APT_LOG=str(log))
    result = install(checkout, 'deps', env=env)
    assert result.returncode != 0
    assert '--yes' in result.stderr
    assert not log.exists()


def test_permissions_add_customer_groups_and_require_fresh_login(checkout):
    if os.geteuid() == 0:
        pytest.skip('top-level device provisioning is for normal users')
    fake_command(checkout, 'id', '''
case "$1" in
    -un) printf 'customer\\n' ;;
    -nG) printf 'customer\\n' ;;
    *) exit 22 ;;
esac
''')
    fake_command(checkout, 'getent', 'exit 0')
    fake_command(checkout, 'usermod', 'printf "%s\\n" "$*" >> "$FAKE_GROUP_LOG"')
    fake_command(checkout, 'sudo', '"$@"')
    log = checkout / 'groups.log'
    env = dict(os.environ, PATH=str(checkout / 'fake-bin') + ':' + os.environ['PATH'],
               FAKE_GROUP_LOG=str(log))
    result = install(checkout, 'permissions', '--yes', env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert log.read_text().splitlines() == ['-aG video customer', '-aG input customer']
    assert '注销重登' in result.stderr


def test_permissions_preserve_disabled_board_input_and_existing_camera_access(checkout):
    if os.geteuid() == 0:
        pytest.skip('top-level device provisioning is for normal users')
    write(checkout / 'backend/config.json', json.dumps({'input_board_enable': False}))
    fake_command(checkout, 'id', '''
case "$1" in
    -un) printf 'customer\\n' ;;
    -nG) printf 'customer video\\n' ;;
    *) exit 22 ;;
esac
''')
    fake_command(checkout, 'sudo', 'exit 91')
    env = dict(os.environ, PATH=str(checkout / 'fake-bin') + ':' + os.environ['PATH'])
    result = install(checkout, 'permissions', env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((checkout / 'backend/config.json').read_text())['input_board_enable'] is False


def test_invalid_global_config_is_reported_before_permission_changes(checkout):
    write(checkout / 'backend/config.json', '{')
    fake_command(checkout, 'sudo', 'exit 91')
    env = dict(os.environ, PATH=str(checkout / 'fake-bin') + ':' + os.environ['PATH'])
    result = install(checkout, 'permissions', env=env)
    assert result.returncode != 0
    if os.geteuid() != 0:
        assert '未修改设备权限' in result.stderr


def test_default_install_runs_robot_and_permissions_before_successful_check(checkout):
    for name in ('check_board_environment.sh', 'detect.sh'):
        write(checkout / 'scripts' / name, '#!/bin/bash\nexit 0\n')
    write(checkout / 'scripts/robot_dependencies.sh', '#!/bin/bash\n[[ \"$1\" == check ]]\n')
    for name, data in moss_payload().items():
        write(checkout / 'tts/moss-tts-nano' / name, data)
    fake_command(checkout, 'uname', 'printf "riscv64\\n"')
    fake_command(checkout, 'dpkg-query', 'printf "installed"')
    fake_command(checkout, 'gst-inspect-1.0', 'exit 0')
    fake_command(checkout, 'cmake', 'exit 0')
    fake_command(checkout, 'curl', 'exit 22')
    fake_command(checkout, 'ss', 'exit 1')
    fake_command(checkout, 'id', '''
case "$1" in
    -un) printf 'customer\\n' ;;
    -nG) printf 'customer input video\\n' ;;
    *) exit 22 ;;
esac
''')
    fake_command(checkout, 'sudo', 'exit 91')
    env = dict(os.environ, PATH=str(checkout / 'fake-bin') + ':' + os.environ['PATH'])
    result = install(checkout, '--yes', '--no-mediamtx', env=env)
    if os.geteuid() == 0:
        assert result.returncode != 0
        assert '普通用户' in result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'robot 依赖已通过' in result.stdout
        assert 'permissions 通过' in result.stdout
        assert '程序依赖和无硬件自检通过' in result.stdout


def test_verified_complete_part_is_promoted_without_redownloading(checkout):
    path = cache_archive(checkout, moss_payload())
    partial = Path(str(path) + '.part')
    path.rename(partial)
    fake_command(checkout, 'curl', 'exit 22')
    env = dict(os.environ, PATH=str(checkout / 'fake-bin') + ':' + os.environ['PATH'])
    result = install(checkout, 'models', env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert path.is_file()
    assert not partial.exists()
    assert (checkout / 'tts/moss-tts-nano' / MOSS_MODEL / 'prefill.onnx').is_file()


def test_robot_dependency_installer_rejects_non_root_before_system_mutation():
    if os.geteuid() == 0:
        pytest.skip('non-root rejection only')
    result = subprocess.run(['bash', str(ROOT / 'scripts/robot_dependencies.sh'), 'install'],
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 2
    assert 'robot_dependencies.sh install' in result.stderr


def test_optional_asr_build_does_not_require_sdk_audio_or_python_binding_dependencies(checkout):
    log = checkout / 'cmake.log'
    fake_command(checkout, 'cmake', 'printf "%s\\n" "$*" >> "$FAKE_CMAKE_LOG"')
    env = dict(os.environ, PATH=str(checkout / 'fake-bin') + ':' + os.environ['PATH'],
               FAKE_CMAKE_LOG=str(log))
    result = install(checkout, 'build', '--with-asr', env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    commands = log.read_text().splitlines()
    command = next(line for line in commands if line.startswith('-S asr/sensevoice '))
    for flag in ('-DBUILD_STREAM_DEMO=OFF', '-DBUILD_PYTHON_BINDINGS=OFF', '-DASR_MODEL_FETCH_OFF=ON'):
        assert flag in command.split()
