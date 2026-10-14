#!/usr/bin/env python3
"""Check delivered assets without loading models or opening any device."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def configuration(root, engine):
    component = {'moss': 'tts_moss_nano', 'matcha': 'tts_matcha',
                 'sensevoice': 'asr_sensevoice'}[engine]
    return json.loads((root / f'backend/components/{component}/config.json').read_text())


def asset_root(root, config):
    runtime = config['runtime']
    return (root / runtime.get('root', runtime.get('working_dir'))).resolve()


def missing_assets(root, engine):
    config = configuration(root, engine)
    base = asset_root(root, config)
    missing = []

    def require(path):
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(str(path))
            return False
        return True

    def require_glob(pattern):
        if not any(p.is_file() and p.stat().st_size for p in base.glob(pattern)):
            missing.append(str(base / pattern))

    def metadata(path):
        if not require(path):
            return None
        return json.loads(path.read_text())

    runtime = config['runtime']
    if engine == 'moss':
        # A tracked tokenizer/manifest creates the directory on a fresh clone.
        # Check its payload, including codec external data, instead of the dir.
        for directory in ('python', 'lib', 'voice', 'assets'):
            if not (base / directory).is_dir():
                missing.append(str(base / directory))
        for relative in ('python/spacemit_ort/__init__.py', 'python/scipy/__init__.py',
                         'python/sentencepiece/__init__.py', 'python/soundfile.py',
                         'python/cffi/__init__.py', 'lib/libspacemit_ep.so', 'lib/libspert.so.1'):
            require(base / relative)
        for pattern in ('python/_cffi_backend*.so', 'python/sentencepiece/_sentencepiece*.so',
                        'python/spacemit_ort/libspacemit_ep.so*'):
            require_glob(pattern)
        voice = config.get('voice', {})
        if voice.get('mode', 'builtin') == 'clone':
            require(base / voice['reference_audio'])
        model = base / runtime['model_dir']
        manifest = metadata(model / 'browser_poc_manifest.json')
        if manifest is not None:
            files = manifest['model_files']
            require(model / files['tokenizer_model'])
            for key in ('tts_meta', 'codec_meta'):
                path = (model / files[key]).resolve()
                meta = metadata(path)
                if meta is not None:
                    # Fail malformed metadata rather than treating it as an
                    # empty, complete collection of weights.
                    required_keys = (('prefill', 'decode_step', 'local_fixed_sampled_frame')
                                     if key == 'tts_meta' else ('encode', 'decode_full'))
                    for field in required_keys:
                        require(path.parent / meta['files'][field])
                    for filenames in meta.get('external_data_files', {}).values():
                        for filename in filenames:
                            require(path.parent / filename)
    elif engine == 'matcha':
        require(base / runtime['binary'])
        for filename in ('model-steps-3.q.onnx', 'vocos-16khz-univ.q.onnx',
                         'lexicon.txt', 'tokens.txt'):
            require(base / runtime['model_dir'] / filename)
        for filename in ('libsherpa-onnx-c-api.so', 'libonnxruntime.so.1'):
            require(base / runtime['sherpa_lib_dir'] / filename)
    else:
        model = (root / runtime['model_dir']).resolve()
        for filename in ('model_quant_optimized.onnx', 'tokens.txt'):
            require(model / filename)
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('engine', choices=('moss', 'matcha', 'sensevoice'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--path', action='store_true', help='Print the configured asset root')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    try:
        if args.path:
            print(asset_root(args.root, configuration(args.root, args.engine)))
            return 0
        missing = missing_assets(args.root, args.engine)
        if missing:
            if not args.quiet:
                print('Missing or empty ' + args.engine + ' assets:\n  ' + '\n  '.join(missing), file=sys.stderr)
            return 1
        if not args.quiet:
            print(args.engine + ' assets OK')
        return 0
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        if not args.quiet:
            print(f'{args.engine} asset check failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
