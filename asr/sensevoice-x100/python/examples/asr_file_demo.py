#!/usr/bin/env python3
"""
ASR 静态文件识别示例

只依赖 spacemit_asr 模块，用于识别音频文件。

Usage:
    python asr_file_demo.py audio.wav
    python asr_file_demo.py --list-backends
"""

import sys
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='ASR 静态文件识别')
    parser.add_argument('file', nargs='?', help='要识别的音频文件（例如 WAV 或 MP3）')
    parser.add_argument('--engine', default='sensevoice',
                        choices=['sensevoice', 'zipformer', 'funasr', 'qwen3-asr',
                                 'gemma4-asr'],
                        help='识别引擎 (默认: sensevoice)')
    parser.add_argument('--model-dir', '-m', default=None,
                        help='本地模型目录 (SenseVoice/Zipformer 可选)')
    parser.add_argument('--language', '-l', default='auto',
                        choices=['zh', 'en', 'ja', 'ko', 'yue', 'auto'],
                        help='识别语言 (默认: auto)')
    parser.add_argument('--list-backends', action='store_true',
                        help='列出可用后端')
    parser.add_argument('--provider', '-p', default='spacemit',
                        choices=['cpu', 'spacemit'],
                        help='执行提供程序 (默认: spacemit)')
    parser.add_argument('--rounds', '-r', type=int, default=1,
                        help='重复识别轮次 (默认: 1)')
    parser.add_argument('--enable-emotion', action='store_true',
                        help='启用 SenseVoice 情绪识别 (默认关闭)')
    parser.add_argument('--endpoint', default=None,
                        help='llama-server URL')
    parser.add_argument('--model', default=None,
                        help='llama-server model tag')
    parser.add_argument('--timeout', type=int, default=60,
                        help='llama-server 请求超时秒数 (默认: 60)')
    parser.add_argument('--task', default='transcribe',
                        choices=['transcribe', 'translate'],
                        help='Gemma4 任务 (默认: transcribe)')

    args = parser.parse_args()

    # 导入 spacemit_asr
    try:
        import spacemit_asr
    except ImportError as e:
        print("错误: 无法导入 spacemit_asr 模块")
        print("请确保已设置 PYTHONPATH:")
        print("  export PYTHONPATH=/path/to/asr/build/python:/path/to/asr/python")
        print(f"\n详细错误: {e}")
        sys.exit(1)

    print(f"ASR 版本: {spacemit_asr.__version__}")

    if args.list_backends:
        backends = ", ".join(
            str(backend) for backend in spacemit_asr.Engine.get_available_backends())
        print(f"可用后端: {backends}")
        return

    if not args.file:
        parser.print_help()
        return

    if not os.path.exists(args.file):
        print(f"错误: 文件不存在 - {args.file}")
        sys.exit(1)

    # 配置
    lang_map = {
        'zh': spacemit_asr.Language.ZH,
        'en': spacemit_asr.Language.EN,
        'ja': spacemit_asr.Language.JA,
        'ko': spacemit_asr.Language.KO,
        'yue': spacemit_asr.Language.YUE,
        'auto': spacemit_asr.Language.AUTO,
    }

    if args.engine == 'sensevoice':
        config = spacemit_asr.Config.sensevoice(
            args.model_dir or '~/.cache/models/asr/sensevoice',
            enable_emotion=args.enable_emotion)
    elif args.engine == 'zipformer':
        config = spacemit_asr.Config.zipformer(
            args.model_dir or '~/.cache/models/asr/zipformer')
    elif args.engine == 'funasr':
        config = spacemit_asr.Config.funasr(
            endpoint=args.endpoint or
            'http://127.0.0.1:8063/v1/audio/transcriptions',
            model=args.model or 'funasr',
            timeout=args.timeout)
    elif args.engine == 'qwen3-asr':
        config = spacemit_asr.Config.qwen3_asr(
            endpoint=args.endpoint or
            'http://127.0.0.1:8063/v1/chat/completions',
            model=args.model or 'qwen3-asr',
            timeout=args.timeout)
    else:
        config = spacemit_asr.Config.gemma4_asr(
            endpoint=args.endpoint or
            'http://127.0.0.1:8063/v1/audio/transcriptions',
            model=args.model or 'gemma4-asr',
            timeout=args.timeout,
            task=args.task)

    config.language = lang_map[args.language]
    config.punctuation_enabled = True
    config.provider = args.provider
    if args.engine == 'sensevoice':
        config.enable_emotion = args.enable_emotion

    # 识别
    print(f"\n文件: {args.file}")
    print(f"引擎: {args.engine}")
    print(f"语言: {args.language}")
    if args.engine == 'gemma4-asr':
        print(f"任务: {args.task}")
    print(f"Provider: {args.provider}")
    print(f"情绪识别: {'启用' if args.enable_emotion else '禁用'}")
    print("正在识别...\n")

    with spacemit_asr.Engine(config) as engine:
        import time
        import numpy as np
        if args.engine not in ('funasr', 'qwen3-asr', 'gemma4-asr'):
            print(">>> Warmup...")
            silence = np.zeros(8000, dtype=np.float32)
            t0 = time.monotonic()
            engine.recognize(silence)
            warmup_ms = (time.monotonic() - t0) * 1000
            print(f"Warmup done: {warmup_ms:.0f} ms\n")

        total_audio_ms = 0
        total_proc_ms = 0
        for round_idx in range(args.rounds):
            result = engine.recognize_file(args.file)
            total_audio_ms += result.audio_duration_ms
            total_proc_ms += result.processing_time_ms
            print(f"[轮次 {round_idx + 1}/{args.rounds}] {result.text}")
            if result.emotion:
                print(f"情绪: {result.emotion}")

        if args.rounds > 1:
            print(f"\n--- 汇总 ({args.rounds} 轮) ---")
            print(f"平均音频时长: {total_audio_ms // args.rounds} ms")
            print(f"平均处理时间: {total_proc_ms // args.rounds} ms")
            print(f"平均 RTF: {total_proc_ms / max(total_audio_ms, 1):.3f}")
        else:
            print(f"音频时长: {result.audio_duration_ms} ms")
            print(f"处理时间: {result.processing_time_ms} ms")
            print(f"RTF: {result.rtf:.3f}")


if __name__ == "__main__":
    main()
