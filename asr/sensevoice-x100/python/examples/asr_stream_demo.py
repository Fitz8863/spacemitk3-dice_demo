#!/usr/bin/env python3
"""
ASR 流式识别示例 - 边录边识别 (定时 flush)

使用 multiprocessing 避免 GIL 导致的 overrun 问题。
音频采集在独立进程中运行，通过 multiprocessing.Queue 传递数据。

依赖:
    1. spacemit_asr 模块 (asr/build/python)
    2. spacemit_audio 模块 (audio/python)

编译 audio 组件:
    cd audio && mkdir -p build && cd build
    cmake .. && make -j$(nproc)

Usage:
    python asr_stream_demo.py                 # 默认设备，持续识别
    python asr_stream_demo.py -l              # 列出音频设备
    python asr_stream_demo.py -d 0            # 使用设备 0
    python asr_stream_demo.py --duration 10   # 录制 10 秒
    python asr_stream_demo.py --flush 5       # 每 5 秒 flush
"""

import sys
import time
import argparse
import multiprocessing as mp
from multiprocessing import Process, Queue
import numpy as np

# 导入 spacemit_asr
try:
    import spacemit_asr
    from spacemit_asr import AsrCallback
except ImportError as e:
    print("错误: 无法导入 spacemit_asr 模块")
    print("请确保已设置 PYTHONPATH:")
    print("  export PYTHONPATH=/path/to/asr/build/python:/path/to/asr/python")
    print(f"\n详细错误: {e}")
    sys.exit(1)


# =============================================================================
# 音频处理工具
# =============================================================================

def resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """重采样音频 (简单线性下采样，适合实时处理)"""
    if src_rate == dst_rate:
        return audio

    # 使用简单的线性插值，速度快适合实时
    ratio = dst_rate / src_rate
    new_len = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def stereo_to_mono(audio: np.ndarray, channels: int) -> np.ndarray:
    """多声道混音为单声道"""
    if channels == 1:
        return audio
    # 假设交错格式: L R L R ...
    audio = audio.reshape(-1, channels)
    return audio.mean(axis=1).astype(np.float32)


# =============================================================================
# 自定义回调类
# =============================================================================

class StreamingCallback(AsrCallback):
    """流式识别回调 - 实时打印识别结果"""

    def __init__(self, show_emotion: bool = False):
        super().__init__()
        self.show_emotion = show_emotion
        self.final_results = []
        self.sentence_count = 0

    def on_open(self) -> None:
        print(">>> [回调] 识别会话开始")

    def on_event(self, result) -> None:
        # 获取文本
        text = getattr(result, 'text', '') or result.get_text() if hasattr(result, 'get_text') else str(result)

        if text:
            self.sentence_count += 1
            print(f"[句子 {self.sentence_count}]: {text}")
            emotion = getattr(result, 'emotion', None)
            if self.show_emotion and emotion:
                print(f"[情绪 {self.sentence_count}]: {emotion}")
            self.final_results.append(text)

    def on_complete(self) -> None:
        print("\n>>> [回调] 识别任务完成")

    def on_error(self, result) -> None:
        msg = getattr(result, 'message', str(result))
        print(f"\n>>> [回调] 错误: {msg}")

    def on_close(self) -> None:
        print(">>> [回调] 会话关闭")


def list_devices():
    """列出音频设备"""
    try:
        import spacemit_audio  # noqa: F401
        from spacemit_audio import AudioCapture
    except ImportError as e:
        print("错误: 无法导入 spacemit_audio 模块")
        print(f"详细错误: {e}")
        return

    print("=== 音频输入设备 ===")
    for idx, name in AudioCapture.list_devices():
        print(f"  [{idx}] {name}")


# =============================================================================
# 音频采集进程
# =============================================================================

def audio_capture_process(audio_queue: Queue, stop_event, device: int,
                          duration: float, config_queue: Queue, channels: int = 1):
    """
    独立进程中运行音频采集

    Args:
        audio_queue: 音频数据队列 (输出)
        stop_event: 停止信号
        device: 音频设备索引
        duration: 录音时长
        config_queue: 配置信息队列 (输出采样率和声道数)
    """
    try:
        import spacemit_audio
        from spacemit_audio import AudioCapture
    except ImportError as e:
        config_queue.put({'error': str(e)})
        return

    try:
        # 初始化音频
        spacemit_audio.init(
            sample_rate=16000,
            channels=channels,
            chunk_size=3200,  # 100ms @ 16kHz
            capture_device=device,
        )

        config = spacemit_audio.get_config()
        input_rate = config['sample_rate']
        input_channels = config['channels']

        # 发送配置信息给主进程
        config_queue.put({
            'sample_rate': input_rate,
            'channels': input_channels,
        })

        frame_count = 0

        def on_audio(data: bytes):
            """音频回调 - 直接放入队列，无 GIL 竞争"""
            nonlocal frame_count
            frame_count += 1
            try:
                # 使用 put_nowait 避免阻塞
                audio_queue.put_nowait(data)
            except Exception:
                pass  # 队列满时丢弃

        with AudioCapture() as cap:
            cap.set_callback(on_audio)
            cap.start()

            print(f"[采集进程] 开始录音 ({duration}秒)...")

            start_time = time.time()
            while time.time() - start_time < duration:
                if stop_event.is_set():
                    break
                time.sleep(0.1)

            print(f"[采集进程] 录音完成，共 {frame_count} 帧")

        # 发送结束标记
        audio_queue.put(None)

    except Exception as e:
        print(f"[采集进程] 错误: {e}")
        config_queue.put({'error': str(e)})
        audio_queue.put(None)


# =============================================================================
# 主函数
# =============================================================================

def run_streaming_recognition(args):
    """运行流式识别 (multiprocessing 版本)"""

    # 创建进程间通信队列
    audio_queue = Queue(maxsize=1000)  # 限制队列大小防止内存溢出
    config_queue = Queue()
    stop_event = mp.Event()

    # 启动音频采集进程
    capture_proc = Process(
        target=audio_capture_process,
        args=(audio_queue, stop_event, args.device, args.duration, config_queue, args.channels)
    )
    capture_proc.start()

    # 等待配置信息
    try:
        config = config_queue.get(timeout=10)
    except Exception:
        print("错误: 无法获取音频配置")
        capture_proc.terminate()
        return

    if 'error' in config:
        print(f"错误: {config['error']}")
        capture_proc.join()
        return

    input_rate = config['sample_rate']
    input_channels = config['channels']
    target_rate = 16000

    print(f"音频配置: {input_rate}Hz, {input_channels}ch")
    need_resample = (input_rate != target_rate)
    need_mix = (input_channels > 1)
    if need_resample:
        print(f"  -> 重采样: {input_rate}Hz → {target_rate}Hz (线性插值)")
    if need_mix:
        print(f"  -> 混音: {input_channels}ch → 1ch")

    # 配置 ASR
    lang_map = {
        'zh': spacemit_asr.Language.ZH,
        'en': spacemit_asr.Language.EN,
        'ja': spacemit_asr.Language.JA,
        'ko': spacemit_asr.Language.KO,
        'yue': spacemit_asr.Language.YUE,
        'auto': spacemit_asr.Language.AUTO,
    }

    asr_config = spacemit_asr.Config(args.model_dir)
    asr_config.language = lang_map[args.language]
    asr_config.punctuation_enabled = True
    asr_config.enable_emotion = args.enable_emotion
    asr_config.provider = args.provider

    print(
        f"ASR 配置: 语言={args.language}, provider={args.provider}, "
        f"emotion={'on' if args.enable_emotion else 'off'}"
    )
    print(f"Flush 间隔: {args.flush}秒")
    print(f"录制时长: {args.duration}秒")
    print("\n使用 multiprocessing 避免 GIL 导致的 overrun")
    print("按 Ctrl+C 退出\n")

    # 创建回调
    callback = StreamingCallback(show_emotion=args.enable_emotion)

    sentence_count = 0
    frame_count = 0

    with spacemit_asr.Engine(asr_config) as engine:
        try:
            print(">>> Warmup...")
            silence = np.zeros(8000, dtype=np.float32)
            t0 = time.monotonic()
            engine.recognize(silence)
            warmup_ms = (time.monotonic() - t0) * 1000
            print(f"Warmup done: {warmup_ms:.0f} ms\n")

            engine.start(callback=callback)

            print("开始处理音频...")
            last_flush_time = time.monotonic()

            while True:
                try:
                    # 从队列获取音频数据 (带超时)
                    data = audio_queue.get(timeout=0.5)

                    if data is None:
                        # 结束标记
                        break

                    frame_count += 1

                    # 转换音频格式
                    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

                    if need_mix:
                        audio = stereo_to_mono(audio, input_channels)
                    if need_resample:
                        audio = resample_audio(audio, input_rate, target_rate)

                    # 发送给 ASR
                    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
                    engine.send_audio_frame(audio_int16.tobytes())

                    # 定时 flush
                    now = time.monotonic()
                    if now - last_flush_time >= args.flush:
                        sentence_count += 1
                        print(f"[句子 {sentence_count}] 定时 flush ({args.flush}s)...")
                        engine.flush()
                        last_flush_time = now

                    # 定期显示进度
                    if frame_count % 50 == 0:
                        print(f"[处理中] 已处理 {frame_count} 帧, 句子: {sentence_count}")

                except mp.queues.Empty:
                    # 队列超时，检查采集进程是否还在运行
                    if not capture_proc.is_alive():
                        break
                    continue

            # 处理剩余
            sentence_count += 1
            print(f"[句子 {sentence_count}] flush (剩余)...")
            engine.flush()

            engine.stop()

            if callback.final_results:
                print("\n=== 识别结果汇总 ===")
                for i, text in enumerate(callback.final_results, 1):
                    print(f"  {i}. {text}")
            else:
                print("\n(无识别结果)")

        except KeyboardInterrupt:
            print("\n\n用户中断")
            stop_event.set()
            engine.stop()

    # 等待采集进程结束
    capture_proc.join(timeout=2)
    if capture_proc.is_alive():
        capture_proc.terminate()


def main():
    parser = argparse.ArgumentParser(
        description='ASR 流式识别 (定时 flush 边录边识别, multiprocessing 版本)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python asr_stream_demo.py                    # 默认设备，持续识别
  python asr_stream_demo.py -l                 # 列出音频设备
  python asr_stream_demo.py -d 0               # 使用设备 0
  python asr_stream_demo.py --duration 30      # 录制 30 秒
  python asr_stream_demo.py --language en      # 英文识别
  python asr_stream_demo.py -f 5               # 每 5 秒 flush
        """
    )
    parser.add_argument('-l', '--list', action='store_true',
                        help='列出音频设备')
    parser.add_argument('-d', '--device', type=int, default=-1,
                        help='音频设备索引 (默认: -1 自动选择)')
    parser.add_argument('--channels', '-c', type=int, default=1,
                        choices=[1, 2],
                        help='采集声道数 (默认: 1)')
    parser.add_argument('--duration', type=float, default=30.0,
                        help='录音时长秒数 (默认: 30.0)')
    parser.add_argument('--model-dir', '-m', default='~/.cache/models/asr/sensevoice',
                        help='模型目录 (默认: ~/.cache/models/asr/sensevoice)')
    parser.add_argument('--language', default='auto',
                        choices=['zh', 'en', 'ja', 'ko', 'yue', 'auto'],
                        help='识别语言 (默认: auto)')
    parser.add_argument('--flush', '-f', type=float, default=3.0,
                        help='Flush 间隔秒数 (默认: 3.0)')
    parser.add_argument('--provider', '-p', default='spacemit',
                        choices=['cpu', 'spacemit'],
                        help='执行提供程序 (默认: spacemit)')
    parser.add_argument('--enable-emotion', action='store_true',
                        help='启用 SenseVoice 情绪识别 (默认关闭)')

    args = parser.parse_args()

    print(f"ASR 版本: {spacemit_asr.__version__}")

    if args.list:
        list_devices()
        return

    run_streaming_recognition(args)


if __name__ == "__main__":
    # multiprocessing 在某些平台需要这个
    mp.set_start_method('spawn', force=True)
    main()
