#!/usr/bin/env python3
"""Qwen3-TTS K3 low-latency interactive streaming player.

The current server returns one complete WAV per request rather than streaming
PCM frames.  This client therefore implements ordered segment-level streaming:
it splits the text, prefetches a small bounded window of segments in parallel,
starts playback as soon as the first segment is ready, and feeds later PCM
segments to one long-lived player in text order.  No WAV file is written.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass

HOST = os.getenv("QWEN3_TTS_HOST", "127.0.0.1")
PORT = int(os.getenv("QWEN3_TTS_PORT", "18080"))
BASE_URL = f"http://{HOST}:{PORT}"
SPEECH_URL = f"{BASE_URL}/v1/audio/speech"
MAX_CHARS = max(8, int(os.getenv("QWEN3_TTS_CHUNK_CHARS", "24")))
PLAYER = os.getenv("QWEN3_TTS_PLAYER", "aplay")
PLAYBACK_DEVICE = os.getenv("QWEN3_TTS_PLAYBACK_DEVICE", "default")
# The board measured approximately RTF 1.12-1.45 for short utterances.  The
# default is intentionally low-latency: a first segment is played immediately
# instead of waiting for enough audio to hide the whole RTF>1 deficit.
STREAM_RTF_HINT = max(0.1, float(os.getenv("QWEN3_TTS_STREAM_RTF_HINT", "1.20")))
STREAM_RTF_SAFETY = max(1.0, float(os.getenv("QWEN3_TTS_STREAM_RTF_SAFETY", "1.00")))
STREAM_MIN_BUFFER = max(0.0, float(os.getenv("QWEN3_TTS_STREAM_MIN_BUFFER", "0.35")))
STREAM_BUFFER_MARGIN = max(0.0, float(os.getenv("QWEN3_TTS_STREAM_BUFFER_MARGIN", "0.10")))
PLAYBACK_BUFFER_TIME_US = max(0, int(os.getenv("QWEN3_TTS_PLAYBACK_BUFFER_US", "250000")))
PLAYBACK_PERIOD_TIME_US = max(0, int(os.getenv("QWEN3_TTS_PLAYBACK_PERIOD_US", "30000")))
PLAYBACK_START_DELAY_US = max(0, int(os.getenv("QWEN3_TTS_PLAYBACK_START_DELAY_US", "0")))


def env_enabled(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


STREAM_PLAY_DEFAULT = env_enabled("QWEN3_TTS_STREAM_PLAY", True)
STREAM_LOW_LATENCY = env_enabled("QWEN3_TTS_STREAM_LOW_LATENCY", True)
# Generate a small ordered window of future segments in parallel.  The client
# still feeds the player in text order, while the next request can overlap
# playback of the current segment.  Two workers are a conservative default
# for the K3 runtime; set this to 1 to restore strictly sequential requests.
PREFETCH_CONCURRENCY = max(1, int(os.getenv("QWEN3_TTS_PREFETCH_CONCURRENCY", "2")))


@dataclass
class AudioChunk:
    wav: bytes
    # Wall-clock time observed by this client for one HTTP request. With
    # prefetch enabled this can include server queueing and HTTP overhead.
    client_request_seconds: float
    # Always calculated from the returned WAV duration, so this is the
    # user-visible request RTF rather than the server's independent metric.
    client_request_rtf: float | None
    # Optional server-side metrics returned in X-TTS-* headers.
    audio_seconds: float | None
    wall_seconds: float | None
    server_rtf: float | None


@dataclass
class DecodedChunk:
    frames: bytes
    frame_count: int
    duration: float
    audio_format: tuple[int, int, int, str]


class StreamingPlayer:
    """Feed raw PCM chunks to one long-lived aplay process in a worker thread."""

    _SAMPLE_FORMATS = {
        1: "U8",
        2: "S16_LE",
        3: "S24_3LE",
        4: "S32_LE",
    }

    def __init__(self, audio_format: tuple[int, int, int, str]) -> None:
        channels, sample_width, sample_rate, compression = audio_format
        if compression != "NONE":
            raise RuntimeError(f"播放器不支持 WAV 压缩格式：{compression}")
        sample_format = self._SAMPLE_FORMATS.get(sample_width)
        if sample_format is None:
            raise RuntimeError(f"播放器不支持 {sample_width * 8}-bit PCM")
        executable = shutil.which(PLAYER)
        if executable is None:
            raise RuntimeError(f"找不到播放器：{PLAYER}")

        self.command = [
            executable,
            "-q",
            "-D",
            PLAYBACK_DEVICE,
            "-t",
            "raw",
            "-f",
            sample_format,
            "-c",
            str(channels),
            "-r",
            str(sample_rate),
        ]
        # A smaller device buffer lowers latency, but the board's PipeWire
        # bridge can underrun when requests arrive in bursts.  250 ms / 30 ms
        # is a safer low-latency compromise than the original ~500 ms buffer.
        if PLAYBACK_BUFFER_TIME_US:
            self.command += ["--buffer-time", str(PLAYBACK_BUFFER_TIME_US)]
        if PLAYBACK_PERIOD_TIME_US:
            self.command += ["--period-time", str(PLAYBACK_PERIOD_TIME_US)]
        if PLAYBACK_START_DELAY_US:
            self.command += ["--start-delay", str(PLAYBACK_START_DELAY_US)]
        # Bound queued PCM so a fast producer cannot add unbounded latency or
        # memory while the audio device is momentarily busy.  One segment is
        # enough for low-latency playback; the writer thread blocks the
        # producer when this queue is full, which naturally applies backpressure.
        queue_limit = max(1, int(os.getenv("QWEN3_TTS_PLAYBACK_QUEUE_SEGMENTS", "2")))
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=queue_limit)
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self.error: str | None = None
        self.warning: str | None = None

    @property
    def started(self) -> bool:
        return self._thread is not None

    def feed(self, frames: bytes) -> None:
        if frames:
            self._queue.put(frames)

    def start(self) -> None:
        if self.started:
            return
        self._thread = threading.Thread(target=self._run, name="qwen3-tts-player", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        process: subprocess.Popen[bytes] | None = None
        try:
            process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            with self._lock:
                self._process = process
            assert process.stdin is not None
            while True:
                frames = self._queue.get()
                if frames is None:
                    break
                process.stdin.write(frames)
                process.stdin.flush()
            process.stdin.close()
            stderr = process.stderr.read() if process.stderr is not None else b""
            return_code = process.wait()
            detail = stderr.decode("utf-8", errors="replace").strip()
            if return_code != 0:
                self.error = f"播放器退出码 {return_code}" + (f"：{detail}" if detail else "")
            elif detail:
                self.warning = detail
        except (BrokenPipeError, OSError) as exc:
            if process is not None:
                detail = ""
                if process.stderr is not None:
                    try:
                        detail = process.stderr.read().decode("utf-8", errors="replace").strip()
                    except OSError:
                        pass
                self.error = "播放器管道提前关闭" + (f"：{detail}" if detail else f"（{exc}）")
            else:
                self.error = "播放器管道提前关闭"
        except Exception as exc:
            self.error = str(exc)
        finally:
            with self._lock:
                self._process = None

    def finish(self) -> None:
        if not self.started:
            self.start()
        self._queue.put(None)
        assert self._thread is not None
        self._thread.join()

    def abort(self) -> None:
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        if self.started:
            self._queue.put(None)
            assert self._thread is not None
            self._thread.join(timeout=3)


def health_ok() -> bool:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def split_text(text: str) -> list[str]:
    """Split at natural punctuation; never send an unfinished fragment to TTS."""
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []

    sentences = [part.strip() for part in re.findall(r".*?(?:[。！？!?；;\n]+|$)", text) if part.strip()]
    chunks: list[str] = []
    terminal = re.compile(r"[。！？!?；;，,：:.]$")

    def finish(part: str) -> str:
        part = part.strip()
        if not part or terminal.search(part):
            return part
        # An explicit terminator helps Qwen3-TTS emit EOS reliably.
        return part + ("。" if re.search(r"[\u3400-\u9fff]", part) else ".")

    for sentence in sentences:
        if len(sentence) <= MAX_CHARS:
            chunks.append(finish(sentence))
            continue

        # Prefer commas/colons as low-latency clause boundaries. The delimiter is retained.
        natural = [part.strip() for part in re.findall(r".*?(?:[，,：:]+|$)", sentence) if part.strip()]
        if len(natural) > 1:
            chunks.extend(finish(part) for part in natural if finish(part))
            continue

        # Last resort for a very long unpunctuated sentence: split Chinese by characters
        # and English by complete words, then add a terminator to every chunk.
        if " " in sentence:
            current: list[str] = []
            current_len = 0
            for word in sentence.split():
                extra = len(word) + (1 if current else 0)
                if current and current_len + extra > MAX_CHARS:
                    chunks.append(finish(" ".join(current)))
                    current = [word]
                    current_len = len(word)
                else:
                    current.append(word)
                    current_len += extra
            if current:
                chunks.append(finish(" ".join(current)))
        else:
            chunks.extend(finish(sentence[i : i + MAX_CHARS]) for i in range(0, len(sentence), MAX_CHARS))
    return chunks


def synthesize(text: str) -> AudioChunk:
    body = json.dumps(
        {
            "model": "qwen3-tts",
            "input": text,
            "voice": "default",
            "response_format": "wav",
            "speed": 1.0,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        SPEECH_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            wav = response.read()
            headers = response.headers
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    elapsed = time.monotonic() - started

    def header_float(name: str) -> float | None:
        value = headers.get(name)
        try:
            return float(value) if value is not None else None
        except ValueError:
            return None

    if len(wav) < 44 or wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
        raise RuntimeError("服务端没有返回有效的 WAV 音频")
    audio_seconds = header_float("X-TTS-Audio-Seconds")
    wall_seconds = header_float("X-TTS-Wall-Seconds")
    server_rtf = header_float("X-TTS-RTF")

    # Do not use X-TTS-Audio-Seconds for the client metric: the displayed
    # request RTF must be derived from the actual WAV that this request
    # returned. This also makes the fallback correct for older servers.
    with wave.open(io.BytesIO(wav), "rb") as wav_file:
        returned_audio_seconds = wav_file.getnframes() / wav_file.getframerate()
    client_rtf = (elapsed / returned_audio_seconds) if returned_audio_seconds > 0 else None
    return AudioChunk(
        wav=wav,
        client_request_seconds=elapsed,
        client_request_rtf=client_rtf,
        audio_seconds=audio_seconds,
        wall_seconds=wall_seconds,
        server_rtf=server_rtf,
    )


def iter_synthesized(chunks: list[str]):
    """Yield completed segments in text order with a bounded request window.

    The HTTP endpoint returns a complete WAV per request, so this cannot make
    one request PCM-streaming.  It does, however, overlap the next segment's
    server work with playback of the current segment.  Results are yielded in
    order even when a later request finishes first.
    """
    if PREFETCH_CONCURRENCY <= 1 or len(chunks) <= 1:
        for chunk_text in chunks:
            yield synthesize(chunk_text)
        return

    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=PREFETCH_CONCURRENCY,
        thread_name_prefix="qwen3-tts-prefetch",
    )
    pending: dict[int, concurrent.futures.Future[AudioChunk]] = {}
    next_submit = 0
    try:
        while next_submit < min(PREFETCH_CONCURRENCY, len(chunks)):
            pending[next_submit] = executor.submit(synthesize, chunks[next_submit])
            next_submit += 1

        for index in range(len(chunks)):
            # Waiting for this exact index preserves speech order.  As soon as
            # it is consumed, refill the bounded window with the next request.
            audio = pending.pop(index).result()
            if next_submit < len(chunks):
                pending[next_submit] = executor.submit(synthesize, chunks[next_submit])
                next_submit += 1
            yield audio
    except BaseException:
        for future in pending.values():
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)


def decode_chunk(audio: AudioChunk) -> DecodedChunk:
    with wave.open(io.BytesIO(audio.wav), "rb") as reader:
        audio_format = (
            reader.getnchannels(),
            reader.getsampwidth(),
            reader.getframerate(),
            reader.getcomptype(),
        )
        if audio_format[3] != "NONE":
            raise RuntimeError(f"不支持的 WAV 压缩格式：{audio_format[3]}")
        frame_count = reader.getnframes()
        frames = reader.readframes(frame_count)
    return DecodedChunk(
        frames=frames,
        frame_count=frame_count,
        duration=frame_count / audio_format[2],
        audio_format=audio_format,
    )


def text_weight(text: str) -> int:
    """Return a rough speech-duration weight without punctuation/whitespace."""
    return max(1, len(re.sub(r"[\W_]+", "", text, flags=re.UNICODE)))


def playback_start_plan(
    chunks: list[str],
    generated_count: int,
    buffered_seconds: float,
    generated_weight: int,
    observed_rtf: float | None,
) -> tuple[bool, float, float, float]:
    """Estimate whether the prebuffer can cover future slower-than-real-time work.

    The server only releases complete segment WAVs.  Besides the long-run
    ``(RTF - 1) * remaining_audio`` deficit, the buffer must cover the full
    synthesis latency of the next segment because no PCM from that segment is
    available until its HTTP response finishes.
    """
    if generated_count >= len(chunks):
        return True, 0.0, 0.0, max(STREAM_RTF_HINT, observed_rtf or 0.0)

    seconds_per_weight = buffered_seconds / max(1, generated_weight)
    remaining_weights = [text_weight(chunk) for chunk in chunks[generated_count:]]
    estimated_durations = [seconds_per_weight * weight for weight in remaining_weights]
    estimated_remaining = sum(estimated_durations)
    guarded_rtf = max(STREAM_RTF_HINT, observed_rtf or 0.0) * STREAM_RTF_SAFETY

    # Each HTTP request returns a whole WAV at once.  For every future segment,
    # ensure the existing buffer can survive until that complete response
    # arrives.  This prefix calculation is stricter than merely checking the
    # total long-run deficit and avoids a late underrun when RTF > 1.
    prefix_audio = 0.0
    prefix_required = 0.0
    for duration in estimated_durations:
        need_until_arrival = guarded_rtf * duration + max(0.0, guarded_rtf - 1.0) * prefix_audio
        prefix_required = max(prefix_required, need_until_arrival)
        prefix_audio += duration

    required = max(STREAM_MIN_BUFFER, prefix_required) + STREAM_BUFFER_MARGIN
    return buffered_seconds >= required, required, estimated_remaining, guarded_rtf


def stream_speech(text: str, *, play: bool = STREAM_PLAY_DEFAULT) -> None:
    """Synthesize text segments and play them with the smallest practical delay.

    The HTTP API still returns one complete WAV per segment.  In low-latency
    mode the first decoded segment starts playback as soon as it is available;
    this deliberately accepts a short gap when the measured RTF is above 1.
    Set ``QWEN3_TTS_STREAM_LOW_LATENCY=0`` to restore the conservative
    RTF-aware prebuffer calculation.
    """
    chunks = split_text(text)
    if not chunks:
        raise ValueError("输入文字为空")

    started = time.monotonic()
    expected_format: tuple[int, int, int, str] | None = None
    total_frames = 0
    total_client_request_seconds = 0.0
    total_server_wall_seconds = 0.0
    total_header_audio_seconds = 0.0
    has_server_wall = False
    has_header_audio = False
    generated_weight = 0
    player: StreamingPlayer | None = None
    playback_started = False
    playback_start_after = 0.0

    if not play:
        mode = "只合成、不播放"
    elif STREAM_LOW_LATENCY:
        mode = "低延迟流式播放（首段就绪即播放）"
    else:
        mode = "RTF 动态预缓冲流式播放"
    print(f"已分成 {len(chunks)} 段，开始合成（{mode}）……", flush=True)

    try:
        for index, (chunk_text, audio) in enumerate(zip(chunks, iter_synthesized(chunks)), start=1):
            decoded = decode_chunk(audio)
            total_client_request_seconds += audio.client_request_seconds
            if audio.wall_seconds is not None and audio.wall_seconds >= 0:
                total_server_wall_seconds += audio.wall_seconds
                has_server_wall = True
            if audio.audio_seconds is not None and audio.audio_seconds > 0:
                total_header_audio_seconds += audio.audio_seconds
                has_header_audio = True

            current_format = decoded.audio_format
            if expected_format is None:
                expected_format = current_format
                if play:
                    player = StreamingPlayer(current_format)
            elif current_format != expected_format:
                raise RuntimeError(
                    f"第 {index} 段 WAV 格式与第一段不一致："
                    f"{current_format} != {expected_format}"
                )

            total_frames += decoded.frame_count
            generated_weight += text_weight(chunk_text)
            if player is not None:
                player.feed(decoded.frames)

            client_rtf_text = (
                f"客户端请求RTF={audio.client_request_rtf:.2f}"
                if audio.client_request_rtf is not None
                else "客户端请求RTF=未知"
            )
            server_rtf_text = (
                f"服务端RTF={audio.server_rtf:.2f}"
                if audio.server_rtf is not None
                else "服务端RTF=未知"
            )
            print(
                f"第 {index}/{len(chunks)} 段合成完成：HTTP请求耗时 {audio.client_request_seconds:.2f} 秒"
                f"；音频时长 {decoded.duration:.2f} 秒；{client_rtf_text}；{server_rtf_text}"
                f"（RTF越小越实时）",
                flush=True,
            )

            if player is not None and not playback_started:
                buffered_seconds = total_frames / current_format[2]
                if STREAM_LOW_LATENCY:
                    # A complete segment is the smallest unit exposed by the
                    # current HTTP API. Do not wait for extra seconds here: the
                    # first segment is already a usable playback buffer.
                    should_start = index >= 1
                    required = STREAM_MIN_BUFFER
                    estimated_remaining = 0.0
                    guarded_rtf = max(
                        STREAM_RTF_HINT,
                        audio.server_rtf or audio.client_request_rtf or 0.0,
                    ) * STREAM_RTF_SAFETY
                else:
                    if has_server_wall and has_header_audio and total_header_audio_seconds > 0:
                        observed_rtf = total_server_wall_seconds / total_header_audio_seconds
                    elif total_frames > 0:
                        # With concurrent prefetch, summing individual HTTP
                        # durations overstates the wall-clock wait. At this
                        # point playback has not started yet, so elapsed wall
                        # time is the correct client-side fallback.
                        client_elapsed = time.monotonic() - started
                        observed_rtf = client_elapsed / buffered_seconds
                    else:
                        observed_rtf = None
                    should_start, required, estimated_remaining, guarded_rtf = playback_start_plan(
                        chunks,
                        index,
                        buffered_seconds,
                        generated_weight,
                        observed_rtf,
                    )

                if should_start:
                    player.start()
                    playback_started = True
                    playback_start_after = time.monotonic() - started
                    if STREAM_LOW_LATENCY:
                        print(
                            f"开始低延迟播放：首段已就绪，已缓存 {buffered_seconds:.2f} 秒音频；"
                            f"播放缓冲目标 {required:.2f} 秒；后续分段将边生成边播放",
                            flush=True,
                        )
                    else:
                        print(
                            f"开始流式播放：已预缓冲 {buffered_seconds:.2f} 秒音频；"
                            f"观测/保护 RTF={observed_rtf or STREAM_RTF_HINT:.2f}/{guarded_rtf:.2f}；"
                            f"估计剩余音频 {estimated_remaining:.2f} 秒",
                            flush=True,
                        )
                elif not STREAM_LOW_LATENCY:
                    print(
                        f"继续预缓冲：已有 {buffered_seconds:.2f} 秒，"
                        f"当前估计需要 {required:.2f} 秒；保护 RTF={guarded_rtf:.2f}",
                        flush=True,
                    )

        if expected_format is None:
            raise RuntimeError("没有生成音频")
        duration = total_frames / expected_format[2]
        synthesis_elapsed = time.monotonic() - started

        if player is not None:
            if not playback_started:
                player.start()
                playback_started = True
                playback_start_after = synthesis_elapsed
                print(f"开始播放：全部 {duration:.2f} 秒音频已经生成完成", flush=True)
            player.finish()
            if player.error:
                raise RuntimeError(f"播放失败：{player.error}")
            if player.warning:
                print(f"播放器提示：{player.warning}", file=sys.stderr, flush=True)
            print("播放完成。", flush=True)

        elapsed = time.monotonic() - started
        if has_server_wall and has_header_audio and total_header_audio_seconds > 0:
            server_total_rtf = total_server_wall_seconds / total_header_audio_seconds
        else:
            server_total_rtf = None
        # This is the end-to-end generation-stage wall-clock RTF. Do not sum
        # per-request times here: with prefetch workers that would count
        # overlapping requests more than once.
        client_total_rtf = (synthesis_elapsed / duration) if duration > 0 else None
        client_total_rtf_text = f"{client_total_rtf:.2f}" if client_total_rtf is not None else "未知"
        server_total_rtf_text = f"{server_total_rtf:.2f}" if server_total_rtf is not None else "未知"
        print(
            f"音频时长：{duration:.2f} 秒；合成阶段墙钟耗时：{synthesis_elapsed:.2f} 秒；"
            f"播放结束耗时：{elapsed:.2f} 秒；首播延迟：{playback_start_after:.2f} 秒",
            flush=True,
        )
        print(
            f"本轮客户端墙钟RTF：{client_total_rtf_text}；"
            f"本轮服务端统计RTF：{server_total_rtf_text}"
            f"（RTF越小越实时；RTF<1 表示快于实时）",
            flush=True,
        )
        if PREFETCH_CONCURRENCY > 1:
            print(
                f"说明：各段HTTP请求耗时相加为 {total_client_request_seconds:.2f} 秒；"
                "并发预取时包含重叠等待，仅作诊断，不作为本轮墙钟RTF。",
                flush=True,
            )
    except Exception:
        if player is not None:
            player.abort()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="在 K3 上分段合成 Qwen3-TTS，并以低延迟方式直接播放，不保存 WAV。"
    )
    playback = parser.add_mutually_exclusive_group()
    playback.add_argument(
        "--play",
        action="store_true",
        help="直接播放（默认）",
    )
    playback.add_argument(
        "--no-play",
        action="store_true",
        help="仅合成、不播放；用于测速，不保存 WAV",
    )
    parser.add_argument("text", nargs="*", help="单次合成文本；省略时进入交互模式")
    args = parser.parse_args()
    play = STREAM_PLAY_DEFAULT
    if args.play:
        play = True
    elif args.no_play:
        play = False

    if not health_ok():
        print("Qwen3-TTS 服务未就绪，请先运行 ./start_server.sh", file=sys.stderr)
        return 1

    if args.text:
        try:
            stream_speech(" ".join(args.text), play=play)
            return 0
        except Exception as exc:
            print(f"合成或播放失败：{exc}", file=sys.stderr)
            return 1

    print("Qwen3-TTS 交互式低延迟流式播放模式")
    print("- 输入中文、英文或中英混合文字，回车后分段生成并直接播放")
    print("- 不保存 WAV；输入 quit / exit / 退出 可结束")
    print(f"- 播放缓冲：{PLAYBACK_BUFFER_TIME_US / 1000:.0f} ms；周期：{PLAYBACK_PERIOD_TIME_US / 1000:.0f} ms")
    print(f"- 分段上限：{MAX_CHARS} 字符；首段就绪后目标缓冲：{STREAM_MIN_BUFFER:.2f} 秒")
    print(f"- 低延迟模式：{'是' if STREAM_LOW_LATENCY else '否'}")
    print(f"- 并行预取：{PREFETCH_CONCURRENCY} 段（按原顺序播放；设 QWEN3_TTS_PREFETCH_CONCURRENCY=1 可关闭）")

    while True:
        try:
            text = input("\n请输入文字> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in {"quit", "exit"} or text == "退出":
            break
        try:
            stream_speech(text, play=play)
        except KeyboardInterrupt:
            print("\n本轮已中断", file=sys.stderr)
        except Exception as exc:
            print(f"合成或播放失败：{exc}", file=sys.stderr)
    print("已退出。后台服务仍在运行；停止服务请执行 ./stop_server.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
