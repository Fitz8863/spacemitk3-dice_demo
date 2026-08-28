from __future__ import annotations

import hashlib
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


class PcmChunkPlayer:
    """Stream interleaved PCM16 chunks to a persistent ALSA ``aplay`` process."""

    def __init__(self, *, device: str, sample_rate: int, channels: int) -> None:
        self.device = str(device or "default")
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.process: subprocess.Popen[bytes] | None = None
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._writer_thread: threading.Thread | None = None
        self._writer_error: BaseException | None = None

    def start(self) -> None:
        executable = shutil.which("aplay")
        if executable is None:
            raise RuntimeError("PCM playback requires `aplay`; use --no-pcm-playback to disable it")
        command = [
            executable,
            "--quiet",
            "--device",
            self.device,
            "--format",
            "S16_LE",
            "--channels",
            str(self.channels),
            "--rate",
            str(self.sample_rate),
            "--file-type",
            "raw",
        ]
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise RuntimeError(f"cannot start PCM player: {' '.join(command)}: {exc}") from exc
        self._writer_thread = threading.Thread(
            target=self._writer_main,
            name="moss-pcm-player",
            daemon=True,
        )
        self._writer_thread.start()

    def _writer_main(self) -> None:
        process = self.process
        try:
            if process is None or process.stdin is None:
                raise RuntimeError("PCM player was not initialized")
            while True:
                payload = self._queue.get()
                if payload is None:
                    break
                process.stdin.write(payload)
                process.stdin.flush()
        except BaseException as exc:
            self._writer_error = exc
        finally:
            if process is not None and process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass

    def _raise_writer_error(self) -> None:
        if self._writer_error is not None:
            raise RuntimeError(f"PCM player stopped while writing: {self._writer_error}")

    def write(self, waveform: np.ndarray, sample_rate: int, metadata: dict[str, Any]) -> None:
        if int(sample_rate) != self.sample_rate:
            raise RuntimeError(
                f"PCM sample-rate mismatch: expected {self.sample_rate}, got {sample_rate}"
            )
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("PCM player is not running")
        self._raise_writer_error()
        if self.process.poll() is not None:
            raise RuntimeError(f"PCM player exited with status {self.process.returncode}")

        audio = np.asarray(waveform, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        if audio.ndim != 2 or audio.shape[1] != self.channels:
            raise RuntimeError(
                f"PCM channel mismatch: expected (?, {self.channels}), got {audio.shape}"
            )
        if audio.shape[0] == 0:
            return
        if not np.isfinite(audio).all():
            raise RuntimeError("PCM chunk contains NaN or infinity")

        pcm16 = np.rint(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2", copy=False)
        self._queue.put(np.ascontiguousarray(pcm16).tobytes())
        kind = str(metadata.get("kind", "audio"))
        duration = float(metadata.get("duration_seconds", audio.shape[0] / self.sample_rate))
        print(f"playback queued: {kind} chunk, {duration:.3f} s", flush=True)

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        writer_thread = self._writer_thread
        self._writer_thread = None
        self._queue.put(None)
        if writer_thread is not None:
            writer_thread.join(timeout=10.0)
            if writer_thread.is_alive():
                process.terminate()
                writer_thread.join(timeout=1.0)
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        self._raise_writer_error()

    def __enter__(self) -> "PcmChunkPlayer":
        self.start()
        return self

    def __exit__(self, _exc_type: Any, _exc_value: Any, _traceback: Any) -> None:
        self.close()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _chunk_generated_frames(result: dict[str, Any]) -> list[int]:
    return [len(chunk_result["generated_frames"]) for chunk_result in result["chunk_results"]]


def _chunk_stop_reasons(result: dict[str, Any]) -> list[str]:
    return [str(chunk_result["stop_reason"]) for chunk_result in result["chunk_results"]]


def _next_output_path(output_dir: Path, request_index: int) -> tuple[Path, int]:
    resolved_index = request_index
    while True:
        candidate = output_dir / f"request-{resolved_index:04d}.wav"
        if not candidate.exists():
            return candidate, resolved_index
        resolved_index += 1


def run_interactive_session(
    *,
    args: Any,
    runtime: Any,
    session_seconds: float,
    ep_library: Path,
    prompt_audio_path: Path | None,
) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    voice_prepare_start = time.perf_counter()
    prompt_audio_codes = runtime.resolve_prompt_audio_codes(
        voice=args.voice,
        prompt_audio_path=prompt_audio_path,
    )
    voice_prepare_seconds = time.perf_counter() - voice_prepare_start

    decode_input = next(value for value in runtime.sessions["decode"].get_inputs() if value.name == "past_key_0")
    raw_capacity = decode_input.shape[1] if len(decode_input.shape) > 1 else None
    fixed_kv_capacity = int(raw_capacity) if isinstance(raw_capacity, int) else None
    fixed_prompt_rows = len(runtime.build_voice_clone_request_rows(prompt_audio_codes, [])["inputIds"])
    chunk_token_budget = int(args.voice_clone_max_text_tokens)
    if fixed_kv_capacity is not None:
        capacity_token_budget = fixed_kv_capacity - fixed_prompt_rows - args.frames
        if capacity_token_budget <= 0:
            raise RuntimeError(
                f"fixed KV capacity {fixed_kv_capacity} cannot reserve {args.frames} generation frames; "
                f"the selected voice prompt occupies {fixed_prompt_rows} rows"
            )
        chunk_token_budget = min(chunk_token_budget, capacity_token_budget)

    runtime.manifest["generation_defaults"]["max_new_frames"] = int(args.frames)
    warmup_path = output_dir / ".warmup.wav"
    warmup_start = time.perf_counter()
    try:
        runtime.warmup()
        for _ in range(args.warmup):
            warmup_result = runtime.synthesize(
                text=args.warmup_text,
                voice=args.voice,
                prompt_audio_codes=prompt_audio_codes,
                output_audio_path=warmup_path,
                sample_mode="fixed",
                do_sample=True,
                streaming=False,
                max_new_frames=args.frames,
                voice_clone_max_text_tokens=chunk_token_budget,
                first_chunk_text_tokens=(
                    args.first_chunk_text_tokens
                    if args.pcm_playback and args.first_chunk_text_tokens > 0
                    else None
                ),
                enable_wetext=False,
                enable_normalize_tts_text=False,
                seed=args.seed,
            )
            warmup_frame_counts = _chunk_generated_frames(warmup_result)
            warmup_stop_reasons = _chunk_stop_reasons(warmup_result)
            if any(reason != "audio_end" for reason in warmup_stop_reasons):
                raise RuntimeError(
                    f"warmup did not reach audio_end: {warmup_stop_reasons}; "
                    f"frames={warmup_frame_counts}"
                )
    finally:
        warmup_path.unlink(missing_ok=True)
    warmup_seconds = time.perf_counter() - warmup_start

    mode = (
        f"voice clone ({prompt_audio_path})"
        if prompt_audio_path is not None
        else f"built-in voice {args.voice}"
    )
    print("MOSS-TTS-Nano interactive runtime", flush=True)
    print(f"mode: {mode}", flush=True)
    print(f"session creation: {session_seconds:.3f} s", flush=True)
    print(f"voice preparation: {voice_prepare_seconds:.3f} s", flush=True)
    print(f"warmup completed: {warmup_seconds:.3f} s", flush=True)
    print(
        "EP workers: "
        f"{os.environ.get('SPACEMIT_EP_INTRA_THREAD_NUM', 'unknown')} on "
        f"CPU {os.environ.get('SPACEMIT_EP_INTRA_THREAD_AFFINITY', 'unknown')}",
        flush=True,
    )
    print(f"EP SHA256: {_sha256_file(ep_library)}", flush=True)
    print(f"output directory: {output_dir}", flush=True)
    pcm_player: PcmChunkPlayer | None = None
    if bool(args.pcm_playback):
        pcm_player = PcmChunkPlayer(
            device=str(args.audio_device),
            sample_rate=int(runtime.codec_meta["codec_config"]["sample_rate"]),
            channels=int(runtime.codec_meta["codec_config"]["channels"]),
        )
        pcm_player.start()
        print(
            "PCM playback: enabled via aplay "
            f"(device={pcm_player.device}, rate={pcm_player.sample_rate}, channels={pcm_player.channels})",
            flush=True,
        )
    else:
        print("PCM playback: disabled", flush=True)
    print("runtime ready; enter text below (:quit to exit)", flush=True)

    request_index = 1
    try:
        while True:
            try:
                text = input("text> ").strip()
            except EOFError:
                print("\ninput closed; exiting", flush=True)
                break
            except KeyboardInterrupt:
                print("\ninterrupted; exiting", flush=True)
                break
            if not text:
                continue
            if text in {":q", ":quit", ":exit"}:
                print("exiting", flush=True)
                break

            output_path: Path | None = None
            if bool(args.save_wav):
                output_path, request_index = _next_output_path(output_dir, request_index)
            try:
                start = time.perf_counter()
                result = runtime.synthesize(
                    text=text,
                    voice=args.voice,
                    prompt_audio_codes=prompt_audio_codes,
                    output_audio_path=output_path,
                    sample_mode="fixed",
                    do_sample=True,
                    streaming=pcm_player is not None,
                    on_pcm_chunk=pcm_player.write if pcm_player is not None else None,
                    write_wav=bool(args.save_wav),
                    max_new_frames=args.frames,
                    voice_clone_max_text_tokens=chunk_token_budget,
                    first_chunk_text_tokens=(
                        args.first_chunk_text_tokens
                        if args.pcm_playback and args.first_chunk_text_tokens > 0
                        else None
                    ),
                    enable_wetext=False,
                    enable_normalize_tts_text=False,
                    seed=args.seed,
                )
                elapsed = time.perf_counter() - start
                waveform = np.asarray(result["waveform"], dtype=np.float32)
                audio_seconds = waveform.shape[0] / float(result["sample_rate"])
                frame_counts = _chunk_generated_frames(result)
                stop_reasons = _chunk_stop_reasons(result)
                if any(reason != "audio_end" for reason in stop_reasons):
                    if output_path is not None:
                        output_path.unlink(missing_ok=True)
                    print(
                        f"request {request_index}: rejected because a chunk did not reach "
                        f"audio_end: reasons={stop_reasons}, frames={frame_counts}",
                        flush=True,
                    )
                    request_index += 1
                    continue
                if not np.isfinite(waveform).all() or audio_seconds <= 0:
                    if output_path is not None:
                        output_path.unlink(missing_ok=True)
                    print(f"request {request_index}: rejected invalid waveform", flush=True)
                    request_index += 1
                    continue
                output_summary = f", output={output_path}" if output_path is not None else ""
                print(
                    f"request {request_index}: synthesis={elapsed:.3f} s, "
                    f"audio={audio_seconds:.3f} s, warm RTF={elapsed / audio_seconds:.4f}, "
                    f"frames={sum(frame_counts)}, retries={len(result['retry_events'])}"
                    f"{output_summary}",
                    flush=True,
                )
            except Exception as exc:
                if output_path is not None:
                    output_path.unlink(missing_ok=True)
                print(f"request {request_index}: ERROR: {exc}", file=sys.stderr, flush=True)
            request_index += 1
    finally:
        if pcm_player is not None:
            pcm_player.close()
