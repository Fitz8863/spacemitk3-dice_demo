#!/usr/bin/env python3
"""Local HTTP bridge for the board-local MOSS-TTS-Nano runtime.

The bridge imports the packaged K3 delivery from Dice Arena's
``tts/moss-tts-nano`` directory instead of launching its interactive CLI and
scraping diagnostic output.  This is important for two reasons: request
completion is no longer coupled to log wording, and the runtime's PCM callback
can be forwarded as soon as each decoded text chunk is available.

The source delivery is versioned with the game in ``tts/moss-tts-nano``.
Board-specific model files, Python wheels, and native libraries remain ignored
artifacts in that directory, matching the existing ``tts/qwen3-tts`` layout;
``DICE_MOSS_TTS_ROOT`` can still override the path for development.
"""
from __future__ import annotations

import gc
import io
import json
import os
import signal
import sys
import threading
import wave
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
COMPONENT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
from core.env import load_board_env
from core.tts_config import load_component_config
from core.tts_protocol import encode_audio_frame, encode_end_frame, encode_error_frame
from components.tts_moss_nano.settings import load_settings

load_board_env()
# Compatibility export for diagnostics; runtime settings are read through the
# package's single settings loader in ``main``.
TTS_CONFIG = load_component_config(COMPONENT_DIR)


class MossRuntimeError(RuntimeError):
    """A model/runtime error that can be returned as a service response."""


def _append_sys_path(path: Path) -> None:
    value = str(path.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def _prepend_library_path(path: Path) -> None:
    value = str(path.resolve())
    current = os.environ.get("LD_LIBRARY_PATH", "")
    entries = [item for item in current.split(":") if item]
    if value not in entries:
        entries.insert(0, value)
    os.environ["LD_LIBRARY_PATH"] = ":".join(entries)


def _wave_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    """Encode one float32 interleaved waveform chunk as a self-contained WAV."""
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    if audio.ndim != 2 or audio.shape[1] <= 0:
        raise MossRuntimeError(f"unexpected waveform shape: {audio.shape}")
    if audio.shape[0] == 0:
        return b""
    if not np.isfinite(audio).all():
        raise MossRuntimeError("MOSS produced a non-finite waveform chunk")
    pcm16 = np.rint(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2", copy=False)
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(int(audio.shape[1]))
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(np.ascontiguousarray(pcm16).tobytes())
    return output.getvalue()


class MossRuntime:
    """Own one warmed direct-import MOSS runtime and serialize requests."""

    def __init__(
        self,
        *,
        root: Path,
        model_dir: Path,
        voice: str,
        reference_audio: Path | None,
        max_new_frames: int,
        voice_clone_max_text_tokens: int,
        first_chunk_text_tokens: int,
        warmup_text: str,
        seed: int,
        ep_intra_thread_num: int = 4,
        ep_inter_thread_num: int = 1,
        ep_intra_thread_affinity: str = "8;9;10;11",
        ep_disable_op_type_filter: str = "",
    ) -> None:
        self.root = root.resolve()
        self.model_dir = model_dir.resolve()
        self.voice = voice.strip() or "Junhao"
        self.reference_audio = reference_audio.resolve() if reference_audio else None
        self.max_new_frames = int(max_new_frames)
        self.voice_clone_max_text_tokens = int(voice_clone_max_text_tokens)
        self.first_chunk_text_tokens = int(first_chunk_text_tokens)
        self.warmup_text = str(warmup_text).strip() or "你好，这是 MOSS TTS Nano 在 K3 上的演示。"
        self.seed = int(seed)
        self.ep_intra_thread_num = int(ep_intra_thread_num)
        self.ep_inter_thread_num = int(ep_inter_thread_num)
        self.ep_intra_thread_affinity = str(ep_intra_thread_affinity)
        self.ep_disable_op_type_filter = str(ep_disable_op_type_filter)

        self._runtime: Any | None = None
        self._prompt_audio_codes: list[list[int]] | None = None
        self._chunk_token_budget = self.voice_clone_max_text_tokens
        self._sample_rate: int | None = None
        self._channels: int | None = None
        self._request_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._startup_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._server: ThreadingHTTPServer | None = None
        self.ready = False
        self.starting = True
        self.error = ""

    def start(self) -> None:
        if self._startup_thread is not None:
            return
        self._stop_event.clear()
        self._startup_thread = threading.Thread(
            target=self._initialize,
            name="moss-tts-runtime-start",
            daemon=True,
        )
        self._startup_thread.start()

    def _configure_external_runtime(self) -> None:
        python_dir = self.root / "python"
        source_dir = self.root / "src"
        library_dir = self.root / "lib"
        if not python_dir.is_dir() or not source_dir.is_dir():
            raise MossRuntimeError(
                f"MOSS runtime layout is incomplete under {self.root} "
                f"(expected python/ and src/)"
            )
        if not library_dir.is_dir():
            raise MossRuntimeError(f"MOSS runtime library directory is missing: {library_dir}")
        _append_sys_path(python_dir)
        _append_sys_path(source_dir)
        _prepend_library_path(library_dir)
        os.environ.setdefault("PYTHONUNBUFFERED", "1")
        os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        os.environ.setdefault("SPACEMIT_EP_INTRA_THREAD_NUM", str(self.ep_intra_thread_num))
        os.environ.setdefault("SPACEMIT_EP_INTER_THREAD_NUM", str(self.ep_inter_thread_num))
        os.environ.setdefault("SPACEMIT_EP_INTRA_THREAD_AFFINITY", self.ep_intra_thread_affinity)
        os.environ.setdefault("SPACEMIT_EP_DISABLE_OP_TYPE_FILTER", self.ep_disable_op_type_filter)

    def _initialize(self) -> None:
        try:
            self._configure_external_runtime()
            if self._stop_event.is_set():
                return
            from onnx_tts_runtime import OnnxTtsRuntime

            if self.reference_audio is not None and not self.reference_audio.is_file():
                raise MossRuntimeError(f"reference audio not found: {self.reference_audio}")
            if not (self.model_dir / "browser_poc_manifest.json").is_file():
                raise MossRuntimeError(
                    f"MOSS model manifest is missing: {self.model_dir / 'browser_poc_manifest.json'}"
                )

            runtime = OnnxTtsRuntime(
                model_dir=self.model_dir,
                thread_count=self.ep_intra_thread_num,
                max_new_frames=self.max_new_frames,
                do_sample=True,
                sample_mode="fixed",
                execution_provider="spacemit",
                output_dir=self.root / "outputs" / "dice-arena-bridge",
            )
            prompt_audio_codes = runtime.resolve_prompt_audio_codes(
                voice=self.voice,
                prompt_audio_path=self.reference_audio,
            )
            self._chunk_token_budget = self._capacity_aware_chunk_budget(
                runtime,
                prompt_audio_codes,
            )

            # Match the board delivery's startup path: initialize all sessions,
            # run the inexpensive runtime warmup, then synthesize one short
            # request without writing audio before advertising readiness.
            runtime.warmup()
            if self._stop_event.is_set():
                return
            warmup_result = runtime.synthesize(
                text=self.warmup_text,
                voice=self.voice,
                prompt_audio_codes=prompt_audio_codes,
                sample_mode="fixed",
                do_sample=True,
                streaming=False,
                max_new_frames=self.max_new_frames,
                voice_clone_max_text_tokens=self._chunk_token_budget,
                first_chunk_text_tokens=None,
                enable_wetext=False,
                enable_normalize_tts_text=False,
                seed=self.seed,
                write_wav=False,
            )
            self._validate_result(warmup_result)
            if self._stop_event.is_set():
                return

            codec_config = runtime.codec_meta["codec_config"]
            with self._state_lock:
                self._runtime = runtime
                self._prompt_audio_codes = prompt_audio_codes
                self._sample_rate = int(codec_config["sample_rate"])
                self._channels = int(codec_config["channels"])
                self.ready = True
                self.starting = False
                self.error = ""
        except BaseException as exc:
            with self._state_lock:
                self.ready = False
                self.starting = False
                self.error = f"MOSS runtime initialization failed: {exc}"
            print(f"[tts_moss_nano] {self.error}", flush=True)

    def _capacity_aware_chunk_budget(self, runtime: Any, prompt_audio_codes: list[list[int]]) -> int:
        budget = max(1, self.voice_clone_max_text_tokens)
        decode_input = next(
            value for value in runtime.sessions["decode"].get_inputs()
            if value.name == "past_key_0"
        )
        raw_capacity = decode_input.shape[1] if len(decode_input.shape) > 1 else None
        if not isinstance(raw_capacity, int):
            return budget
        fixed_prompt_rows = len(runtime.build_voice_clone_request_rows(prompt_audio_codes, [])[
            "inputIds"
        ])
        capacity_budget = int(raw_capacity) - fixed_prompt_rows - self.max_new_frames
        if capacity_budget <= 0:
            raise MossRuntimeError(
                f"fixed KV capacity {raw_capacity} cannot reserve {self.max_new_frames} frames; "
                f"prompt occupies {fixed_prompt_rows} rows"
            )
        return min(budget, capacity_budget)

    @staticmethod
    def _validate_result(result: dict[str, Any]) -> None:
        waveform = np.asarray(result.get("waveform"), dtype=np.float32)
        if waveform.ndim != 2 or waveform.shape[0] <= 0:
            raise MossRuntimeError("MOSS produced an empty waveform")
        if not np.isfinite(waveform).all():
            raise MossRuntimeError("MOSS produced a non-finite waveform")
        stop_reasons = [str(item.get("stop_reason")) for item in result.get("chunk_results", [])]
        if any(reason != "audio_end" for reason in stop_reasons):
            raise MossRuntimeError(f"MOSS synthesis did not reach audio_end: {stop_reasons}")

    def health(self) -> dict[str, Any]:
        with self._state_lock:
            runtime_loaded = self._runtime is not None
            ready = self.ready
            starting = self.starting
            error = self.error
            sample_rate = self._sample_rate
            channels = self._channels
        return {
            "ok": bool(ready),
            "ready": bool(ready),
            "starting": bool(starting),
            "runtime_loaded": runtime_loaded,
            "root": str(self.root),
            "model_dir": str(self.model_dir),
            "voice": self.voice,
            "reference_audio": str(self.reference_audio) if self.reference_audio else None,
            "sample_rate": sample_rate,
            "channels": channels,
            "max_new_frames": self.max_new_frames,
            "voice_clone_max_text_tokens": self._chunk_token_budget,
            "first_chunk_text_tokens": self.first_chunk_text_tokens,
            "error": error,
        }

    def _require_runtime(self) -> tuple[Any, list[list[int]]]:
        with self._state_lock:
            runtime = self._runtime
            prompt_audio_codes = self._prompt_audio_codes
            ready = self.ready
            detail = self.error
        if not ready or runtime is None or prompt_audio_codes is None:
            raise MossRuntimeError(detail or "MOSS runtime is still warming up")
        return runtime, prompt_audio_codes

    def synthesize(
        self,
        text: str,
        voice: str,
        *,
        on_pcm_chunk: Callable[[np.ndarray, int, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        normalized_text = " ".join(str(text).splitlines()).strip()
        if not normalized_text:
            raise ValueError("input text is empty")
        if voice not in {"", "default", self.voice}:
            raise ValueError(f"MOSS session voice is {self.voice!r}, not {voice!r}")

        with self._request_lock:
            runtime, prompt_audio_codes = self._require_runtime()
            return runtime.synthesize(
                text=normalized_text,
                voice=self.voice,
                prompt_audio_codes=prompt_audio_codes,
                sample_mode="fixed",
                do_sample=True,
                streaming=on_pcm_chunk is not None,
                max_new_frames=self.max_new_frames,
                voice_clone_max_text_tokens=self._chunk_token_budget,
                first_chunk_text_tokens=(
                    self.first_chunk_text_tokens if on_pcm_chunk is not None else None
                ),
                enable_wetext=False,
                enable_normalize_tts_text=False,
                seed=self.seed,
                on_pcm_chunk=on_pcm_chunk,
                write_wav=False,
            )

    def stop(self) -> None:
        self._stop_event.set()
        server = self._server
        if server is not None:
            threading.Thread(target=server.shutdown, daemon=True).start()
        with self._state_lock:
            self.ready = False
            self.starting = False
            self._runtime = None
            self._prompt_audio_codes = None
        # ONNX Runtime owns native session resources. Dropping the Python
        # object before process shutdown makes the TCM release deterministic on
        # boards where the provider keeps allocator state alive until GC.
        gc.collect()


class MossHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], runtime: MossRuntime) -> None:
        self.runtime = runtime
        super().__init__(address, MossRequestHandler)


class MossRequestHandler(BaseHTTPRequestHandler):
    server: MossHttpServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[tts_moss_nano] {fmt % args}", flush=True)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_payload(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 65536:
            raise ValueError("request body must be between 1 and 64KB")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        if urlsplit(self.path).path == "/health":
            self._send_json(self.server.runtime.health())
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _run_request(self, payload: dict[str, Any], *, stream: bool) -> None:
        text = str(payload.get("input", ""))
        voice = str(payload.get("voice", "default"))
        if stream:
            # Check readiness before sending the streaming headers so startup
            # failures remain ordinary JSON/HTTP errors.
            self.server.runtime._require_runtime()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/x-moss-tts-wav-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-TTS-Engine", "moss-tts-nano-spacemit-ep")
            self.send_header("X-TTS-Sample-Rate", str(self.server.runtime._sample_rate or ""))
            self.send_header("X-TTS-Channels", str(self.server.runtime._channels or ""))
            self.send_header("Connection", "close")
            self.end_headers()

            def write_frame(waveform: np.ndarray, sample_rate: int, metadata: dict[str, Any]) -> None:
                frame = _wave_bytes(waveform, sample_rate)
                if not frame:
                    return
                self.wfile.write(encode_audio_frame(frame))
                self.wfile.flush()
                print(
                    f"[tts_moss_nano] stream chunk kind={metadata.get('kind', 'audio')} "
                    f"bytes={len(frame)}",
                    flush=True,
                )

            try:
                result = self.server.runtime.synthesize(text, voice, on_pcm_chunk=write_frame)
                self.server.runtime._validate_result(result)
                self.wfile.write(encode_end_frame())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                print(f"[tts_moss_nano] streaming synthesis failed: {exc}", flush=True)
                try:
                    self.wfile.write(encode_error_frame(str(exc)))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            return

        result = self.server.runtime.synthesize(text, voice)
        self.server.runtime._validate_result(result)
        audio = _wave_bytes(np.asarray(result["waveform"], dtype=np.float32), int(result["sample_rate"]))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(audio)))
        self.send_header("X-TTS-Engine", "moss-tts-nano-spacemit-ep")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(audio)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path not in {"/v1/audio/speech", "/v1/audio/speech/stream"}:
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self._read_payload()
            self._run_request(payload, stream=path.endswith("/stream"))
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except MossRuntimeError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.SERVICE_UNAVAILABLE)
        except Exception as exc:
            print(f"[tts_moss_nano] request failed: {exc}", flush=True)
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_GATEWAY)


def main() -> None:
    # This is an internal HTTP process, not a user-facing CLI.  The package
    # settings loader is the single source for env/config precedence shared
    # with the provider and lifecycle scripts.
    settings = load_settings(TTS_CONFIG)
    runtime = MossRuntime(
        root=settings.root,
        model_dir=settings.model_dir,
        voice=settings.voice,
        reference_audio=settings.reference_audio,
        max_new_frames=settings.max_new_frames,
        voice_clone_max_text_tokens=settings.voice_clone_max_text_tokens,
        first_chunk_text_tokens=settings.first_chunk_text_tokens,
        warmup_text=settings.warmup_text,
        seed=settings.seed,
        ep_intra_thread_num=settings.ep_intra_thread_num,
        ep_inter_thread_num=settings.ep_inter_thread_num,
        ep_intra_thread_affinity=settings.ep_intra_thread_affinity,
        ep_disable_op_type_filter=settings.ep_disable_op_type_filter,
    )
    runtime.start()
    server = MossHttpServer((settings.host, settings.port), runtime)
    runtime._server = server

    def shutdown(_signum: int, _frame: Any) -> None:
        runtime.stop()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    print(
        f"[tts_moss_nano] direct bridge listening on http://{settings.host}:{settings.port}; "
        f"root={settings.root}; model_dir={settings.model_dir}; voice={runtime.voice}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        runtime.stop()
        server.server_close()


if __name__ == "__main__":
    main()
