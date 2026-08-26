#!/usr/bin/env python3
"""Small K3-local HTTP bridge for the Dice Arena web UI.

The bridge deliberately starts the checked-in YOLOv8 C++ application for each
reveal.  The C++ application owns camera capture, OpenCL preprocessing,
SpaceMIT ONNX Runtime EP inference, stable-frame filtering, and LLM
verification.  The browser never receives the LLM key and never decides the
winner itself.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import mimetypes
import os
import selectors
import sys
import signal
import subprocess
import threading
import time
import uuid
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
VISION_ROOT = ROOT / "vision" / "yolov8_objdetect"
TTS_ROOT = ROOT / "tts" / "qwen3-tts"
DEFAULT_BINARY = VISION_ROOT / "build" / "yolov8_camera"
ENV_FILE = ROOT / ".dice-arena.env"
TTS_REQUEST_LOCK = threading.Lock()
TTS_STREAM_END = 0
TTS_STREAM_ERROR = 0xFFFFFFFF
TTS_ENGINE = "qwen3-tts-k3-llama-server"
TTS_SPEAKER_FILE = "anke.spk.bin"
_tts_interactive_module = None


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE lines without overwriting the process env."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


load_env_file(ENV_FILE)

# Resolve runtime settings only after loading the board-local env file.
JOB_TIMEOUT_SECONDS = int(os.environ.get("DICE_JOB_TIMEOUT_SECONDS", "120"))
TTS_URL = os.environ.get("DICE_TTS_URL", "http://127.0.0.1:18080").rstrip("/")
TTS_TIMEOUT_SECONDS = float(os.environ.get("DICE_TTS_TIMEOUT_SECONDS", "120"))


def configured_llm() -> bool:
    if os.environ.get("DICE_LLM_API_KEY", "").strip():
        return True
    try:
        config = json.loads((VISION_ROOT / "config.json").read_text(encoding="utf-8"))
        return bool(config.get("llm", {}).get("api_key", "").strip())
    except (OSError, ValueError, AttributeError):
        return False


def yolo_binary() -> Path:
    return Path(os.environ.get("DICE_YOLO_BINARY", str(DEFAULT_BINARY))).resolve()


def now_ms() -> int:
    return int(time.time() * 1000)


def tts_health() -> bool:
    """Return whether the board-local Qwen3-TTS llama-server is reachable."""
    try:
        with urllib.request.urlopen(f"{TTS_URL}/health", timeout=1.5) as response:
            return 200 <= response.status < 300
    except (OSError, urllib.error.URLError):
        return False


def validate_tts_payload(payload: dict[str, Any]) -> tuple[str, str, float]:
    text = str(payload.get("text", "")).strip()
    if not text:
        raise ValueError("text is required")
    if len(text) > 4000:
        raise ValueError("text is too long; limit is 4000 characters")

    speed = payload.get("speed", 1.0)
    try:
        speed = float(speed)
    except (TypeError, ValueError) as exc:
        raise ValueError("speed must be a number") from exc
    speed = max(0.25, min(4.0, speed))
    voice = str(payload.get("voice", "default"))
    return text, voice, speed


def get_tts_interactive_module():
    """Load the migrated interactive client so HTTP uses the same splitter."""
    global _tts_interactive_module
    if _tts_interactive_module is None:
        path = TTS_ROOT / "qwen3_tts_interactive.py"
        spec = importlib.util.spec_from_file_location("dice_arena_qwen3_tts_interactive", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"unable to load interactive TTS client: {path}")
        module = importlib.util.module_from_spec(spec)
        # dataclasses and other decorators may resolve the module through
        # sys.modules while the file is being executed.
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(spec.name, None)
            raise
        _tts_interactive_module = module
    return _tts_interactive_module


def stream_tts(payload: dict[str, Any], write_frame) -> None:
    """Generate one browser request with ordered WAV frames like run_interactive.sh."""
    text, voice, speed = validate_tts_payload(payload)
    client = get_tts_interactive_module()
    chunks = client.split_text(text)
    if not chunks:
        raise ValueError("text is empty after normalization")

    # The browser sends one request for the complete announcement. Internally
    # we reuse the board interactive client's punctuation-aware generation and
    # emit each completed WAV frame immediately, so playback can begin after
    # the first frame without exposing segment requests to the UI.
    print(f"[tts] stream start engine={TTS_ENGINE} speaker={TTS_SPEAKER_FILE} chunks={len(chunks)} text={text[:120]!r}", flush=True)
    with TTS_REQUEST_LOCK:
        for index, chunk_text in enumerate(chunks, start=1):
            audio = client.synthesize(chunk_text, voice=voice, speed=speed).wav
            print(f"[tts] generated frame={index}/{len(chunks)} bytes={len(audio)} speaker={TTS_SPEAKER_FILE}", flush=True)
            write_frame(audio)
    print(f"[tts] stream complete engine={TTS_ENGINE} speaker={TTS_SPEAKER_FILE}", flush=True)


def synthesize_tts(payload: dict[str, Any]) -> tuple[bytes, dict[str, str]]:
    """Proxy a text-to-WAV request to the board-local Qwen3-TTS service."""
    text, voice, speed = validate_tts_payload(payload)

    body = json.dumps({
        "model": "qwen3-tts",
        "input": text,
        "voice": voice,
        "response_format": "wav",
        "speed": speed,
    }, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{TTS_URL}/v1/audio/speech",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # The K3 TTS runtime is a single expensive local llama-server. Serialize
    # synthesis requests so rapid UI events cannot make several generations
    # compete for the same model/AI cores.
    try:
        with TTS_REQUEST_LOCK:
            with urllib.request.urlopen(request, timeout=TTS_TIMEOUT_SECONDS) as response:
                audio = response.read()
                headers = {
                    name: value
                    for name, value in response.headers.items()
                    if name.lower().startswith("x-tts-")
                }
                content_type = response.headers.get("Content-Type", "audio/wav")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"TTS HTTP {exc.code}: {detail}") from exc
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"TTS service unavailable at {TTS_URL}: {exc}") from exc

    if len(audio) < 44 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
        raise RuntimeError("TTS service did not return a valid WAV")
    headers["Content-Type"] = content_type
    return audio, headers


class AnalysisJob:
    def __init__(self) -> None:
        self.id = uuid.uuid4().hex
        self.status = "queued"
        self.phase = "queued"
        self.error = ""
        self.result: dict[str, Any] | None = None
        self.logs: list[str] = []
        self.started_at = now_ms()
        self.finished_at: int | None = None
        self.process: subprocess.Popen[str] | None = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name=f"dice-yolo-{self.id[:8]}", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def add_log(self, line: str) -> None:
        line = line.rstrip()
        if not line:
            return
        print(f"[vision:{self.id[:8]}] {line}", flush=True)
        with self.lock:
            self.logs.append(line[-500:])
            self.logs = self.logs[-40:]
            if line.startswith("[YOLO]") or "OpenCL GPU" in line or "Model loaded" in line:
                self.phase = "detecting"
            if line.startswith("[LLM]") or "calling LLM" in line:
                self.phase = "verifying"

    def _run(self) -> None:
        result_path = Path("/tmp") / f"dice-arena-{self.id}.json"
        binary = yolo_binary()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            self._fail(f"YOLOv8 executable not found or not executable: {binary}")
            return
        if not configured_llm():
            self._fail("LLM is not configured; set DICE_LLM_API_KEY or .dice-arena.env")
            return

        command = [
            str(binary),
            "--config", "config.json",
            "--no-display",
            "--rejudge-on-change",
            "--require-llm",
            "--result-file", str(result_path),
            "--exit-on-result",
        ]
        env = os.environ.copy()
        # Keep the secret only in this child process.  The browser/API response
        # never contains it, and the C++ verifier removes it before invoking curl.
        self.status = "running"
        self.phase = "starting"
        try:
            self.process = subprocess.Popen(
                command,
                cwd=VISION_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            deadline = time.monotonic() + JOB_TIMEOUT_SECONDS
            assert self.process.stdout is not None
            selector = selectors.DefaultSelector()
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while True:
                if time.monotonic() > deadline:
                    self._terminate()
                    self._fail(f"YOLOv8 analysis timed out after {JOB_TIMEOUT_SECONDS}s")
                    return
                events = selector.select(timeout=0.5)
                for key, _ in events:
                    line = key.fileobj.readline()
                    if line:
                        self.add_log(line)
                if self.process.poll() is not None:
                    for line in self.process.stdout:
                        self.add_log(line)
                    break
            selector.close()
            return_code = self.process.wait(timeout=5)
            if result_path.is_file():
                try:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    self._fail(f"YOLOv8 wrote an invalid result: {exc}")
                else:
                    self._succeed(result)
            elif return_code != 0:
                self._fail(f"YOLOv8 process exited with code {return_code}")
            else:
                self._fail("YOLOv8 stopped before an LLM-verified result was produced")
        except Exception as exc:  # keep the HTTP server alive on runtime errors
            self._fail(f"failed to start YOLOv8: {exc}")
        finally:
            try:
                result_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _succeed(self, result: dict[str, Any]) -> None:
        with self.lock:
            self.result = result
            self.status = "success"
            self.phase = "complete"
            self.finished_at = now_ms()

    def _fail(self, message: str) -> None:
        with self.lock:
            self.error = message
            self.status = "error"
            self.phase = "error"
            self.finished_at = now_ms()
            self.logs.append(message)
            self.logs = self.logs[-40:]

    def _terminate(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass

    def cancel(self) -> None:
        self._terminate()
        self._fail("analysis cancelled")

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "job_id": self.id,
                "status": self.status,
                "phase": self.phase,
                "error": self.error,
                "result": self.result,
                "logs": list(self.logs),
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }


jobs: dict[str, AnalysisJob] = {}
jobs_lock = threading.Lock()
active_job_id: str | None = None


def create_job() -> AnalysisJob:
    global active_job_id
    with jobs_lock:
        if active_job_id:
            active = jobs.get(active_job_id)
            if active and active.status in {"queued", "running"}:
                raise RuntimeError("another dice analysis is already running")
        job = AnalysisJob()
        jobs[job.id] = job
        active_job_id = job.id
        job.start()
        return job


class Handler(BaseHTTPRequestHandler):
    server_version = "DiceArenaK3/0.2"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep the board console readable; analysis logs are exposed per job.
        print(f"[web] {self.address_string()} {fmt % args}", flush=True)

    def send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(min(length, 65536))
        value = json.loads(raw.decode("utf-8"))
        return value if isinstance(value, dict) else {}

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/health":
            binary = yolo_binary()
            self.send_json({
                "ok": True,
                "backend": "k3-local-yolov8-llm-tts-bridge",
                "yolo_binary": str(binary),
                "yolo_ready": binary.is_file() and os.access(binary, os.X_OK),
                "llm_configured": configured_llm(),
                "tts_url": TTS_URL,
                "tts_ready": tts_health(),
                "tts_engine": TTS_ENGINE,
                "tts_speaker": TTS_SPEAKER_FILE,
                "tts_root": str(TTS_ROOT),
                "camera": os.environ.get("DICE_CAMERA", "config.json"),
            })
            return
        if self.path == "/api/tts/health":
            self.send_json({"ok": tts_health(), "url": TTS_URL, "root": str(TTS_ROOT)})
            return
        if self.path.startswith("/api/analyze/"):
            job_id = self.path[len("/api/analyze/"):].split("/", 1)[0]
            with jobs_lock:
                job = jobs.get(job_id)
            if not job:
                self.send_json({"error": "job not found"}, HTTPStatus.NOT_FOUND)
            else:
                self.send_json(job.snapshot())
            return
        self.serve_static()

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/tts/stream":
            try:
                payload = self.read_json()
                validate_tts_payload(payload)
            except (ValueError, UnicodeDecodeError) as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/x-dice-arena-wav-stream")
            self.send_header("X-Dice-TTS-Engine", TTS_ENGINE)
            self.send_header("X-Dice-TTS-Speaker", TTS_SPEAKER_FILE)
            self.send_header("X-Dice-TTS-Source", "board-local-llama-server")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()

            def write_frame(audio: bytes) -> None:
                self.wfile.write(len(audio).to_bytes(4, "big"))
                self.wfile.write(audio)
                self.wfile.flush()

            try:
                stream_tts(payload, write_frame)
                self.wfile.write(TTS_STREAM_END.to_bytes(4, "big"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # The browser can cancel the one streaming request when a new
                # announcement supersedes it.
                pass
            except Exception as exc:
                print(f"[tts] stream failed: {exc}", flush=True)
                try:
                    message = str(exc).encode("utf-8")[:2000]
                    self.wfile.write(TTS_STREAM_ERROR.to_bytes(4, "big"))
                    self.wfile.write(len(message).to_bytes(4, "big"))
                    self.wfile.write(message)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            return
        if self.path == "/api/tts/synthesize":
            try:
                audio, headers = synthesize_tts(self.read_json())
            except (ValueError, UnicodeDecodeError) as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except RuntimeError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_GATEWAY)
            else:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", headers.pop("Content-Type", "audio/wav"))
                self.send_header("Content-Length", str(len(audio)))
                self.send_header("Cache-Control", "no-store")
                for name, value in headers.items():
                    self.send_header(name, value)
                self.end_headers()
                try:
                    self.wfile.write(audio)
                except (BrokenPipeError, ConnectionResetError):
                    # A browser may cancel an in-flight speech request when a
                    # newer announcement supersedes it. The synthesis itself
                    # has completed; do not turn the expected disconnect into
                    # a noisy server traceback.
                    pass
            return
        if self.path == "/api/analyze":
            try:
                self.read_json()
                job = create_job()
            except RuntimeError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.CONFLICT)
            except (ValueError, UnicodeDecodeError) as exc:
                self.send_json({"error": f"invalid request: {exc}"}, HTTPStatus.BAD_REQUEST)
            else:
                self.send_json(job.snapshot(), HTTPStatus.ACCEPTED)
            return
        if self.path.startswith("/api/analyze/") and self.path.endswith("/cancel"):
            job_id = self.path[len("/api/analyze/"):-len("/cancel")].rstrip("/")
            with jobs_lock:
                job = jobs.get(job_id)
            if not job:
                self.send_json({"error": "job not found"}, HTTPStatus.NOT_FOUND)
            else:
                job.cancel()
                self.send_json(job.snapshot())
            return
        self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def serve_static(self) -> None:
        path = self.path.split("?", 1)[0]
        relative = path.lstrip("/") or "index.html"
        candidate = (WEB_ROOT / relative).resolve()
        try:
            candidate.relative_to(WEB_ROOT.resolve())
        except ValueError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="K3 Dice Arena YOLOv8 + LLM HTTP bridge")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Dice Arena K3 backend listening on http://{args.host}:{args.port}", flush=True)
    print(f"YOLOv8 binary: {yolo_binary()}", flush=True)
    print(f"LLM configured: {configured_llm()}", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
