#!/usr/bin/env python3
"""Small K3-local HTTP bridge for the Dice Arena web UI.

The bridge owns only HTTP routing, static serving, and per-round job lifecycle.
Board-local capabilities live in pluggable components
(``components/vision_yolo.py``, ``components/tts_qwen3.py``) and games declare
which components they orchestrate (``games/*/``). The browser never receives
the LLM key and never decides the winner itself.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from components.tts_qwen3 import (
    TTS_ENGINE,
    TTS_ROOT,
    TTS_SPEAKER_FILE,
    TTS_URL,
    stream_tts,
    synthesize_tts,
    tts_health,
    validate_tts_payload,
)
from components.vision_yolo import configured_llm, yolo_binary
from core.env import load_board_env
from core.errors import (
    DiceArenaError,
    InvalidRequestError,
    JobAlreadyExistsError,
    JobNotFoundError,
)
from core.games import load_games, require_game, run_game
from core.jobs import ComponentJob

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
TTS_STREAM_END = 0
TTS_STREAM_ERROR = 0xFFFFFFFF

load_board_env()

# Resolve runtime settings only after loading the board-local env file.
JOB_TIMEOUT_SECONDS = int(os.environ.get("DICE_JOB_TIMEOUT_SECONDS", "120"))
GAMES = load_games()


def _vision_phase_of(line: str) -> str | None:
    """Map a YOLOv8 stdout line to a coarse job phase label."""
    if line.startswith("[YOLO]") or "OpenCL GPU" in line or "Model loaded" in line:
        return "detecting"
    if line.startswith("[LLM]") or "calling LLM" in line:
        return "verifying"
    return None


jobs: dict[str, ComponentJob] = {}
jobs_lock = threading.Lock()
active_job_id: str | None = None


def create_job(game_id: str) -> ComponentJob:
    global active_job_id
    require_game(GAMES, game_id)  # raises GameNotFoundError / GameDisabledError
    with jobs_lock:
        if active_job_id:
            active = jobs.get(active_job_id)
            if active and active.status in {"queued", "running"}:
                raise JobAlreadyExistsError(active_job_id)

        def run_fn(on_log, is_cancelled):
            return run_game(GAMES, game_id, on_log, is_cancelled, JOB_TIMEOUT_SECONDS)

        job = ComponentJob(run_fn=run_fn, phase_of=_vision_phase_of, name=f"{game_id}-job")
        jobs[job.id] = job
        active_job_id = job.id
        job.start()
        return job


class Handler(BaseHTTPRequestHandler):
    server_version = "DiceArenaK3/0.3"

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

    def send_error_json(self, error: DiceArenaError) -> None:
        """Send a standardized error response from a DiceArenaError."""
        self.send_json(error.to_dict(), error.status)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        if length > 65536:
            raise InvalidRequestError("request body too large (max 64KB)")
        try:
            raw = self.rfile.read(length)
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                raise InvalidRequestError("request body must be a JSON object")
            return value
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise InvalidRequestError(f"invalid JSON: {exc}") from exc

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
        if self.path == "/api/games":
            self.send_json({"games": GAMES.all()})
            return
        if self.path.startswith("/api/analyze/"):
            job_id = self.path[len("/api/analyze/"):].split("/", 1)[0]
            with jobs_lock:
                job = jobs.get(job_id)
            if not job:
                self.send_error_json(JobNotFoundError(job_id))
            else:
                self.send_json(job.snapshot())
            return
        self.serve_static()

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/tts/stream":
            try:
                payload = self.read_json()
                validate_tts_payload(payload)
            except DiceArenaError as exc:
                self.send_error_json(exc)
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
            except DiceArenaError as exc:
                self.send_error_json(exc)
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
                payload = self.read_json()
                game_id = str(payload.get("game") or "dice")
                job = create_job(game_id)
                self.send_json(job.snapshot(), HTTPStatus.ACCEPTED)
            except DiceArenaError as exc:
                self.send_error_json(exc)
            return
        if self.path.startswith("/api/analyze/") and self.path.endswith("/cancel"):
            job_id = self.path[len("/api/analyze/"):-len("/cancel")].rstrip("/")
            with jobs_lock:
                job = jobs.get(job_id)
            if not job:
                self.send_error_json(JobNotFoundError(job_id))
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


if __name__ == "__main__":
    main()
