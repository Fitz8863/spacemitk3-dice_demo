#!/usr/bin/env python3
"""Small K3-local HTTP bridge for the Dice Arena web UI.

The bridge owns only HTTP routing, static serving, and per-round job lifecycle.
Board-local capabilities live in pluggable components
(``components/<id>/provider.py``) and games declare
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
from urllib.parse import parse_qs, urlsplit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from core.components import build_registry
from core.env import load_board_env
from core.errors import (
    DiceArenaError,
    InvalidRequestError,
    JobAlreadyExistsError,
    JobNotFoundError,
)
from core.games import load_games, require_game, resolve_provider_id, run_game
from core.jobs import ComponentJob

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
TTS_STREAM_END = 0
TTS_STREAM_ERROR = 0xFFFFFFFF

load_board_env()

# Resolve runtime settings only after loading the board-local env file.
JOB_TIMEOUT_SECONDS = int(os.environ.get("DICE_JOB_TIMEOUT_SECONDS", "120"))
COMPONENTS = build_registry()
GAMES = load_games()


def _game_provider_id(game_id: str, provider_slot: str, fallback: str) -> str:
    manifest = require_game(GAMES, game_id)
    return resolve_provider_id(manifest, provider_slot, fallback)


def _provider_health(
    provider_id: str,
    provider_type: str,
    provider_role: str | None = None,
) -> dict[str, Any]:
    """Return health without allowing a broken optional provider to break /health."""
    try:
        provider = COMPONENTS.require(
            provider_id,
            expected_type=provider_type,
            expected_role=provider_role,
        )
    except DiceArenaError as exc:
        return {
            "id": provider_id,
            "type": provider_type,
            "role": provider_role or "",
            "ok": False,
            "error": exc.message,
        }
    try:
        health = provider.health()
    except Exception as exc:
        return {
            "id": provider_id,
            "type": provider_type,
            "role": provider_role or "",
            "ok": False,
            "error": str(exc),
        }
    if not isinstance(health, dict):
        return {
            "id": provider_id,
            "type": provider_type,
            "role": provider_role or "",
            "ok": False,
            "error": "health() must return an object",
        }
    normalized = dict(health)
    normalized.setdefault("id", provider.id)
    normalized.setdefault("type", provider.type)
    normalized.setdefault("role", provider.role)
    return normalized


def _selected_provider_id(game_id: str, provider_slot: str, fallback: str) -> str:
    return _game_provider_id(game_id, provider_slot, fallback)


def _selected_tts_id(game_id: str = "dice") -> str:
    try:
        return _selected_provider_id(game_id, "tts", "tts_qwen3")
    except DiceArenaError:
        return "tts_qwen3"


def _tts_provider(payload: dict[str, Any], game_id: str | None = None):
    # Explicit request selection is useful for diagnostics. Normal game calls
    # send only ``game`` so the backend remains the authority for provider
    # selection and an operator can switch it with DICE_TTS_PROVIDER.
    provider_id = payload.get("provider")
    if not provider_id:
        provider_id = os.environ.get("DICE_TTS_PROVIDER")
    if not provider_id and game_id:
        provider_id = _game_provider_id(game_id, "tts", "tts_qwen3")
    provider_id = str(provider_id or _selected_tts_id())
    return COMPONENTS.require(provider_id, expected_type="tts")


jobs: dict[str, ComponentJob] = {}
jobs_lock = threading.Lock()
active_job_id: str | None = None

_ADJUDICATION_JOB_PREFIXES = ("/api/adjudicate/", "/api/analyze/")


def _adjudication_job_remainder(path: str) -> str | None:
    """Return the job-route suffix for the canonical or legacy endpoint."""
    for prefix in _ADJUDICATION_JOB_PREFIXES:
        if path.startswith(prefix):
            return path[len(prefix):]
    return None


def create_adjudication_job(game_id: str) -> ComponentJob:
    global active_job_id
    require_game(GAMES, game_id)  # raises GameNotFoundError / GameDisabledError
    with jobs_lock:
        if active_job_id:
            active = jobs.get(active_job_id)
            if active and active.status in {"queued", "running"}:
                raise JobAlreadyExistsError(active_job_id)

        def run_fn(on_log, is_cancelled, on_event):
            return run_game(
                GAMES, game_id, on_log, is_cancelled, on_event,
                JOB_TIMEOUT_SECONDS, COMPONENTS,
            )

        job = ComponentJob(run_fn=run_fn, name=f"{game_id}-job")
        jobs[job.id] = job
        active_job_id = job.id
        job.start()
        return job


class Handler(BaseHTTPRequestHandler):
    server_version = "DiceArenaK3/0.3"

    protocol_version = "HTTP/1.1"

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

    def send_sse_snapshot(self, snapshot: dict[str, Any], event: str = "update") -> None:
        payload = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
        self.wfile.write(f"event: {event}\ndata: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()

    @staticmethod
    def job_stream_delta(snapshot: dict[str, Any], after_sequence: int) -> dict[str, Any]:
        """Build a compact SSE update instead of resending logs/event history."""
        return {
            "job_id": snapshot["job_id"],
            "status": snapshot["status"],
            "phase": snapshot["phase"],
            "error": snapshot["error"],
            "cancelled": snapshot["cancelled"],
            "result": snapshot["result"],
            "events": [
                event for event in snapshot["events"]
                if int(event.get("sequence", 0)) > after_sequence
            ],
            "event_sequence": snapshot["event_sequence"],
            "revision": snapshot["revision"],
            "started_at": snapshot["started_at"],
            "finished_at": snapshot["finished_at"],
        }

    def stream_job_events(self, job: ComponentJob) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        snapshot = job.snapshot()
        revision = snapshot["revision"]
        event_sequence = snapshot["event_sequence"]
        self.send_sse_snapshot(snapshot, "snapshot")
        try:
            while snapshot["status"] in {"queued", "running"}:
                snapshot = job.wait_for_update(revision, timeout=15.0)
                if snapshot["revision"] <= revision:
                    self.send_sse_snapshot({
                        "job_id": job.id,
                        "revision": revision,
                        "event_sequence": event_sequence,
                    }, "heartbeat")
                    continue
                delta = self.job_stream_delta(snapshot, event_sequence)
                revision = snapshot["revision"]
                event_sequence = snapshot["event_sequence"]
                if snapshot["status"] in {"success", "error"}:
                    self.send_sse_snapshot(delta, "complete")
                    return
                self.send_sse_snapshot(delta)
            self.send_sse_snapshot(self.job_stream_delta(snapshot, event_sequence), "complete")
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.close_connection = True

    def do_GET(self) -> None:  # noqa: N802
        parsed_url = urlsplit(self.path)
        path = parsed_url.path
        if path == "/api/health":
            component_items = COMPONENTS.all(include_health=True)
            tts_id = _selected_tts_id()
            tts_health = _provider_health(tts_id, "tts")
            adjudicator_id = _selected_provider_id(
                "dice", "vision_adjudicator", "vision_yolo"
            )
            adjudicator_health = _provider_health(
                adjudicator_id, "vision", "adjudicator"
            )
            self.send_json({
                "ok": True,
                "backend": "k3-local-component-bridge",
                "components": component_items,
                "adjudicator_provider": adjudicator_id,
                "adjudicator": adjudicator_health,
                # Compatibility alias for existing dashboards/clients.
                "vision": adjudicator_health,
                "yolo_binary": adjudicator_health.get("binary", ""),
                "yolo_ready": bool(adjudicator_health.get("ready", False)),
                "llm_configured": bool(adjudicator_health.get("llm_configured", False)),
                "tts_provider": tts_id,
                "tts": tts_health,
                "tts_url": tts_health.get("url", ""),
                "tts_ready": bool(tts_health.get("ok", False)),
                "tts_engine": tts_health.get("engine", ""),
                "tts_speaker": tts_health.get("speaker", ""),
                "camera": os.environ.get("DICE_CAMERA", "config.json"),
            })
            return
        if path == "/api/tts/health":
            requested = parse_qs(parsed_url.query).get("provider", [""])[0]
            provider_id = requested or _selected_tts_id("dice")
            health = _provider_health(provider_id, "tts")
            self.send_json(health, HTTPStatus.OK if health.get("ok") is not False else HTTPStatus.SERVICE_UNAVAILABLE)
            return
        if path == "/api/components":
            self.send_json({"components": COMPONENTS.all(include_health=True)})
            return
        if path == "/api/games":
            self.send_json({"games": GAMES.all()})
            return
        remainder = _adjudication_job_remainder(path)
        if remainder is not None:
            job_id, _, suffix = remainder.partition("/")
            with jobs_lock:
                job = jobs.get(job_id)
            if not job:
                self.send_error_json(JobNotFoundError(job_id))
            elif suffix == "stream":
                self.stream_job_events(job)
            elif suffix == "events":
                snapshot = job.snapshot()
                self.send_json({
                    "job_id": job_id,
                    "status": snapshot["status"],
                    "phase": snapshot["phase"],
                    "events": snapshot["events"],
                    "event_sequence": snapshot["event_sequence"],
                    "revision": snapshot["revision"],
                })
            else:
                self.send_json(job.snapshot())
            return
        self.serve_static(path)

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/api/tts/stream":
            try:
                payload = self.read_json()
                provider = _tts_provider(payload, payload.get("game"))
                if not callable(getattr(provider, "stream", None)):
                    raise InvalidRequestError(f"TTS provider {provider.id} does not implement stream()")
            except DiceArenaError as exc:
                self.send_error_json(exc)
                return
            except Exception as exc:
                self.send_error_json(DiceArenaError(str(exc), "TTS_PROVIDER_ERROR", 502))
                return

            provider_health = _provider_health(provider.id, "tts")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/x-dice-arena-wav-stream")
            self.send_header("X-Dice-TTS-Provider", provider.id)
            self.send_header("X-Dice-TTS-Engine", str(provider_health.get("engine", provider.id)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()

            def write_frame(audio: bytes) -> None:
                self.wfile.write(len(audio).to_bytes(4, "big"))
                self.wfile.write(audio)
                self.wfile.flush()

            try:
                provider.stream(payload, write_frame)
                self.wfile.write(TTS_STREAM_END.to_bytes(4, "big"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                print(f"[tts] stream provider={provider.id} failed: {exc}", flush=True)
                try:
                    message = str(exc).encode("utf-8")[:2000]
                    self.wfile.write(TTS_STREAM_ERROR.to_bytes(4, "big"))
                    self.wfile.write(len(message).to_bytes(4, "big"))
                    self.wfile.write(message)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            return
        if path == "/api/tts/synthesize":
            try:
                payload = self.read_json()
                provider = _tts_provider(payload, payload.get("game"))
                synthesize = getattr(provider, "synthesize", None)
                if not callable(synthesize):
                    raise InvalidRequestError(f"TTS provider {provider.id} does not implement synthesize()")
                audio, headers = synthesize(payload)
            except DiceArenaError as exc:
                self.send_error_json(exc)
            except Exception as exc:
                self.send_error_json(DiceArenaError(str(exc), "TTS_PROVIDER_ERROR", 502))
            else:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", headers.pop("Content-Type", "audio/wav"))
                self.send_header("Content-Length", str(len(audio)))
                self.send_header("X-Dice-TTS-Provider", provider.id)
                self.send_header("Cache-Control", "no-store")
                for name, value in headers.items():
                    self.send_header(name, value)
                self.end_headers()
                try:
                    self.wfile.write(audio)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            return
        if path in {"/api/adjudicate", "/api/analyze"}:
            try:
                payload = self.read_json()
                game_id = str(payload.get("game") or "dice")
                job = create_adjudication_job(game_id)
                self.send_json(job.snapshot(), HTTPStatus.ACCEPTED)
            except DiceArenaError as exc:
                self.send_error_json(exc)
            return
        remainder = _adjudication_job_remainder(path)
        if remainder is not None and remainder.endswith("/cancel"):
            job_id = remainder[:-len("/cancel")].rstrip("/")
            with jobs_lock:
                job = jobs.get(job_id)
            if not job:
                self.send_error_json(JobNotFoundError(job_id))
            else:
                job.cancel()
                self.send_json(job.snapshot())
            return
        self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def serve_static(self, path: str) -> None:
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
    parser = argparse.ArgumentParser(description="K3 Dice Arena provider bridge")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Dice Arena K3 backend listening on http://{args.host}:{args.port}", flush=True)
    adjudicator = next((item for item in COMPONENTS.all(include_health=True)
                        if item["type"] == "vision" and item["role"] == "adjudicator"), None)
    print(
        f"Vision adjudicator provider: {adjudicator.get('id') if adjudicator else 'none'}",
        flush=True,
    )
    print(f"Components: {', '.join(COMPONENTS.ids()) or 'none'}", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
