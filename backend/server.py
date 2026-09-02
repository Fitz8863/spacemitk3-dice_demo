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
import signal
import threading
from http import HTTPStatus
from urllib.parse import parse_qs, urlsplit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from core.components import build_registry
from core.errors import (
    DiceArenaError,
    InvalidRequestError,
    JobAlreadyExistsError,
    JobNotFoundError,
)
from core.games import (
    GAMES_ROOT,
    GameRegistry,
    load_games,
    require_game,
    resolve_game_audio_path,
    resolve_provider_id,
    resolve_adjudication_timeout,
    run_game,
)
from core.jobs import ComponentJob
from core.state_machine import GameRound, IntentRejectedError, RoundClosedError
from core.tts_dispatch import TtsDispatcher
from core.tts_protocol import (
    encode_audio_frame,
    encode_end_frame,
    encode_error_frame,
)
from components.vision_yolov8_adjudicator.profile import (
    compose_video_url,
    load_component_config,
    load_runtime_config,
    resolve_runtime_config_path,
)

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"

# Operational fallback; each game manifest's timeouts.adjudication_seconds owns
# the per-round budget.
JOB_TIMEOUT_SECONDS = 120
COMPONENTS = build_registry()
GAMES = load_games()

# Manifest hot reload: game manifests are small and reread whenever their
# mtime changes, so line edits, wav swaps, and speech-mode changes take
# effect on the next page load or request without a service restart.
# Engine processes keep their boot-time set: slot ids are expected to keep
# pointing at the providers that were started (per-line changes between
# local/remote/audio need no new process).
_GAMES_ROOT = GAMES_ROOT
_GAMES_LOCK = threading.Lock()


def _manifest_mtimes() -> dict[str, float]:
    return {
        str(path): path.stat().st_mtime
        for path in sorted(_GAMES_ROOT.glob("*/manifest.json"))
    }


def get_games() -> GameRegistry:
    """Return the game registry, reloading manifests when their mtime changes.

    A reload that would drop a previously loaded game means a broken manifest
    edit; keep serving the last good registry in that case so a typo cannot
    make games vanish from the UI.  Removing a game on purpose needs a
    service restart.
    """
    global GAMES, _GAMES_MTIMES
    with _GAMES_LOCK:
        mtimes = _manifest_mtimes()
        if mtimes != _GAMES_MTIMES:
            candidate = load_games(_GAMES_ROOT)
            missing = {m["id"] for m in GAMES.all()} - {m["id"] for m in candidate.all()}
            if missing:
                print(
                    f"[games] reload skipped; broken manifest removed {sorted(missing)}, "
                    "keeping last good config",
                    flush=True,
                )
            else:
                GAMES = candidate
                _GAMES_MTIMES = mtimes
        return GAMES


_GAMES_MTIMES: dict[str, float] = _manifest_mtimes()


def _game_provider_id(game_id: str, provider_slot: str, fallback: str) -> str:
    manifest = require_game(get_games(), game_id)
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
        return TtsDispatcher(COMPONENTS, get_games()).provider_id(game_id)
    except DiceArenaError:
        return "tts_qwen3"


def _safe_profile_metadata(profile: dict[str, Any], base_url: str, runtime: dict[str, Any]) -> dict[str, Any]:
    """Project one profile to public health fields; never recurse into registry metadata."""
    video = profile.get("video", {})
    path = video.get("path") if isinstance(video, dict) else ""
    result: dict[str, Any] = {
        "profile_id": profile.get("game_id", ""),
        "video_enabled": bool(video.get("enabled", True)) if isinstance(video, dict) else False,
        "video_path": path if isinstance(path, str) else "",
        "mode": runtime.get("mode", ""),
        "prewarm": bool(runtime.get("prewarm_camera", False)),
        "runtime_mode": runtime.get("mode", ""),
        "prewarm_camera": bool(runtime.get("prewarm_camera", False)),
        "mediamtx_base_url": base_url,
        "webrtc_base_url": base_url,
    }
    if result["video_enabled"] and result["video_path"] and isinstance(base_url, str):
        try:
            result["video_url"] = compose_video_url(base_url, result["video_path"])
        except Exception:
            pass
    multi = profile.get("multi_view") if isinstance(profile.get("multi_view"), dict) else {}
    views: list[dict[str, Any]] = []
    for view in multi.get("views", []) if isinstance(multi.get("views", []), list) else []:
        if not isinstance(view, dict) or not isinstance(view.get("id"), str):
            continue
        view_video = view.get("video") if isinstance(view.get("video"), dict) else {}
        view_path = view_video.get("path", "")
        item = {"id": view["id"], "video_path": view_path if isinstance(view_path, str) else ""}
        if bool(view_video.get("enabled", True)) and item["video_path"] and isinstance(base_url, str):
            try:
                item["video_url"] = compose_video_url(base_url, item["video_path"])
            except Exception:
                pass
        views.append(item)
    result["multi_view"] = {"enabled": bool(multi.get("enabled", False)), "min_views": int(multi.get("min_views", 1)), "views": views}
    return result


def _vision_profile_metadata(game_id: str, provider_id: str) -> dict[str, Any]:
    """Expose safe, deployment-facing vision metadata without prompts/secrets."""
    try:
        manifest = require_game(get_games(), game_id)
        profile = manifest.get("vision_profile")
        if not isinstance(profile, dict):
            return {}
        package_dir = ROOT / "backend" / "components" / provider_id
        config = load_component_config(package_dir)
        video = profile.get("video", {}) if isinstance(profile.get("video"), dict) else {}
        component_video = config.get("video", {}) if isinstance(config.get("video"), dict) else {}
        runtime_video = {}
        try:
            runtime_config = load_runtime_config(resolve_runtime_config_path(config))
            runtime_video = runtime_config.get("video", {}) if isinstance(runtime_config.get("video"), dict) else {}
        except Exception:
            runtime_video = {}
        base_url = video.get("webrtc_base_url", "") or runtime_video.get("webrtc_base_url", "") or component_video.get("webrtc_base_url", "")
        runtime = config.get("runtime", {})
        runtime = runtime if isinstance(runtime, dict) else {}
        metadata = _safe_profile_metadata(profile, base_url, runtime)
        profile_metadata = []
        for item in get_games().all():
            item_profile = item.get("vision_profile")
            if not isinstance(item_profile, dict):
                continue
            item_video = item_profile.get("video", {})
            item_base = (
                item_video.get("webrtc_base_url", "") if isinstance(item_video, dict) else ""
            ) or runtime_video.get("webrtc_base_url", "") or component_video.get("webrtc_base_url", "")
            profile_metadata.append(_safe_profile_metadata(item_profile, item_base, runtime))
        metadata["profiles"] = profile_metadata
        return metadata
    except Exception:
        # Provider health already reports component/configuration failures;
        # metadata is optional and must never mask that primary signal.
        return {}


def _tts_provider(payload: dict[str, Any], game_id: str | None = None):
    """Compatibility wrapper for callers that still import this helper."""
    return TtsDispatcher(COMPONENTS, get_games()).provider(payload, game_id)


def _tts_dispatcher() -> TtsDispatcher:
    # Resolve on demand so tests and board operators can replace the registry
    # or environment without restarting this module object.
    return TtsDispatcher(COMPONENTS, get_games())


jobs: dict[str, ComponentJob] = {}
jobs_lock = threading.Lock()
active_job_id: str | None = None

# Authoritative game rounds.  One active round at a time mirrors the
# single-YOLO-job rule; creating a new round cancels a stale one so a
# browser refresh (which abandons its round) cannot block the next session.
rounds: dict[str, GameRound] = {}
rounds_lock = threading.Lock()

_ROUND_TIMEOUT_FALLBACK_SECONDS = JOB_TIMEOUT_SECONDS


def _round_adjudicate_fn(game_id: str):
    """Bridge a round's adjudicate action onto the shared provider pipeline."""

    def adjudicate(manifest, on_event, is_cancelled, on_log):
        return run_game(
            get_games(),
            game_id,
            on_log,
            is_cancelled,
            on_event,
            resolve_adjudication_timeout(
                require_game(get_games(), game_id), _ROUND_TIMEOUT_FALLBACK_SECONDS
            ),
            COMPONENTS,
        )

    return adjudicate


def create_round(game_id: str) -> GameRound:
    manifest = require_game(get_games(), game_id)
    if not isinstance(manifest.get("state_machine"), dict):
        raise InvalidRequestError(f"game {game_id} declares no state_machine")
    with rounds_lock:
        for existing in list(rounds.values()):
            if existing.status == "running":
                existing.cancel()
        # Finished rounds are dropped once the table grows past a small
        # watermark; snapshots stay queryable for the current session.
        finished = [rid for rid, item in rounds.items() if item.status != "running"]
        for stale_id in finished[: max(0, len(rounds) - 16)]:
            rounds.pop(stale_id, None)
        round_ = GameRound(
            game_id=game_id,
            manifest=manifest,
            adjudicate_fn=_round_adjudicate_fn(game_id),
            log=lambda line: print(f"[round:{round_.id[:8]}] {line}", flush=True),
        )
        rounds[round_.id] = round_
    round_.start()
    return round_


def _lookup_round(round_id: str) -> GameRound:
    with rounds_lock:
        round_ = rounds.get(round_id)
    if round_ is None:
        raise JobNotFoundError(round_id)
    return round_


def _shutdown_runtime_components() -> None:
    """Stop provider-owned resident workers before the backend exits."""
    with jobs_lock:
        active = jobs.get(active_job_id) if active_job_id else None
    if active is not None and active.status in {"queued", "running"}:
        active.cancel()

    # Providers own their runtime processes and may expose a provider-specific
    # shutdown hook.  Keep this seam optional so cloud/fixture providers do
    # not need lifecycle code just to participate in the registry.
    for component_id in COMPONENTS.ids():
        try:
            component = COMPONENTS.get(component_id)
            shutdown = getattr(component, "shutdown", None)
            if callable(shutdown):
                shutdown()
        except Exception as exc:
            print(f"[components] shutdown failed for {component_id}: {exc}", flush=True)

_ADJUDICATION_JOB_PREFIXES = ("/api/adjudicate/", "/api/analyze/")


def _adjudication_job_remainder(path: str) -> str | None:
    """Return the job-route suffix for the canonical or legacy endpoint."""
    for prefix in _ADJUDICATION_JOB_PREFIXES:
        if path.startswith(prefix):
            return path[len(prefix):]
    return None


def create_adjudication_job(game_id: str) -> ComponentJob:
    global active_job_id
    require_game(get_games(), game_id)  # raises GameNotFoundError / GameDisabledError
    with jobs_lock:
        if active_job_id:
            active = jobs.get(active_job_id)
            if active and active.status in {"queued", "running"}:
                raise JobAlreadyExistsError(active_job_id)

        def run_fn(on_log, is_cancelled, on_event):
            return run_game(
                get_games(), game_id, on_log, is_cancelled, on_event,
                resolve_adjudication_timeout(require_game(get_games(), game_id), JOB_TIMEOUT_SECONDS), COMPONENTS,
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

    @staticmethod
    def round_stream_delta(snapshot: dict[str, Any], after_sequence: int) -> dict[str, Any]:
        """Build a compact SSE update for one authoritative round."""
        return {
            "round_id": snapshot["round_id"],
            "game_id": snapshot["game_id"],
            "status": snapshot["status"],
            "state": snapshot["state"],
            "error": snapshot["error"],
            "result": snapshot["result"],
            "events": [
                event for event in snapshot["events"]
                if int(event.get("sequence", 0)) > after_sequence
            ],
            "event_sequence": snapshot["event_sequence"],
            "revision": snapshot["revision"],
        }

    def stream_round_events(self, round_: GameRound) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        snapshot = round_.snapshot()
        revision = snapshot["revision"]
        event_sequence = snapshot["event_sequence"]
        self.send_sse_snapshot(snapshot, "snapshot")
        try:
            while snapshot["status"] == "running":
                snapshot = round_.wait_for_update(revision, timeout=15.0)
                if snapshot["revision"] <= revision:
                    self.send_sse_snapshot({
                        "round_id": round_.id,
                        "revision": revision,
                        "event_sequence": event_sequence,
                    }, "heartbeat")
                    continue
                delta = self.round_stream_delta(snapshot, event_sequence)
                revision = snapshot["revision"]
                event_sequence = snapshot["event_sequence"]
                if snapshot["status"] != "running":
                    self.send_sse_snapshot(delta, "complete")
                    return
                self.send_sse_snapshot(delta)
            self.send_sse_snapshot(self.round_stream_delta(snapshot, event_sequence), "complete")
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.close_connection = True

    def send_round_speech_frames(self, round_: GameRound, directive_id: str) -> None:
        """Stream one directive's audio: manifest WAV or provider TTS frames."""
        directive = round_.find_directive(directive_id)
        if directive is None:
            self.send_error_json(DiceArenaError(
                f"speech directive not found: {directive_id}",
                "SPEECH_DIRECTIVE_NOT_FOUND",
                404,
            ))
            return
        game_id = round_.game_id
        provider = None
        speech_provider_id = None
        speech_payload: dict[str, Any] | None = None
        frame: bytes | None = None
        if directive.get("mode") == "audio":
            try:
                audio_path = resolve_game_audio_path(game_id, str(directive.get("audio") or ""), root=_GAMES_ROOT)
                frame = encode_audio_frame(audio_path.read_bytes())
            except FileNotFoundError as exc:
                self.send_error_json(DiceArenaError(
                    f"audio file not found: {exc}", "AUDIO_FILE_NOT_FOUND", 404
                ))
                return
            except (OSError, ValueError) as exc:
                self.send_error_json(DiceArenaError(str(exc), "AUDIO_CONFIG_ERROR", 400))
                return
        else:
            entry: dict[str, Any] = {"mode": str(directive.get("mode") or "tts_local")}
            if directive.get("provider"):
                entry["provider"] = str(directive["provider"])
            try:
                dispatcher = _tts_dispatcher()
                speech_provider_id = dispatcher.provider_id_for_speech_entry(entry, game_id)
                speech_payload = {
                    "game": game_id,
                    "text": str(directive.get("text") or ""),
                    "voice": str(directive.get("voice") or "default"),
                    "speed": directive.get("speed", 1.0),
                }
                provider = dispatcher.provider(
                    speech_payload, game_id=game_id, provider_id=speech_provider_id
                )
                if not callable(getattr(provider, "stream", None)):
                    raise InvalidRequestError(
                        f"TTS provider {provider.id} does not implement stream()"
                    )
            except DiceArenaError as exc:
                self.send_error_json(exc)
                return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-dice-arena-wav-stream")
        self.send_header("X-Dice-Speech-Directive", directive_id)
        self.send_header("X-Dice-Speech-Mode", str(directive.get("mode") or "tts_local"))
        if provider is not None:
            self.send_header("X-Dice-TTS-Provider", provider.id)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            if provider is not None and speech_payload is not None:
                def write_frame(audio: bytes) -> None:
                    self.wfile.write(encode_audio_frame(audio))
                    self.wfile.flush()

                _tts_dispatcher().stream(
                    speech_payload,
                    write_frame,
                    game_id=game_id,
                    provider_id=speech_provider_id,
                )
            elif frame is not None:
                self.wfile.write(frame)
            self.wfile.write(encode_end_frame())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            print(f"[round] speech stream directive={directive_id} failed: {exc}", flush=True)
            try:
                self.wfile.write(encode_error_frame(str(exc)))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_GET(self) -> None:  # noqa: N802
        parsed_url = urlsplit(self.path)
        path = parsed_url.path
        if path == "/api/health":
            component_items = COMPONENTS.all(include_health=True)
            tts_id = _selected_tts_id()
            tts_health = _provider_health(tts_id, "tts")
            remote_id = _selected_provider_id("dice", "tts_remote", "")
            remote_health = (
                _provider_health(remote_id, "tts") if remote_id else {
                    "id": "", "type": "tts", "role": "", "ok": False,
                    "configured": False,
                }
            )
            adjudicator_id = _selected_provider_id(
                "dice", "vision_adjudicator", "vision_yolov8_adjudicator"
            )
            adjudicator_health = _provider_health(
                adjudicator_id, "vision", "adjudicator"
            )
            adjudicator_health = {
                **adjudicator_health,
                **_vision_profile_metadata("dice", adjudicator_id),
            }
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
                "tts_remote_provider": remote_id,
                "tts_remote": {**remote_health, "configured": bool(remote_id)},
                "camera": "config.json",
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
            self.send_json({"games": get_games().public_all()})
            return
        if path.startswith("/api/game/rounds/"):
            remainder = path[len("/api/game/rounds/"):]
            round_id, _, suffix = remainder.partition("/")
            try:
                round_ = _lookup_round(round_id)
            except JobNotFoundError as exc:
                self.send_error_json(exc)
                return
            if suffix == "stream":
                self.stream_round_events(round_)
            else:
                self.send_json(round_.snapshot())
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
                dispatcher = _tts_dispatcher()
                provider = dispatcher.provider(payload)
                if not callable(getattr(provider, "stream", None)):
                    raise InvalidRequestError(
                        f"TTS provider {provider.id} does not implement stream()"
                    )
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
                self.wfile.write(encode_audio_frame(audio))
                self.wfile.flush()

            try:
                dispatcher.stream(payload, write_frame)
                self.wfile.write(encode_end_frame())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                print(f"[tts] stream provider={provider.id} failed: {exc}", flush=True)
                try:
                    self.wfile.write(encode_error_frame(str(exc)))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            return
        if path == "/api/tts/synthesize":
            try:
                payload = self.read_json()
                dispatcher = _tts_dispatcher()
                provider = dispatcher.provider(payload)
                audio, headers = dispatcher.synthesize(payload)
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
        if path == "/api/game/rounds":
            try:
                payload = self.read_json()
                game_id = str(payload.get("game") or "dice")
                round_ = create_round(game_id)
                self.send_json(round_.snapshot(), HTTPStatus.CREATED)
            except DiceArenaError as exc:
                self.send_error_json(exc)
            return
        if path.startswith("/api/game/rounds/"):
            remainder = path[len("/api/game/rounds/"):]
            round_id, _, suffix = remainder.partition("/")
            if suffix in {"intents", "cancel", "speech"}:
                try:
                    round_ = _lookup_round(round_id)
                except JobNotFoundError as exc:
                    self.send_error_json(exc)
                    return
                if suffix == "speech":
                    try:
                        payload = self.read_json()
                        directive_id = str(payload.get("directive_id") or "").strip()
                        if not directive_id:
                            raise InvalidRequestError("directive_id is required")
                    except DiceArenaError as exc:
                        self.send_error_json(exc)
                        return
                    self.send_round_speech_frames(round_, directive_id)
                    return
                if suffix == "cancel":
                    round_.cancel()
                    self.send_json(round_.snapshot())
                    return
                # intents
                try:
                    payload = self.read_json()
                    intent = str(payload.get("intent") or "").strip()
                    if not intent:
                        raise InvalidRequestError("intent is required")
                    self.send_json(round_.submit_intent(intent, payload))
                except DiceArenaError as exc:
                    self.send_error_json(exc)
                except Exception as exc:
                    self.send_error_json(DiceArenaError(str(exc), "ROUND_INTENT_ERROR", 500))
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
    shutdown_requested = threading.Event()

    def request_shutdown(signum: int, _frame: Any) -> None:
        if shutdown_requested.is_set():
            return
        shutdown_requested.set()
        print(f"Received signal {signum}; shutting down runtime components", flush=True)
        _shutdown_runtime_components()
        # ``serve_forever`` must be stopped from another thread when called
        # by a signal handler on the serving thread.
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown_runtime_components()
        server.server_close()


if __name__ == "__main__":
    main()
