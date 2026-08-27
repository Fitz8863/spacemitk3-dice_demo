"""Qwen3-TTS component: board-local llama-server text-to-speech.

Wraps the migrated Qwen3-TTS interactive client so the HTTP layer and future
game pipelines share the same punctuation-aware splitter and synthesis path.
The component owns no game logic: it turns text into WAV and nothing else.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from core.components import Component
from core.env import load_board_env

load_board_env()

ROOT = Path(__file__).resolve().parents[2]  # repo root (main/)
TTS_ROOT = ROOT / "tts" / "qwen3-tts"
TTS_URL = os.environ.get("DICE_TTS_URL", "http://127.0.0.1:18080").rstrip("/")
TTS_TIMEOUT_SECONDS = float(os.environ.get("DICE_TTS_TIMEOUT_SECONDS", "120"))
TTS_ENGINE = "qwen3-tts-k3-llama-server"
TTS_SPEAKER_FILE = "anke.spk.bin"
TTS_REQUEST_LOCK = threading.Lock()
_tts_interactive_module = None


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


def stream_tts(payload: dict[str, Any], write_frame: Callable[[bytes], None]) -> None:
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


class TtsQwen3(Component):
    id = "tts_qwen3"
    type = "tts"

    def health(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "ok": tts_health(),
            "url": TTS_URL,
            "engine": TTS_ENGINE,
            "speaker": TTS_SPEAKER_FILE,
        }

    def invoke(self, payload: dict[str, Any]) -> bytes:
        """Synthesize a single WAV; convenient for game pipelines / debugging."""
        audio, _ = synthesize_tts(payload)
        return audio
