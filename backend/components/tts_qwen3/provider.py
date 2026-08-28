"""Qwen3-TTS provider: one implementation of the common TTS contract."""
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

from core.tts import TtsProvider
from core.env import load_board_env
from core.errors import TtsServiceError, TtsValidationError

load_board_env()

ROOT = Path(__file__).resolve().parents[3]
TTS_ROOT = ROOT / "tts" / "qwen3-tts"
TTS_URL = os.environ.get("DICE_QWEN3_TTS_URL", os.environ.get("DICE_TTS_URL", "http://127.0.0.1:18080")).rstrip("/")
TTS_TIMEOUT_SECONDS = float(os.environ.get("DICE_QWEN3_TTS_TIMEOUT_SECONDS", os.environ.get("DICE_TTS_TIMEOUT_SECONDS", "120")))
TTS_ENGINE = "qwen3-tts-k3-llama-server"
TTS_SPEAKER_FILE = os.environ.get("DICE_QWEN3_TTS_SPEAKER", "anke.spk.bin")
TTS_REQUEST_LOCK = threading.Lock()
_tts_interactive_module = None


def _load_interactive_client():
    global _tts_interactive_module
    if _tts_interactive_module is not None:
        return _tts_interactive_module
    path = TTS_ROOT / "qwen3_tts_interactive.py"
    spec = importlib.util.spec_from_file_location("dice_arena_qwen3_tts_interactive", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load interactive TTS client: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    _tts_interactive_module = module
    return module



def _health_ok() -> bool:
    try:
        with urllib.request.urlopen(f"{TTS_URL}/health", timeout=1.5) as response:
            return 200 <= response.status < 300
    except (OSError, urllib.error.URLError):
        return False


class TtsQwen3(TtsProvider):
    id = "tts_qwen3"
    type = "tts"
    name = "Qwen3-TTS"
    version = "1.0"

    def health(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "ok": _health_ok(),
            "url": TTS_URL,
            "engine": TTS_ENGINE,
            "speaker": TTS_SPEAKER_FILE,
        }

    def stream(self, payload: dict[str, Any], write_frame: Callable[[bytes], None]) -> None:
        text, voice, speed = self.validate(payload)
        client = _load_interactive_client()
        chunks = client.split_text(text)
        if not chunks:
            raise TtsValidationError("text is empty after normalization")
        print(f"[tts] stream start provider={self.id} chunks={len(chunks)}", flush=True)
        with TTS_REQUEST_LOCK:
            for index, chunk_text in enumerate(chunks, start=1):
                audio = client.synthesize(chunk_text, voice=voice, speed=speed).wav
                print(f"[tts] generated provider={self.id} frame={index}/{len(chunks)} bytes={len(audio)}", flush=True)
                write_frame(audio)
        print(f"[tts] stream complete provider={self.id}", flush=True)

    def synthesize(self, payload: dict[str, Any]) -> tuple[bytes, dict[str, str]]:
        text, voice, speed = self.validate(payload)
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
        try:
            with TTS_REQUEST_LOCK:
                with urllib.request.urlopen(request, timeout=TTS_TIMEOUT_SECONDS) as response:
                    audio = response.read()
                    headers = {
                        name: value for name, value in response.headers.items()
                        if name.lower().startswith("x-tts-")
                    }
                    content_type = response.headers.get("Content-Type", "audio/wav")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise TtsServiceError(f"provider={self.id} HTTP {exc.code}: {detail}") from exc
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            raise TtsServiceError(f"provider={self.id} unavailable at {TTS_URL}: {exc}") from exc
        if len(audio) < 44 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
            raise TtsServiceError(f"provider={self.id} did not return a valid WAV")
        headers["Content-Type"] = content_type
        return audio, headers
