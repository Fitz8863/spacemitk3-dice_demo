"""Qwen3-TTS provider: one implementation of the common TTS contract."""
from __future__ import annotations

import threading
from typing import Any, Callable

from core.tts import TtsProvider
from core.errors import TtsValidationError
from components.tts_qwen3.client import QwenClient, split_text
from components.tts_qwen3.settings import load_settings


SETTINGS = load_settings()
TTS_ROOT = SETTINGS.root
TTS_MODEL_DIR = SETTINGS.model_dir
TTS_URL = SETTINGS.url
TTS_TIMEOUT_SECONDS = SETTINGS.timeout_seconds
TTS_ENGINE = "qwen3-tts-k3-llama-server"
TTS_SPEAKER_FILE = SETTINGS.speaker_file
TTS_DEFAULT_VOICE = SETTINGS.default_voice
TTS_DEFAULT_SPEED = SETTINGS.default_speed
TTS_CHUNK_CHARS = SETTINGS.chunk_chars
TTS_SPEAKER_PATH = TTS_MODEL_DIR / TTS_SPEAKER_FILE
TTS_REQUEST_LOCK = threading.Lock()
QWEN_CLIENT = QwenClient(TTS_URL, TTS_TIMEOUT_SECONDS, TTS_CHUNK_CHARS)


class TtsQwen3(TtsProvider):
    id = "tts_qwen3"
    type = "tts"
    name = "Qwen3-TTS"
    version = "1.0"

    def health(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "ok": QWEN_CLIENT.health(),
            "url": TTS_URL,
            "engine": TTS_ENGINE,
            "speaker": TTS_SPEAKER_FILE,
            "speaker_file": TTS_SPEAKER_FILE,
            "speaker_path": str(TTS_SPEAKER_PATH) if TTS_SPEAKER_PATH else None,
            "supports_voice_clone": False,
        }

    def stream(self, payload: dict[str, Any], write_frame: Callable[[bytes], None]) -> None:
        text, voice, speed = self.validate(payload)
        if "speed" not in payload:
            speed = TTS_DEFAULT_SPEED
        voice = TTS_DEFAULT_VOICE if voice == "default" else voice
        chunks = split_text(text, TTS_CHUNK_CHARS)
        if not chunks:
            raise TtsValidationError("text is empty after normalization")
        print(f"[tts] stream start provider={self.id} chunks={len(chunks)}", flush=True)
        with TTS_REQUEST_LOCK:
            for index, chunk_text in enumerate(chunks, start=1):
                audio = QWEN_CLIENT.synthesize(chunk_text, voice=voice, speed=speed).wav
                print(f"[tts] generated provider={self.id} frame={index}/{len(chunks)} bytes={len(audio)}", flush=True)
                write_frame(audio)
        print(f"[tts] stream complete provider={self.id}", flush=True)

    def synthesize(self, payload: dict[str, Any]) -> tuple[bytes, dict[str, str]]:
        text, voice, speed = self.validate(payload)
        if "speed" not in payload:
            speed = TTS_DEFAULT_SPEED
        voice = TTS_DEFAULT_VOICE if voice == "default" else voice
        with TTS_REQUEST_LOCK:
            result = QWEN_CLIENT.synthesize(text, voice=voice, speed=speed)
        return result.wav, result.headers
