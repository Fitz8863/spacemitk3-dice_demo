"""Dice Arena adapter for the board-local MOSS-TTS-Nano runtime."""
from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from core.env import load_board_env
from core.errors import TtsServiceError, TtsValidationError
from core.tts import TtsProvider

load_board_env()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = str(PROJECT_ROOT / "tts" / "moss-tts-nano")
DEFAULT_URL = "http://127.0.0.1:18082"
MOSS_ROOT = os.environ.get("DICE_MOSS_TTS_ROOT", DEFAULT_ROOT).rstrip("/")
MOSS_URL = os.environ.get("DICE_MOSS_TTS_URL", DEFAULT_URL).rstrip("/")
MOSS_TIMEOUT_SECONDS = float(os.environ.get("DICE_MOSS_TTS_TIMEOUT_SECONDS", "120"))
MOSS_VOICE = os.environ.get("DICE_MOSS_TTS_VOICE", "Junhao").strip() or "Junhao"
MOSS_ENGINE = "moss-tts-nano-spacemit-ep"


def _json_request(url: str, *, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        raise TtsServiceError(f"provider=tts_moss_nano unavailable at {MOSS_URL}: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TtsServiceError("provider=tts_moss_nano returned invalid health JSON") from exc
    if not isinstance(payload, dict):
        raise TtsServiceError("provider=tts_moss_nano returned a non-object health response")
    return payload


def _read_exact(response: Any, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(size)
    while remaining:
        chunk = response.read(remaining)
        if not chunk:
            raise TtsServiceError("provider=tts_moss_nano stream ended before a complete frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _validate_wav(audio: bytes) -> None:
    if len(audio) < 44 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
        raise TtsServiceError("provider=tts_moss_nano did not return a valid WAV")


class TtsMossNano(TtsProvider):
    """Adapt direct-runtime MOSS synthesis and its framed WAV stream."""

    id = "tts_moss_nano"
    type = "tts"
    name = "MOSS-TTS-Nano (SpaceMIT EP)"
    version = "1.1"

    def health(self) -> dict[str, Any]:
        try:
            remote = _json_request(f"{MOSS_URL}/health", timeout=1.5)
        except TtsServiceError as exc:
            return {
                "id": self.id,
                "type": self.type,
                "ok": False,
                "ready": False,
                "url": MOSS_URL,
                "root": MOSS_ROOT,
                "engine": MOSS_ENGINE,
                "voice": MOSS_VOICE,
                "supports_speed": False,
                "supports_stream": True,
                "error": str(exc),
            }

        result = dict(remote)
        result.update({
            "id": self.id,
            "type": self.type,
            "url": MOSS_URL,
            "root": MOSS_ROOT,
            "engine": MOSS_ENGINE,
            "voice": MOSS_VOICE,
            "supports_speed": False,
            "supports_stream": True,
        })
        result.setdefault("ok", bool(result.get("ready", False)))
        return result

    def _validate_payload(self, payload: dict[str, Any]) -> tuple[str, str]:
        text, voice, speed = self.validate(payload)
        if abs(speed - 1.0) > 1e-6:
            raise TtsValidationError(
                "tts_moss_nano currently supports only speed=1.0; "
                "the MOSS SpaceMIT runtime has no speed parameter"
            )
        requested_voice = MOSS_VOICE if voice == "default" else voice
        if requested_voice != MOSS_VOICE:
            raise TtsValidationError(
                f"tts_moss_nano is running with voice {MOSS_VOICE!r}; "
                f"requested voice {requested_voice!r} requires restarting the provider"
            )
        return text, requested_voice

    @staticmethod
    def _body(text: str, voice: str) -> bytes:
        return json.dumps({
            "model": "moss-tts-nano",
            "input": text,
            "voice": voice,
            "response_format": "wav",
            "speed": 1.0,
        }, ensure_ascii=False).encode("utf-8")

    def synthesize(self, payload: dict[str, Any]) -> tuple[bytes, dict[str, str]]:
        text, voice = self._validate_payload(payload)
        request = urllib.request.Request(
            f"{MOSS_URL}/v1/audio/speech",
            data=self._body(text, voice),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=MOSS_TIMEOUT_SECONDS) as response:
                audio = response.read()
                content_type = response.headers.get("Content-Type", "audio/wav")
                headers = {
                    name: value for name, value in response.headers.items()
                    if name.lower().startswith("x-tts-")
                }
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise TtsServiceError(f"provider={self.id} HTTP {exc.code}: {detail}") from exc
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            raise TtsServiceError(f"provider={self.id} unavailable at {MOSS_URL}: {exc}") from exc

        _validate_wav(audio)
        headers["Content-Type"] = content_type
        return audio, headers

    def stream(self, payload: dict[str, Any], write_frame: Callable[[bytes], None]) -> None:
        text, voice = self._validate_payload(payload)
        request = urllib.request.Request(
            f"{MOSS_URL}/v1/audio/speech/stream",
            data=self._body(text, voice),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=MOSS_TIMEOUT_SECONDS) as response:
                while True:
                    length = int.from_bytes(_read_exact(response, 4), "big")
                    if length == 0:
                        return
                    if length == 0xFFFFFFFF:
                        message_length = int.from_bytes(_read_exact(response, 4), "big")
                        if message_length > 64 * 1024:
                            raise TtsServiceError("provider=tts_moss_nano returned an oversized error frame")
                        message = _read_exact(response, message_length).decode("utf-8", errors="replace")
                        raise TtsServiceError(message or "MOSS streaming synthesis failed")
                    if length < 44 or length > 32 * 1024 * 1024:
                        raise TtsServiceError(f"provider=tts_moss_nano returned invalid frame length: {length}")
                    audio = _read_exact(response, length)
                    _validate_wav(audio)
                    write_frame(audio)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise TtsServiceError(f"provider={self.id} HTTP {exc.code}: {detail}") from exc
        except (OSError, urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            raise TtsServiceError(f"provider={self.id} stream unavailable at {MOSS_URL}: {exc}") from exc
