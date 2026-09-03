"""Dice Arena adapter for the board-local MOSS-TTS-Nano runtime."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable
from urllib.parse import urlsplit

from core.errors import TtsServiceError, TtsValidationError
from core.tts import TtsProvider
from core.tts_protocol import TtsProtocolError, iter_audio_frames, validate_wav
from components.tts_moss_nano.settings import load_settings


SETTINGS = load_settings()
MOSS_ROOT = str(SETTINGS.root).rstrip("/")
MOSS_URL = SETTINGS.base_url
MOSS_TIMEOUT_SECONDS = SETTINGS.request_timeout_seconds
MOSS_VOICE = SETTINGS.voice
MOSS_VOICE_MODE = SETTINGS.voice_mode
MOSS_REFERENCE_AUDIO = str(SETTINGS.reference_audio) if SETTINGS.reference_audio else ""
MOSS_ENGINE = "moss-tts-nano-spacemit-ep"


def _moss_url(path: str) -> str:
    """Build a bridge URL after verifying the configured origin is loopback."""
    parsed = urlsplit(MOSS_URL)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise TtsServiceError(
            "tts_moss_nano bridge base_url must be an http:// loopback address"
        )
    return f"{MOSS_URL}{path}"


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


class TtsMossNano(TtsProvider):
    """Adapt direct-runtime MOSS synthesis and its framed WAV stream."""

    id = "tts_moss_nano"
    type = "tts"
    name = "MOSS-TTS-Nano (SpaceMIT EP)"
    version = "1.1"

    def health(self) -> dict[str, Any]:
        try:
            remote = _json_request(_moss_url("/health"), timeout=1.5)
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
                "voice_mode": MOSS_VOICE_MODE,
                "reference_audio": MOSS_REFERENCE_AUDIO or None,
                "supports_voice_clone": True,
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
            "voice_mode": MOSS_VOICE_MODE,
            "reference_audio": MOSS_REFERENCE_AUDIO or None,
            "supports_voice_clone": True,
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
            _moss_url("/v1/audio/speech"),
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

        try:
            validate_wav(audio)
        except TtsProtocolError as exc:
            raise TtsServiceError(f"provider={self.id} did not return a valid WAV") from exc
        headers["Content-Type"] = content_type
        return audio, headers

    def stream(self, payload: dict[str, Any], write_frame: Callable[[bytes], None]) -> None:
        text, voice = self._validate_payload(payload)
        request = urllib.request.Request(
            _moss_url("/v1/audio/speech/stream"),
            data=self._body(text, voice),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=MOSS_TIMEOUT_SECONDS) as response:
                for audio in iter_audio_frames(response):
                    write_frame(audio)
        except TtsProtocolError as exc:
            raise TtsServiceError(f"provider={self.id} stream protocol error: {exc}") from exc
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise TtsServiceError(f"provider={self.id} HTTP {exc.code}: {detail}") from exc
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            raise TtsServiceError(f"provider={self.id} stream unavailable at {MOSS_URL}: {exc}") from exc
