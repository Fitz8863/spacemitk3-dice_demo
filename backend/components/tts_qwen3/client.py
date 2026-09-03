"""Small production HTTP client for the Qwen3-TTS runtime.

This module intentionally has no CLI, audio playback, or process management.
The component provider owns the Dice Arena contract; this client only knows
how to split text and request complete WAV responses from llama-server.
"""
from __future__ import annotations

import io
import json
import re
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass

from core.errors import TtsServiceError


@dataclass(frozen=True)
class QwenAudio:
    wav: bytes
    headers: dict[str, str]


def split_text(text: str, max_chars: int = 24) -> list[str]:
    """Split at natural punctuation and keep every request bounded."""
    max_chars = max(8, int(max_chars))
    text = re.sub(r"\s+", " ", str(text).strip())
    if not text:
        return []

    sentences = [
        part.strip()
        for part in re.findall(r".*?(?:[。！？!?；;\n]+|$)", text)
        if part.strip()
    ]
    chunks: list[str] = []
    terminal = re.compile(r"[。！？!?；;，,：:.]$")

    def finish(part: str) -> str:
        part = part.strip()
        if not part or terminal.search(part):
            return part
        return part + ("。" if re.search(r"[\u3400-\u9fff]", part) else ".")

    for sentence in sentences:
        if len(sentence) <= max_chars:
            chunks.append(finish(sentence))
            continue
        natural = [
            part.strip()
            for part in re.findall(r".*?(?:[，,：:]+|$)", sentence)
            if part.strip()
        ]
        if len(natural) > 1:
            chunks.extend(finish(part) for part in natural if finish(part))
            continue
        if " " in sentence:
            current: list[str] = []
            current_len = 0
            for word in sentence.split():
                extra = len(word) + (1 if current else 0)
                if current and current_len + extra > max_chars:
                    chunks.append(finish(" ".join(current)))
                    current = [word]
                    current_len = len(word)
                else:
                    current.append(word)
                    current_len += extra
            if current:
                chunks.append(finish(" ".join(current)))
        else:
            chunks.extend(
                finish(sentence[index : index + max_chars])
                for index in range(0, len(sentence), max_chars)
            )
    return chunks


class QwenClient:
    """HTTP adapter for one Qwen3-compatible speech endpoint."""

    def __init__(self, base_url: str, timeout_seconds: float, max_chunk_chars: int = 24):
        self.base_url = str(base_url).rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.max_chunk_chars = max(8, int(max_chunk_chars))

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=1.5) as response:
                return 200 <= response.status < 300
        except (OSError, urllib.error.URLError, TimeoutError):
            return False

    def synthesize(self, text: str, *, voice: str, speed: float) -> QwenAudio:
        body = json.dumps({
            "model": "qwen3-tts",
            "input": text,
            "voice": voice,
            "response_format": "wav",
            "speed": speed,
        }, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/v1/audio/speech",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                audio = response.read()
                headers = {
                    name: value
                    for name, value in response.headers.items()
                    if name.lower().startswith("x-tts-")
                }
                headers["Content-Type"] = response.headers.get("Content-Type", "audio/wav")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise TtsServiceError(f"Qwen3-TTS HTTP {exc.code}: {detail}") from exc
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            raise TtsServiceError(
                f"Qwen3-TTS unavailable at {self.base_url}: {exc}"
            ) from exc
        if len(audio) < 44 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
            raise TtsServiceError("Qwen3-TTS did not return a valid WAV")
        return QwenAudio(audio, headers)


def wav_duration(audio: bytes) -> float:
    with wave.open(io.BytesIO(audio), "rb") as wav_file:
        return wav_file.getnframes() / max(1, wav_file.getframerate())
