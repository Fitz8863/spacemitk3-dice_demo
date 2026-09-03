"""Shared length-prefixed WAV framing for Dice Arena TTS streams."""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any


TTS_STREAM_END = 0
TTS_STREAM_ERROR = 0xFFFFFFFF
MAX_AUDIO_FRAME_BYTES = 32 * 1024 * 1024
MAX_ERROR_BYTES = 64 * 1024


class TtsProtocolError(ValueError):
    """Raised when a provider returns an invalid framed audio stream."""


def validate_wav(audio: bytes) -> bytes:
    if len(audio) < 44 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
        raise TtsProtocolError("TTS provider did not return a valid WAV")
    return audio


def encode_audio_frame(audio: bytes) -> bytes:
    validate_wav(audio)
    if len(audio) > MAX_AUDIO_FRAME_BYTES:
        raise TtsProtocolError(f"TTS audio frame is too large: {len(audio)} bytes")
    return len(audio).to_bytes(4, "big") + audio


def encode_end_frame() -> bytes:
    return TTS_STREAM_END.to_bytes(4, "big")


def encode_error_frame(message: str) -> bytes:
    payload = str(message).encode("utf-8")[:2000]
    return (
        TTS_STREAM_ERROR.to_bytes(4, "big")
        + len(payload).to_bytes(4, "big")
        + payload
    )


def read_exact(response: Any, size: int) -> bytes:
    if size < 0:
        raise TtsProtocolError("negative TTS frame size")
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = response.read(remaining)
        if not chunk:
            raise TtsProtocolError("TTS stream ended before a complete frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def iter_audio_frames(response: Any) -> Iterator[bytes]:
    """Yield complete WAV frames from a provider's framed HTTP response."""
    while True:
        length = int.from_bytes(read_exact(response, 4), "big")
        if length == TTS_STREAM_END:
            return
        if length == TTS_STREAM_ERROR:
            message_length = int.from_bytes(read_exact(response, 4), "big")
            if message_length > MAX_ERROR_BYTES:
                raise TtsProtocolError("TTS error frame is too large")
            message = read_exact(response, message_length).decode(
                "utf-8", errors="replace"
            )
            raise TtsProtocolError(message or "TTS provider stream failed")
        if length < 44 or length > MAX_AUDIO_FRAME_BYTES:
            raise TtsProtocolError(f"invalid TTS audio frame length: {length}")
        yield validate_wav(read_exact(response, length))
