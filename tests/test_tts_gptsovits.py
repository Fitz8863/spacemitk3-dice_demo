from __future__ import annotations

import io
import json
import sys
import unittest
import urllib.error
import wave
from pathlib import Path
from unittest.mock import patch

from components.tts_gptsovits import provider as gptsovits_provider


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.components import _validate_manifest  # noqa: E402
from core.errors import TtsServiceError  # noqa: E402
from core.tts_config import validate_tts_component_config  # noqa: E402
from core.tts_protocol import validate_wav  # noqa: E402
from components.tts_gptsovits.provider import (  # noqa: E402
    GPTSOVITS_URL,
    TtsGptSovits,
    _pcm_frame,
)


def _wav_bytes(pcm: bytes, sample_rate: int = 32000, channels: int = 1) -> bytes:
    return _pcm_frame(pcm, sample_rate, channels)


class FakeResponse:
    def __init__(self, chunks: list[bytes], status: int = 200):
        self._chunks = list(chunks)
        self.status = status

    def read(self, size: int = -1) -> bytes:
        if size in (-1, None):
            remaining = b"".join(self._chunks)
            self._chunks = []
            return remaining
        if not self._chunks:
            return b""
        chunk = self._chunks.pop(0)
        return chunk[:size] if size < len(chunk) else chunk

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _http_error(code: int, message: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        GPTSOVITS_URL,
        code,
        "Bad Request",
        {"Content-Type": "application/json"},
        io.BytesIO(json.dumps({"message": message}).encode("utf-8")),
    )


class ProviderContractTests(unittest.TestCase):
    def test_manifest_and_config_are_valid(self):
        manifest_path = ROOT / "backend" / "components" / "tts_gptsovits" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validated = _validate_manifest(manifest_path, manifest)
        self.assertEqual(validated["id"], "tts_gptsovits")
        self.assertEqual(validated["type"], "tts")
        self.assertNotIn("lifecycle", validated)

        config = json.loads(
            (ROOT / "backend" / "components" / "tts_gptsovits" / "config.json").read_text(
                encoding="utf-8"
            )
        )
        validate_tts_component_config(config)
        self.assertEqual(config["runtime"]["kind"], "external")
        self.assertTrue(config["runtime"]["base_url"].startswith("http"))

    def test_stream_wraps_pcm_chunks_into_wav_frames(self):
        chunks = [b"\x01\x02" * 100, b"\x03\x04" * 50]
        bodies = []

        def post_stream(url, body, timeout):
            bodies.append((url, json.loads(body.decode("utf-8"))))
            return FakeResponse(chunks)

        frames = []
        TtsGptSovits(post_stream=post_stream).stream(
            {"text": "你好", "voice": "default", "speed": 1.0}, frames.append
        )
        self.assertEqual(len(frames), 2)
        for frame in frames:
            validate_wav(frame)
            self.assertEqual(frame[:4], b"RIFF")
        self.assertEqual(bodies[0][0], f"{GPTSOVITS_URL}/tts")
        self.assertEqual(bodies[0][1]["streaming_mode"], 3)
        self.assertEqual(bodies[0][1]["media_type"], "raw")

    def test_stream_carries_odd_trailing_byte_across_chunks(self):
        chunks = [b"\x01\x02\x03", b"\x04\x05"]

        def post_stream(url, body, timeout):
            return FakeResponse(chunks)

        frames = []
        TtsGptSovits(post_stream=post_stream).stream(
            {"text": "你好", "speed": 1.0}, frames.append
        )
        # 5 bytes total: the odd trailing byte can never form a complete
        # int16 sample, so it stays in the carry buffer and is dropped at
        # end of stream. Each frame carries exactly 2 aligned samples here.
        self.assertEqual(len(frames), 2)
        self.assertEqual(len(frames[0]), 44 + 2)
        self.assertEqual(len(frames[1]), 44 + 2)

    def test_stream_maps_default_voice_and_clamps_speed(self):
        bodies = []

        def post_stream(url, body, timeout):
            bodies.append(json.loads(body.decode("utf-8")))
            return FakeResponse([b"\x01\x02" * 10])

        provider = TtsGptSovits(post_stream=post_stream)
        provider.stream({"text": "你好", "voice": "default", "speed": 4.0}, lambda _f: None)
        provider.stream({"text": "你好", "speed": 0.1}, lambda _f: None)
        self.assertEqual(bodies[0]["voice"], "demo_female_zh")
        self.assertEqual(bodies[0]["speed"], 2.0)
        self.assertEqual(bodies[1]["speed"], 0.5)

    def test_stream_passes_explicit_voice_through(self):
        bodies = []

        def post_stream(url, body, timeout):
            bodies.append(json.loads(body.decode("utf-8")))
            return FakeResponse([b"\x01\x02" * 10])

        TtsGptSovits(post_stream=post_stream).stream(
            {"text": "你好", "voice": "custom_voice", "speed": 1.0}, lambda _f: None
        )
        self.assertEqual(bodies[0]["voice"], "custom_voice")

    def test_stream_turns_http_400_message_into_service_error(self):
        def post_stream(url, body, timeout):
            raise _http_error(400, "ref_audio_path is required")

        with self.assertRaisesRegex(TtsServiceError, "ref_audio_path is required"):
            TtsGptSovits(post_stream=post_stream).stream(
                {"text": "你好", "speed": 1.0}, lambda _f: None
            )

    def test_stream_lets_broken_pipe_reach_the_server_disconnect_path(self):
        def post_stream(url, body, timeout):
            return FakeResponse([b"\x01\x02" * 1000, b"\x03\x04" * 1000])

        def write_frame(_frame):
            raise BrokenPipeError()

        with self.assertRaises(BrokenPipeError):
            TtsGptSovits(post_stream=post_stream).stream(
                {"text": "你好", "speed": 1.0}, write_frame
            )

    def test_stream_reports_empty_stream_as_service_error(self):
        def post_stream(url, body, timeout):
            return FakeResponse([])

        with self.assertRaisesRegex(TtsServiceError, "empty audio stream"):
            TtsGptSovits(post_stream=post_stream).stream(
                {"text": "你好", "speed": 1.0}, lambda _f: None
            )

    def test_request_body_includes_configured_sampling_defaults(self):
        """The checked-in config pins the engine's documented defaults."""
        bodies = []

        def post_stream(url, body, timeout):
            bodies.append(json.loads(body.decode("utf-8")))
            return FakeResponse([b"\x01\x02" * 10])

        TtsGptSovits(post_stream=post_stream).stream(
            {"text": "你好", "speed": 1.0}, lambda _f: None
        )
        self.assertEqual(bodies[0]["repetition_penalty"], 1.35)
        self.assertEqual(bodies[0]["seed"], -1)
        self.assertEqual(bodies[0]["top_k"], 15)
        self.assertEqual(bodies[0]["fragment_interval"], 0.3)

    def test_request_body_omits_unconfigured_sampling_keys(self):
        """Keys removed from the config fall back to the engine defaults."""
        bodies = []

        def post_stream(url, body, timeout):
            bodies.append(json.loads(body.decode("utf-8")))
            return FakeResponse([b"\x01\x02" * 10])

        with patch.object(gptsovits_provider, "_REQUEST_SAMPLING", {}):
            TtsGptSovits(post_stream=post_stream).stream(
                {"text": "你好", "speed": 1.0}, lambda _f: None
            )
        self.assertNotIn("repetition_penalty", bodies[0])
        self.assertNotIn("seed", bodies[0])
        self.assertIn("text", bodies[0])

    def test_health_ok_and_failure(self):
        ok_provider = TtsGptSovits(get_json=lambda url, timeout: {"voices": []})
        self.assertTrue(ok_provider.health()["ok"])

        def failing_get(url, timeout):
            raise OSError("connection refused")

        failed = TtsGptSovits(get_json=failing_get).health()
        self.assertFalse(failed["ok"])
        self.assertIn("connection refused", failed["error"])

    def test_synthesize_returns_complete_wav(self):
        expected = _wav_bytes(b"\x01\x02" * 320)

        def post_stream(url, body, timeout):
            payload = json.loads(body.decode("utf-8"))
            self.assertEqual(payload["streaming_mode"], 0)
            self.assertEqual(payload["media_type"], "wav")
            return FakeResponse([expected])

        audio, headers = TtsGptSovits(post_stream=post_stream).synthesize(
            {"text": "你好", "speed": 1.0}
        )
        self.assertEqual(audio, expected)
        self.assertEqual(headers["Content-Type"], "audio/wav")
        validate_wav(audio)


if __name__ == "__main__":
    unittest.main()
