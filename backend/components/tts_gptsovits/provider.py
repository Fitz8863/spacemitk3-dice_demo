"""Remote GPT-SoVITS v2ProPlus streaming TTS adapter.

The engine runs on a separate GPU host inside the Tailscale network and is
managed entirely by its own ``start.sh``; this component is a pure HTTP
client with no local lifecycle.  The 9873 ``/tts`` endpoint streams
``streaming_mode=3`` audio as raw int16 LE mono PCM (32 kHz), while the Dice
Arena frame protocol expects one complete WAV per frame - so this adapter
wraps every received PCM chunk in a self-built 44-byte WAV header and hands
it straight to ``write_frame``.  The browser playback path stays unchanged.

The request target is an operator-pinned origin: every URL is built from the
single ``runtime.base_url`` in ``config.json``, restricted to the http(s)
scheme, that exact host, and a fixed path allowlist.  Redirects are refused
so a compromised peer cannot bounce a request elsewhere.
"""
from __future__ import annotations

import io
import json
import threading
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from core.errors import TtsServiceError, TtsValidationError
from core.tts import TtsProvider
from core.tts_config import config_value, load_component_config


COMPONENT_DIR = Path(__file__).resolve().parent
CONFIG = load_component_config(COMPONENT_DIR)
GPTSOVITS_URL = str(config_value(CONFIG, "runtime", "base_url", default="")).rstrip("/")
GPTSOVITS_TEXT_LANG = str(config_value(CONFIG, "request", "text_lang", default="zh"))
GPTSOVITS_STREAMING_MODE = int(config_value(CONFIG, "request", "streaming_mode", default=3))
GPTSOVITS_SPLIT_METHOD = str(config_value(CONFIG, "request", "text_split_method", default="cut5"))
GPTSOVITS_TIMEOUT_SECONDS = float(config_value(CONFIG, "request", "timeout_seconds", default=120))
GPTSOVITS_VOICE = str(config_value(CONFIG, "voice", "name", default="")).strip()
GPTSOVITS_SAMPLE_RATE = int(config_value(CONFIG, "audio", "sample_rate", default=32000))
GPTSOVITS_CHANNELS = int(config_value(CONFIG, "audio", "channels", default=1))
GPTSOVITS_ENGINE = "gpt-sovits-v2proplus"

# Optional quality knobs (see TTS接口文档.md §3).  Keys left out of the
# component config are omitted from requests, so the engine's own defaults
# apply; protocol-critical fields (media_type/streaming_mode) stay in code.
_SAMPLING_KEYS = (
    "top_k", "top_p", "temperature", "repetition_penalty", "seed", "fragment_interval",
)
_REQUEST_SAMPLING: dict[str, Any] = {}
if isinstance(CONFIG.get("request"), dict):
    for _key in _SAMPLING_KEYS:
        _value = CONFIG["request"].get(_key)
        if _value is not None:
            _REQUEST_SAMPLING[_key] = _value

# The upstream service is single-instance serial: concurrent requests queue
# server-side, so serializing here keeps perceived latency predictable.
GPTSOVITS_REQUEST_LOCK = threading.Lock()

# Fixed egress surface: only these paths on the configured origin exist.
_ALLOWED_PATHS = frozenset({"/tts", "/voices"})
_BASE = urlsplit(GPTSOVITS_URL)
_ALLOWED_SCHEMES = frozenset({"http", "https"})
_ALLOWED_HOST = _BASE.hostname or ""
_ALLOWED_PORT = _BASE.port


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: Any, msg: Any, headers: Any, newurl: Any) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)

PostStream = Callable[[str, bytes, float], Any]
GetJson = Callable[[str, float], Any]


def _default_post_stream(url: str, body: bytes, timeout: float) -> Any:
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return _OPENER.open(request, timeout=timeout)


def _default_get_json(url: str, timeout: float) -> Any:
    with _OPENER.open(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _api_url(path: str) -> str:
    """Build a request URL pinned to the configured origin and path allowlist.

    ``path`` only ever comes from module constants, never from request input;
    the guard keeps the egress surface fixed even if that ever changes.
    """
    if path not in _ALLOWED_PATHS:
        raise TtsServiceError(f"tts_gptsovits path is not allowed: {path!r}")
    parsed = urlsplit(GPTSOVITS_URL)
    if (
        parsed.scheme not in _ALLOWED_SCHEMES
        or not parsed.netloc
        or parsed.hostname != _ALLOWED_HOST
        or parsed.port != _ALLOWED_PORT
    ):
        raise TtsServiceError("tts_gptsovits request URL must target the configured service origin")
    return f"{GPTSOVITS_URL}{path}"


def _pcm_frame(pcm: bytes, sample_rate: int, channels: int) -> bytes:
    """Wrap one raw int16 LE PCM chunk in a self-contained 44-byte WAV header."""
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return output.getvalue()


class TtsGptSovits(TtsProvider):
    id = "tts_gptsovits"
    type = "tts"
    name = "GPT-SoVITS v2ProPlus (streaming)"
    version = "1.0"

    def __init__(self, manifest: dict[str, Any] | None = None, *, post_stream: PostStream | None = None, get_json: GetJson | None = None) -> None:
        super().__init__(manifest)
        self._post_stream = post_stream or _default_post_stream
        self._get_json = get_json or _default_get_json

    def health(self) -> dict[str, Any]:
        try:
            self._get_json(_api_url("/voices"), timeout=1.5)
        except Exception as exc:
            return {
                "id": self.id,
                "type": self.type,
                "ok": False,
                "url": GPTSOVITS_URL,
                "engine": GPTSOVITS_ENGINE,
                "error": str(exc),
            }
        return {
            "id": self.id,
            "type": self.type,
            "ok": True,
            "url": GPTSOVITS_URL,
            "engine": GPTSOVITS_ENGINE,
            "voice": GPTSOVITS_VOICE,
            "supports_speed": True,
            "supports_stream": True,
            "supports_voice_clone": True,
        }

    def _prepare(self, payload: dict[str, Any]) -> tuple[str, str, float]:
        text, voice, speed = self.validate(payload)
        # The engine documents 0.5~2.0 as its usable range.
        speed = max(0.5, min(2.0, speed))
        resolved_voice = voice if voice not in ("", "default") else GPTSOVITS_VOICE
        if not resolved_voice:
            raise TtsValidationError("tts_gptsovits requires a voice name (config voice.name)")
        return text, resolved_voice, speed

    @staticmethod
    def _request_body(
        text: str,
        voice: str,
        speed: float,
        *,
        streaming_mode: int,
        media_type: str | None = None,
    ) -> bytes:
        body: dict[str, Any] = {
            "text": text,
            "voice": voice,
            "text_lang": GPTSOVITS_TEXT_LANG,
            "streaming_mode": streaming_mode,
            "speed": speed,
            "text_split_method": GPTSOVITS_SPLIT_METHOD,
        }
        body.update(_REQUEST_SAMPLING)
        if media_type is not None:
            body["media_type"] = media_type
        return json.dumps(body, ensure_ascii=False).encode("utf-8")

    def stream(self, payload: dict[str, Any], write_frame: Callable[[bytes], None]) -> None:
        text, voice, speed = self._prepare(payload)
        body = self._request_body(
            text,
            voice,
            speed,
            streaming_mode=GPTSOVITS_STREAMING_MODE,
            media_type="raw",
        )
        url = _api_url("/tts")
        print(f"[tts] stream start provider={self.id} voice={voice} bytes={len(body)}", flush=True)
        frames = 0
        with GPTSOVITS_REQUEST_LOCK:
            try:
                with self._post_stream(url, body, GPTSOVITS_TIMEOUT_SECONDS) as response:
                    # media_type=raw means the body is pure PCM from byte 0;
                    # an odd trailing byte is carried into the next chunk so
                    # every wrapped frame keeps int16 alignment.
                    carry = b""
                    while True:
                        chunk = response.read(65536)
                        if not chunk:
                            break
                        data = carry + chunk
                        if len(data) % 2:
                            carry = data[-1:]
                            data = data[:-1]
                        else:
                            carry = b""
                        if data:
                            write_frame(_pcm_frame(data, GPTSOVITS_SAMPLE_RATE, GPTSOVITS_CHANNELS))
                            frames += 1
            except (BrokenPipeError, ConnectionResetError):
                # The browser stopped listening; re-raise so the server's
                # quiet disconnect path applies and the upstream connection
                # is closed by the ``with`` block above.
                raise
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                raise TtsServiceError(f"provider={self.id} HTTP {exc.code}: {detail}") from exc
            except (OSError, urllib.error.URLError, TimeoutError) as exc:
                raise TtsServiceError(f"provider={self.id} unavailable at {GPTSOVITS_URL}: {exc}") from exc
        if frames == 0:
            raise TtsServiceError(f"provider={self.id} returned an empty audio stream")
        print(f"[tts] stream complete provider={self.id} frames={frames}", flush=True)

    def synthesize(self, payload: dict[str, Any]) -> tuple[bytes, dict[str, str]]:
        text, voice, speed = self._prepare(payload)
        body = self._request_body(
            text,
            voice,
            speed,
            streaming_mode=0,
            media_type="wav",
        )
        url = _api_url("/tts")
        with GPTSOVITS_REQUEST_LOCK:
            try:
                with self._post_stream(url, body, GPTSOVITS_TIMEOUT_SECONDS) as response:
                    audio = response.read()
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                raise TtsServiceError(f"provider={self.id} HTTP {exc.code}: {detail}") from exc
            except (OSError, urllib.error.URLError, TimeoutError) as exc:
                raise TtsServiceError(f"provider={self.id} unavailable at {GPTSOVITS_URL}: {exc}") from exc
        if len(audio) < 44 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
            raise TtsServiceError(f"provider={self.id} did not return a valid WAV")
        return audio, {"Content-Type": "audio/wav", "X-TTS-Engine": GPTSOVITS_ENGINE}
