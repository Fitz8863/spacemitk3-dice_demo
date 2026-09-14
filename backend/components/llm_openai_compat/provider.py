"""OpenAI-compatible chat LLM adapter (cloud APIs / vLLM / llama.cpp servers).

Pure HTTP client with no local lifecycle: the endpoint is deployment
config.  Implements the arena ``LlmProvider`` contract — one bounded,
structured multimodal request (``verify``) — and deliberately knows nothing
about any game's rules; prompts, allowed outcomes and timeouts arrive with
each call.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import socket
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest

from core.llm import LlmProvider, VerificationResult
from core.tts_config import config_value, load_component_config

COMPONENT_DIR = Path(__file__).resolve().parent

Post = Callable[[str, Mapping[str, Any], Mapping[str, str], float], Any]
Get = Callable[[str, Mapping[str, str], float], Any]


def _default_post(url: str, payload: Mapping[str, Any], headers: Mapping[str, str], timeout: float) -> Any:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urlrequest.Request(url, data=body, headers=dict(headers), method="POST")
    with urlrequest.urlopen(request, timeout=timeout) as response:  # nosec B310 - endpoint is trusted config
        raw = response.read()
    return json.loads(raw.decode("utf-8"))


def _default_get(url: str, headers: Mapping[str, str], timeout: float) -> Any:
    request = urlrequest.Request(url, headers=dict(headers), method="GET")
    with urlrequest.urlopen(request, timeout=timeout) as response:  # nosec B310 - endpoint is trusted config
        raw = response.read()
    return json.loads(raw.decode("utf-8"))


def _chat_completions_url(endpoint: str) -> str:
    """Accept either a base URL (``…/v1``) or a full chat-completions URL."""
    if endpoint and not endpoint.rstrip("/").endswith("chat/completions"):
        return endpoint.rstrip("/") + "/chat/completions"
    return endpoint


REASONING_EFFORT_VALUES = ("none", "low", "high", "max")


def _thinking_payload(effort: str | None) -> dict[str, str] | None:
    """Translate the effort knob into the endpoint's ``thinking`` object.

    The API requires ``type`` inside ``thinking`` — sending only
    ``reasoning_effort`` is rejected with ``missing field ``type``` — so
    ``none`` becomes ``{"type": "disabled"}`` (thinking off, no
    ``reasoning_content``) and the graded levels ride along with
    ``{"type": "enabled"}``.  Unknown values are not sent at all.
    """
    if not isinstance(effort, str):
        return None
    value = effort.strip().lower()
    if value == "none":
        return {"type": "disabled"}
    if value in REASONING_EFFORT_VALUES:
        return {"type": "enabled", "reasoning_effort": value}
    return None


class LlmOpenAiCompat(LlmProvider):
    """Stateless OpenAI-compatible multimodal chat client."""

    id = "llm_openai_compat"
    type = "llm"
    name = "OpenAI-Compatible Chat LLM"
    version = "1.0"

    def __init__(
        self,
        manifest: dict[str, Any] | None = None,
        *,
        config: dict[str, Any] | None = None,
        post: Post | None = None,
        get: Get | None = None,
    ) -> None:
        super().__init__(manifest)
        cfg = config if config is not None else load_component_config(COMPONENT_DIR)
        self._endpoint = str(config_value(cfg, "endpoint", default="") or "").strip()
        self._model = str(config_value(cfg, "model", default="") or "").strip()
        self._api_key = str(config_value(cfg, "api_key", default="") or "").strip()
        # Optional deployment default for the endpoint's thinking depth; a game
        # profile can override it per call.  An unrecognised value is dropped
        # (and surfaced in health) instead of failing startup.
        raw_effort = str(config_value(cfg, "reasoning_effort", default="") or "").strip().lower()
        self._reasoning_effort = raw_effort if raw_effort in REASONING_EFFORT_VALUES else ""
        self._reasoning_effort_ignored = raw_effort if raw_effort != self._reasoning_effort else ""
        self._chat_url = _chat_completions_url(self._endpoint)
        self._post = post or _default_post
        self._get = get or _default_get

    def health(self) -> dict[str, Any]:
        """Report configuration readiness without exposing credentials."""
        payload: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "ok": True,
            "configured": bool(self._endpoint and self._api_key),
            "model": self._model,
            "reasoning_effort": self._reasoning_effort or None,
        }
        if self._reasoning_effort_ignored:
            payload["reasoning_effort_error"] = (
                f"unrecognised reasoning_effort {self._reasoning_effort_ignored!r}; ignored"
            )
        return payload

    def probe(self, timeout_seconds: float = 5.0) -> dict[str, Any]:
        """One cheap connectivity/credential check: ``GET {endpoint}/models``.

        The result is advisory only — the arena treats a failed probe as
        "LLM verification disabled, YOLO fallback" and never refuses
        startup on it.
        """
        if not self._endpoint:
            return {"ok": False, "error": "endpoint is not configured"}
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            self._get(self._endpoint.rstrip("/") + "/models", headers, timeout_seconds)
        except (TimeoutError, socket.timeout) as exc:
            return {"ok": False, "error": f"probe timed out: {exc}"}
        except (urlerror.URLError, OSError, ValueError) as exc:
            return {"ok": False, "error": str(exc) or "probe failed"}
        return {"ok": True}

    def verify(
        self,
        *,
        image_path: str | Path | None = None,
        image_paths: Sequence[str | Path] | None = None,
        system_prompt: str,
        user_prompt: str,
        allowed_outcomes: Sequence[str],
        timeout_seconds: float,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> VerificationResult:
        try:
            image_parts, image_error = self._image_parts(image_path, image_paths)
            if image_error:
                return VerificationResult("failure", error=image_error)
            payload = {
                "model": model or self._model or "",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": user_prompt}, *image_parts],
                    },
                ],
            }
            # A per-call effort (game profile) wins over the deployment default;
            # both end up as the endpoint's required ``type``-bearing object.
            thinking = _thinking_payload(reasoning_effort or self._reasoning_effort)
            if thinking is not None:
                payload["thinking"] = thinking
            response = self._post(self._chat_url, payload, self._headers(), timeout_seconds)
            content = self._extract_content(response)
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                return VerificationResult("failure", error="LLM response must be a JSON object")
            outcome = parsed.get("winner", parsed.get("outcome"))
            if not isinstance(outcome, str) or not outcome.strip():
                return VerificationResult("failure", error="LLM response has no winner/outcome")
            outcome = outcome.strip()
            if outcome not in set(allowed_outcomes):
                return VerificationResult("failure", error="LLM returned unknown outcome")
            return VerificationResult("success", outcome=outcome)
        except (TimeoutError, socket.timeout) as exc:
            return VerificationResult("timeout", error=str(exc) or "LLM request timed out")
        except urlerror.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)) or "timed out" in str(exc.reason).lower():
                return VerificationResult("timeout", error=str(exc.reason) or "LLM request timed out")
            return VerificationResult("failure", error=str(exc))
        except (OSError, ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError) as exc:
            return VerificationResult("failure", error=str(exc) or "LLM request failed")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    @staticmethod
    def _image_parts(
        image_path: str | Path | None,
        image_paths: Sequence[str | Path] | None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Encode the supplied images as OpenAI-style data-URL parts."""
        paths = [Path(p) for p in (image_paths or ([image_path] if image_path is not None else []))]
        if not paths:
            return [], "image_path is required"
        parts: list[dict[str, Any]] = []
        for path in paths:
            data = path.read_bytes()
            mime = mimetypes.guess_type(path.name)[0]
            if mime not in {"image/jpeg", "image/png"}:
                return [], "unsupported image format"
            encoded = base64.b64encode(data).decode("ascii")
            parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
        return parts, None

    @staticmethod
    def _extract_content(response: Any) -> str:
        if isinstance(response, (bytes, bytearray)):
            response = json.loads(bytes(response).decode("utf-8"))
        if isinstance(response, str):
            return response
        if not isinstance(response, Mapping):
            raise ValueError("LLM response must be an object")
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("LLM response has no choices")
        message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if isinstance(content, str):
            return content
        # Some compatible servers return an array of text parts.
        if isinstance(content, list):
            texts = [part.get("text", "") for part in content if isinstance(part, Mapping)]
            if texts and all(isinstance(text, str) for text in texts):
                return "".join(texts)
        raise ValueError("LLM response message content is invalid")
