"""Stateless multimodal verifier for OpenAI-compatible chat endpoints.

The verifier deliberately knows nothing about a game's rules.  A caller gives
it one image, one system prompt and one user prompt; it returns a structured
candidate that the profile/rules layer can validate and combine with YOLO.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import mimetypes
from pathlib import Path
import socket
from typing import Any, Callable, Mapping, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of one bounded LLM request."""

    status: str
    outcome: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class DiagnosisResult:
    """Outcome of one bounded LLM failure-diagnosis request."""

    status: str
    reason_code: str | None = None
    message: str | None = None
    retry: bool = True
    error: str | None = None


Post = Callable[[str, Mapping[str, Any], Mapping[str, str], float], Any]


def _default_post(url: str, payload: Mapping[str, Any], headers: Mapping[str, str], timeout: float) -> Any:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urlrequest.Request(url, data=body, headers=dict(headers), method="POST")
    with urlrequest.urlopen(request, timeout=timeout) as response:  # nosec B310 - endpoint is trusted config
        raw = response.read()
    return json.loads(raw.decode("utf-8"))


class OpenAICompatibleVisionVerifier:
    """Make one stateless OpenAI-compatible multimodal chat request."""

    def __init__(
        self,
        endpoint: str = "",
        *,
        model: str | None = None,
        api_key: str | None = None,
        post: Post | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self._post = post or _default_post

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
    ) -> VerificationResult:
        try:
            paths = [Path(p) for p in (image_paths or ([image_path] if image_path is not None else []))]
            if not paths:
                return VerificationResult("failure", error="image_path is required")
            image_parts = []
            for path in paths:
                data = path.read_bytes()
                mime = mimetypes.guess_type(path.name)[0]
                if mime not in {"image/jpeg", "image/png"}:
                    return VerificationResult("failure", error="unsupported image format")
                encoded = base64.b64encode(data).decode("ascii")
                image_parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
            payload = {
                "model": model or self.model or "",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": user_prompt}, *image_parts],
                    },
                ],
            }
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            response = self._post(self.endpoint, payload, headers, timeout_seconds)
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

    def diagnose(
        self,
        *,
        image_path: str | Path | None = None,
        image_paths: Sequence[str | Path] | None = None,
        system_prompt: str,
        user_prompt: str,
        allowed_reason_codes: Sequence[str],
        timeout_seconds: float,
        model: str | None = None,
    ) -> DiagnosisResult:
        try:
            paths = [Path(p) for p in (image_paths or ([image_path] if image_path is not None else []))]
            if not paths:
                return DiagnosisResult("failure", error="image_path is required")
            image_parts = []
            for path in paths:
                data = path.read_bytes()
                mime = mimetypes.guess_type(path.name)[0]
                if mime not in {"image/jpeg", "image/png"}:
                    return DiagnosisResult("failure", error="unsupported image format")
                encoded = base64.b64encode(data).decode("ascii")
                image_parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
            payload = {
                "model": model or self.model or "",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [{"type": "text", "text": user_prompt}, *image_parts]},
                ],
            }
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            response = self._post(self.endpoint, payload, headers, timeout_seconds)
            parsed = json.loads(self._extract_content(response))
            if not isinstance(parsed, Mapping):
                return DiagnosisResult("failure", error="LLM diagnosis must be a JSON object")
            reason = parsed.get("reason_code")
            message = parsed.get("message")
            retry = parsed.get("retry", True)
            allowed = set(allowed_reason_codes)
            if not isinstance(reason, str) or reason not in allowed:
                return DiagnosisResult("failure", error="LLM diagnosis returned unknown reason_code")
            if not isinstance(message, str) or not message.strip():
                return DiagnosisResult("failure", error="LLM diagnosis has no message")
            if not isinstance(retry, bool):
                return DiagnosisResult("failure", error="LLM diagnosis retry must be boolean")
            return DiagnosisResult("success", reason_code=reason, message=message.strip(), retry=retry)
        except (TimeoutError, socket.timeout) as exc:
            return DiagnosisResult("timeout", error=str(exc) or "LLM diagnosis request timed out")
        except urlerror.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)) or "timed out" in str(exc.reason).lower():
                return DiagnosisResult("timeout", error=str(exc.reason) or "LLM diagnosis request timed out")
            return DiagnosisResult("failure", error=str(exc))
        except (OSError, ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError) as exc:
            return DiagnosisResult("failure", error=str(exc) or "LLM diagnosis failed")

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
