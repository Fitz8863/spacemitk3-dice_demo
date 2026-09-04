"""Unit tests for the llm_openai_compat provider (the moved LLM transport).

The verify/diagnose parsing, timeout and payload-shape cases mirror the
direct-verifier tests that used to live in test_vision_adjudicator.py; the
config-loading, endpoint-normalization and probe cases are new to this
component.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from urllib import error as urlerror

import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from components.llm_openai_compat.provider import LlmOpenAiCompat  # noqa: E402
from core.components import Component, ComponentRegistry  # noqa: E402
from core.llm import DiagnosisResult, LlmProvider, VerificationResult  # noqa: E402

TEST_CONFIG = {
    "endpoint": "https://llm.test/v1",
    "model": "test-model",
    "api_key": "test-key",
}


def _provider(**kwargs):
    config = kwargs.pop("config", TEST_CONFIG)
    return LlmOpenAiCompat(config=config, **kwargs)


def test_verify_roundtrip_payload_and_headers(tmp_path):
    image = tmp_path / "stable.jpg"
    image.write_bytes(b"jpeg-bytes")
    captured = {}

    def fake_post(url, payload, headers, timeout):
        captured.update(url=url, payload=payload, headers=headers, timeout=timeout)
        return {"choices": [{"message": {"content": '{"winner":"LEFT"}'}}]}

    result = _provider(post=fake_post).verify(
        image_path=image,
        system_prompt="Judge the image.",
        user_prompt="Return JSON.",
        allowed_outcomes=["LEFT", "RIGHT", "TIE"],
        timeout_seconds=3,
    )
    assert result.status == "success"
    assert result.outcome == "LEFT"
    # Base endpoint gains the chat-completions suffix; key rides as Bearer.
    assert captured["url"] == "https://llm.test/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["timeout"] == 3
    payload = captured["payload"]
    assert payload["model"] == "test-model"
    assert payload["messages"][0] == {"role": "system", "content": "Judge the image."}
    user_content = payload["messages"][1]["content"]
    assert user_content[0] == {"type": "text", "text": "Return JSON."}
    assert user_content[1]["type"] == "image_url"
    assert user_content[1]["image_url"]["url"] == "data:image/jpeg;base64,anBlZy1ieXRlcw=="


def test_verify_accepts_full_chat_completions_endpoint(tmp_path):
    image = tmp_path / "stable.png"
    image.write_bytes(b"png-bytes")
    seen = {}

    def fake_post(url, payload, headers, timeout):
        seen["url"] = url
        return {"choices": [{"message": {"content": '{"winner":"RIGHT"}'}}]}

    result = LlmOpenAiCompat(
        config={"endpoint": "https://llm.test/v1/chat/completions/", "model": "m", "api_key": ""},
        post=fake_post,
    ).verify(
        image_path=image,
        system_prompt="Judge.",
        user_prompt="Return JSON.",
        allowed_outcomes=["RIGHT"],
        timeout_seconds=1,
    )
    assert result.status == "success"
    # Already-normalized endpoints are used verbatim (no double suffix).
    assert seen["url"] == "https://llm.test/v1/chat/completions/"


def test_verify_call_model_overrides_config_default(tmp_path):
    image = tmp_path / "stable.jpg"
    image.write_bytes(b"jpeg")
    seen = {}

    def fake_post(url, payload, headers, timeout):
        seen["model"] = payload["model"]
        return {"choices": [{"message": {"content": '{"winner":"TIE"}'}}]}

    _provider(post=fake_post).verify(
        image_path=image,
        system_prompt="Judge.",
        user_prompt="Return JSON.",
        allowed_outcomes=["TIE"],
        timeout_seconds=1,
        model="per-game-model",
    )
    assert seen["model"] == "per-game-model"


def test_verify_without_api_key_omits_authorization(tmp_path):
    image = tmp_path / "stable.jpg"
    image.write_bytes(b"jpeg")
    seen = {}

    def fake_post(url, payload, headers, timeout):
        seen["headers"] = headers
        return {"choices": [{"message": {"content": '{"winner":"LEFT"}'}}]}

    LlmOpenAiCompat(
        config={"endpoint": "https://llm.test/v1", "model": "m", "api_key": ""},
        post=fake_post,
    ).verify(
        image_path=image,
        system_prompt="Judge.",
        user_prompt="Return JSON.",
        allowed_outcomes=["LEFT"],
        timeout_seconds=1,
    )
    assert "Authorization" not in seen["headers"]


def test_verify_requires_an_image():
    result = _provider(post=lambda *a, **k: {}).verify(
        system_prompt="Judge.",
        user_prompt="Return JSON.",
        allowed_outcomes=["LEFT"],
        timeout_seconds=1,
    )
    assert result.status == "failure"
    assert result.error == "image_path is required"


@pytest.mark.parametrize("content", [
    "not json",
    '{"winner":"MYSTERY"}',
    '{"no_winner":true}',
    '["array"]',
])
def test_verify_invalid_or_unknown_response_is_failure(tmp_path, content):
    image = tmp_path / "stable.jpg"
    image.write_bytes(b"jpeg-bytes")

    def fake_post(url, payload, headers, timeout):
        return {"choices": [{"message": {"content": content}}]}

    result = _provider(post=fake_post).verify(
        image_path=image,
        system_prompt="Judge.",
        user_prompt="Return JSON.",
        allowed_outcomes=["LEFT", "RIGHT", "TIE"],
        timeout_seconds=1,
    )
    assert result.status == "failure"
    assert result.outcome is None


def test_verify_timeout_is_distinguished_from_failure(tmp_path):
    image = tmp_path / "stable.png"
    image.write_bytes(b"png-bytes")

    def timeout_post(url, payload, headers, timeout):
        raise TimeoutError("deadline exceeded")

    result = _provider(post=timeout_post).verify(
        image_path=image,
        system_prompt="Judge.",
        user_prompt="Return JSON.",
        allowed_outcomes=["LEFT"],
        timeout_seconds=0.1,
    )
    assert result.status == "timeout"
    assert result.outcome is None


def test_verify_urlerror_with_timeout_reason_is_timeout(tmp_path):
    image = tmp_path / "stable.png"
    image.write_bytes(b"png-bytes")

    def timeout_post(url, payload, headers, timeout):
        raise urlerror.URLError(TimeoutError("socket timed out"))

    result = _provider(post=timeout_post).verify(
        image_path=image,
        system_prompt="Judge.",
        user_prompt="Return JSON.",
        allowed_outcomes=["LEFT"],
        timeout_seconds=0.1,
    )
    assert result.status == "timeout"


def test_verify_urlerror_without_timeout_reason_is_failure(tmp_path):
    image = tmp_path / "stable.png"
    image.write_bytes(b"png-bytes")

    def refused_post(url, payload, headers, timeout):
        raise urlerror.URLError(OSError("connection refused"))

    result = _provider(post=refused_post).verify(
        image_path=image,
        system_prompt="Judge.",
        user_prompt="Return JSON.",
        allowed_outcomes=["LEFT"],
        timeout_seconds=0.1,
    )
    assert result.status == "failure"


def test_diagnose_parses_reason_code_and_message(tmp_path):
    image = tmp_path / "diagnostic.jpg"
    image.write_bytes(b"jpeg-bytes")

    def fake_post(url, payload, headers, timeout):
        assert "detector summary" in payload["messages"][1]["content"][0]["text"].lower()
        return {
            "choices": [{"message": {"content":
                '{"reason_code":"OVERLAPPING_OBJECTS","message":"骰子可能叠放。","retry":true}'
            }}]
        }

    result = _provider(post=fake_post).diagnose(
        image_path=image,
        system_prompt="Diagnose only.",
        user_prompt="Detector summary: LEFT=4; RIGHT=5",
        allowed_reason_codes=["OVERLAPPING_OBJECTS", "UNKNOWN"],
        timeout_seconds=1,
    )
    assert result.status == "success"
    assert result.reason_code == "OVERLAPPING_OBJECTS"
    assert result.message == "骰子可能叠放。"
    assert result.retry is True


@pytest.mark.parametrize("content", [
    '{"reason_code":"NOT_ALLOWED","message":"x","retry":true}',
    '{"reason_code":"OVERLAPPING_OBJECTS","retry":true}',
    '{"reason_code":"OVERLAPPING_OBJECTS","message":"x","retry":"yes"}',
    "not json",
])
def test_diagnose_invalid_responses_are_failures(tmp_path, content):
    image = tmp_path / "diagnostic.jpg"
    image.write_bytes(b"jpeg-bytes")

    def fake_post(url, payload, headers, timeout):
        return {"choices": [{"message": {"content": content}}]}

    result = _provider(post=fake_post).diagnose(
        image_path=image,
        system_prompt="Diagnose.",
        user_prompt="Summary",
        allowed_reason_codes=["OVERLAPPING_OBJECTS"],
        timeout_seconds=1,
    )
    assert result.status == "failure"
    assert result.reason_code is None


def test_extract_content_accepts_array_of_text_parts():
    response = {"choices": [{"message": {"content": [
        {"type": "text", "text": '{"winner":'},
        {"type": "text", "text": '"LEFT"}'},
    ]}}]}
    assert LlmOpenAiCompat._extract_content(response) == '{"winner":"LEFT"}'


def test_packaged_config_carries_the_migrated_deepseek_values():
    """The llm transport moved here from vision_yolov8_adjudicator config."""
    provider = LlmOpenAiCompat()
    assert provider._endpoint == "https://api.deepseek.com/v1"
    assert provider._model == "deepseek-v4-flash-vision-exp"
    assert provider._api_key.startswith("sk-") and len(provider._api_key) >= 32
    health = provider.health()
    assert health["configured"] is True
    assert health["ok"] is True
    assert "api_key" not in json.dumps(health)


def test_probe_reports_connectivity_without_endpoint():
    provider = LlmOpenAiCompat(config={"endpoint": "", "model": "m", "api_key": ""})
    assert provider.probe() == {"ok": False, "error": "endpoint is not configured"}


def test_probe_success_and_unreachable():
    seen = {}

    def fake_get(url, headers, timeout):
        seen.update(url=url, headers=headers, timeout=timeout)
        return {"data": []}

    provider = _provider(get=fake_get)
    assert provider.probe(timeout_seconds=2) == {"ok": True}
    assert seen["url"] == "https://llm.test/v1/models"
    assert seen["headers"]["Authorization"] == "Bearer test-key"

    def unreachable_get(url, headers, timeout):
        raise urlerror.URLError(OSError("connection refused"))

    result = _provider(get=unreachable_get).probe()
    assert result["ok"] is False
    assert "connection refused" in result["error"]


def test_registry_enforces_llm_contract():
    """A component claiming type=llm must implement LlmProvider."""

    class NotAnLlm(Component):
        id = "llm_broken"
        type = "llm"

    with pytest.raises(ValueError, match="does not implement LlmProvider"):
        ComponentRegistry().register(NotAnLlm())


def test_provider_satisfies_the_abstract_contract():
    assert isinstance(LlmOpenAiCompat(), LlmProvider)
    # Result types are part of the core contract.
    assert VerificationResult("success", outcome="LEFT").outcome == "LEFT"
    assert DiagnosisResult("success", reason_code="UNKNOWN").retry is True
