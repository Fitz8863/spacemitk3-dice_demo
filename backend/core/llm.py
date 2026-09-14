"""Contract for pluggable large-language-model providers.

An LLM provider is a transport adapter: it knows how to reach a model
endpoint (cloud API or local inference server) and how to turn one bounded,
structured request into a validated result.  It deliberately knows nothing
about any game's rules — prompts, allowed outcomes and timeouts arrive with
each call from the game's vision profile.

The only request in the contract is the pre-winner verification of a stable
frame.  Failure diagnosis is produced locally from detector evidence by the
vision provider; the former LLM diagnosis request was removed on 2026-09-14.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from core.components import Component


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of one bounded LLM request."""

    status: str
    outcome: str | None = None
    error: str | None = None


class LlmProvider(Component, ABC):
    """Small stable interface for structured multimodal LLM adapters."""

    type = "llm"

    @abstractmethod
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
        """Ask for one outcome constrained to ``allowed_outcomes``."""
        raise NotImplementedError
