"""Component registry: game-agnostic model/runtime capabilities.

A component is a pure capability (vision recognition, speech synthesis, LLM
judging, robot commanding). Games orchestrate components; components never
know which game they serve, so they stay reusable across games.
"""
from __future__ import annotations

from typing import Any

from core.errors import ComponentNotFoundError


class Component:
    """Base class for a model/runtime component."""

    id: str = ""
    type: str = ""  # vision | tts | llm | command

    def health(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "ok": True}


class ComponentRegistry:
    """Holds the components a game may depend on (``manifest.components``)."""

    def __init__(self) -> None:
        self._components: dict[str, Component] = {}

    def register(self, component: Component) -> None:
        self._components[component.id] = component

    def get(self, component_id: str) -> Component:
        """Get a component by ID, raises ComponentNotFoundError if not found."""
        component = self._components.get(component_id)
        if component is None:
            raise ComponentNotFoundError(component_id)
        return component

    def ids(self) -> list[str]:
        return sorted(self._components)


def build_registry() -> ComponentRegistry:
    """Construct the registry with the board-local built-in components."""
    # Imported lazily to avoid a circular import during package setup.
    from components.tts_qwen3 import TtsQwen3
    from components.vision_yolo import VisionYolo

    registry = ComponentRegistry()
    registry.register(TtsQwen3())
    registry.register(VisionYolo())
    return registry
