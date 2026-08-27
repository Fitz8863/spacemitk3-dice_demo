"""Component registry: game-agnostic model/runtime capabilities.

A component is a pure capability (vision recognition, speech synthesis, LLM
judging, robot commanding). Games orchestrate components; components never
know which game they serve, so they stay reusable across games.
"""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
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
    """Scan components/ directory and auto-register all Component subclasses.

    Each *.py file in components/ is imported, and any Component subclass found
    is instantiated and registered. This eliminates the need to manually import
    and register new components in this function.
    """
    registry = ComponentRegistry()
    components_root = Path(__file__).resolve().parents[1] / "components"

    if not components_root.is_dir():
        print("[components] components/ directory not found", flush=True)
        return registry

    for py_file in sorted(components_root.glob("*.py")):
        # Skip __init__.py and private modules
        if py_file.stem.startswith("_"):
            continue

        try:
            # Dynamically import the module
            spec = importlib.util.spec_from_file_location(
                f"components.{py_file.stem}", py_file
            )
            if spec is None or spec.loader is None:
                print(f"[components] skip {py_file.name}: cannot create module spec", flush=True)
                continue

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find all Component subclasses and instantiate them
            for name in dir(module):
                obj = getattr(module, name)
                # Check if it's a class, subclass of Component, not Component itself,
                # and not an abstract base (has non-empty id)
                if (inspect.isclass(obj) and
                    issubclass(obj, Component) and
                    obj is not Component):
                    try:
                        instance = obj()
                        if not instance.id:
                            # Skip components with empty id (abstract bases)
                            continue
                        registry.register(instance)
                        print(f"[components] registered {instance.id} ({instance.type}) from {py_file.name}", flush=True)
                    except Exception as exc:
                        print(f"[components] skip {name} from {py_file.name}: {exc}", flush=True)

        except Exception as exc:
            print(f"[components] skip {py_file.name}: {exc}", flush=True)
            continue

    return registry
