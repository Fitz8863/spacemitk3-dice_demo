"""Runtime registry for pluggable model/device providers.

A provider is a small adapter package with a ``manifest.json`` and an entry
point class.  Games and HTTP handlers resolve providers by id instead of
importing a concrete implementation.  Provider implementations may be
replaced or removed without changing the orchestration code.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Callable

from core.errors import ComponentNotFoundError, ComponentNotReadyError


class Component:
    """Base interface shared by all runtime providers."""

    id: str = ""
    type: str = ""  # vision | tts | llm | command
    role: str = ""  # role within a broad type, e.g. vision/adjudicator
    name: str = ""
    version: str = ""

    def __init__(self, manifest: dict[str, Any] | None = None) -> None:
        self.manifest = dict(manifest or {})

    def health(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "ok": True}


_SUPPORTED_TYPES = {"vision", "tts", "llm", "command"}
_VISION_ROLES = {"adjudicator", "localizer"}

# ``vision_yolo`` was the pre-profile provider id.  Keep this one-way alias
# for manifests and clients during migration; it is never registered as a
# second component.
COMPONENT_ID_ALIASES = {
    "vision_yolo": "vision_yolov8_adjudicator",
}


def _validate_component_contract(component: Component) -> None:
    """Enforce the type-specific interface at the provider seam."""
    if component.type == "tts":
        from core.tts import TtsProvider

        expected = TtsProvider
    elif component.type == "vision":
        from core.vision import VisionAdjudicatorProvider, VisionLocalizerProvider

        expected_by_role = {
            "adjudicator": VisionAdjudicatorProvider,
            "localizer": VisionLocalizerProvider,
        }
        expected = expected_by_role.get(component.role)
        if expected is None:
            raise ValueError(
                f"component {component.id or '<unnamed>'!r} declares vision "
                f"role {component.role or '<missing>'!r}; expected one of "
                f"{sorted(expected_by_role)}"
            )
    else:
        return
    if not isinstance(component, expected):
        raise ValueError(
            f"component {component.id or '<unnamed>'!r} declares type "
            f"{component.type!r} but does not implement {expected.__name__}"
        )


def _validate_manifest(manifest_path: Path, manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    component_id = manifest.get("id")
    if (
        not isinstance(component_id, str)
        or not component_id
        or not component_id.replace("_", "").replace("-", "").isalnum()
    ):
        raise ValueError("id must contain only letters, numbers, '_' or '-'")
    provider_type = manifest.get("type")
    if provider_type not in _SUPPORTED_TYPES:
        raise ValueError(f"type must be one of {sorted(_SUPPORTED_TYPES)}")
    if not isinstance(manifest.get("enabled", True), bool):
        raise ValueError("enabled must be boolean")
    role = manifest.get("role", "")
    if provider_type == "vision":
        if role not in _VISION_ROLES:
            raise ValueError(f"vision role must be one of {sorted(_VISION_ROLES)}")
    elif role not in {"", None}:
        raise ValueError("role is currently supported only for vision providers")
    capabilities = manifest.get("capabilities", [])
    if not isinstance(capabilities, list) or not all(
        isinstance(item, str) and item for item in capabilities
    ):
        raise ValueError("capabilities must be a string array")
    if len(capabilities) != len(set(capabilities)):
        raise ValueError("capabilities must not contain duplicates")
    entry = manifest.get("entry")
    if not isinstance(entry, str) or entry.count(":") != 1:
        raise ValueError("entry must be 'module.py:ClassName'")
    module_name, class_name = entry.split(":", 1)
    module_path = Path(module_name)
    if (
        module_path.is_absolute()
        or module_path.suffix != ".py"
        or any(part in {"", ".", ".."} for part in module_path.parts)
        or not class_name.isidentifier()
    ):
        raise ValueError("entry must name a Python module inside the provider package and a class")
    if manifest_path.parent.name != component_id:
        raise ValueError(f"directory name {manifest_path.parent.name!r} must match id {component_id!r}")
    config_file = manifest.get("config")
    if provider_type == "tts" and config_file is None:
        raise ValueError("TTS provider manifest must declare a component-local config")
    if config_file is not None:
        config_path = Path(config_file)
        if (
            not isinstance(config_file, str)
            or config_path.is_absolute()
            or any(part in {"", ".", ".."} for part in config_path.parts)
            or not (manifest_path.parent / config_path).is_file()
        ):
            raise ValueError("config must name an existing file inside the provider package")
    lifecycle = manifest.get("lifecycle", {})
    if not isinstance(lifecycle, dict):
        raise ValueError("lifecycle must be an object")
    for action, command in lifecycle.items():
        if action not in {"start", "stop"}:
            raise ValueError(f"unsupported lifecycle action: {action}")
        if not isinstance(command, list) or not command or not all(
            isinstance(item, str) and item for item in command
        ):
            raise ValueError(f"lifecycle.{action} must be a non-empty string array")
    return manifest


class ComponentRegistry:
    """Resolve providers by stable id and expose their health/metadata."""

    def __init__(self, *, migration_logger: Callable[[str], None] | None = None) -> None:
        self._components: dict[str, Component] = {}
        self._manifests: dict[str, dict[str, Any]] = {}
        self._migration_logger = migration_logger
        self._migration_logged: set[str] = set()

    def canonical_id(self, component_id: str) -> str:
        canonical = COMPONENT_ID_ALIASES.get(component_id, component_id)
        if canonical != component_id and component_id not in self._migration_logged:
            self._migration_logged.add(component_id)
            if self._migration_logger is not None:
                self._migration_logger(
                    f"[components] migration alias {component_id} -> {canonical}"
                )
        return canonical

    def register(self, component: Component, manifest: dict[str, Any] | None = None) -> None:
        if not component.id:
            raise ValueError("component id must not be empty")
        if component.type not in _SUPPORTED_TYPES:
            raise ValueError(f"unsupported component type: {component.type!r}")
        _validate_component_contract(component)
        if component.id in self._components:
            raise ValueError(f"duplicate component id: {component.id}")
        if manifest is not None:
            manifest_id = str(manifest.get("id") or "")
            manifest_type = str(manifest.get("type") or "")
            manifest_role = str(manifest.get("role") or "")
            if manifest_id and manifest_id != component.id:
                raise ValueError(
                    f"manifest id {manifest_id!r} does not match component id {component.id!r}"
                )
            if manifest_type and manifest_type != component.type:
                raise ValueError(
                    f"manifest type {manifest_type!r} does not match component type {component.type!r}"
                )
            if manifest_role != component.role:
                raise ValueError(
                    f"manifest role {manifest_role!r} does not match component role {component.role!r}"
                )
        self._components[component.id] = component
        if manifest is not None:
            self._manifests[component.id] = dict(manifest)

    def get(self, component_id: str) -> Component:
        component_id = self.canonical_id(component_id)
        component = self._components.get(component_id)
        if component is None:
            raise ComponentNotFoundError(component_id)
        return component

    def require(
        self,
        component_id: str,
        expected_type: str | None = None,
        expected_role: str | None = None,
    ) -> Component:
        component_id = self.canonical_id(component_id)
        component = self.get(component_id)
        if expected_type and component.type != expected_type:
            raise ComponentNotReadyError(
                component_id,
                f"type mismatch: expected {expected_type}, got {component.type or 'unknown'}",
            )
        if expected_role and component.role != expected_role:
            raise ComponentNotReadyError(
                component_id,
                f"role mismatch: expected {expected_role}, got {component.role or 'unknown'}",
            )
        return component

    def ids(self) -> list[str]:
        return sorted(self._components)

    def get_manifest(self, component_id: str) -> dict[str, Any]:
        component_id = self.canonical_id(component_id)
        self.get(component_id)
        return dict(self._manifests.get(component_id, {}))

    def provider_ids(self, provider_type: str, role: str | None = None) -> list[str]:
        return sorted(
            component_id for component_id, component in self._components.items()
            if component.type == provider_type and (role is None or component.role == role)
        )

    def all(self, include_health: bool = True) -> list[dict[str, Any]]:
        result = []
        for component_id in self.ids():
            component = self._components[component_id]
            manifest = self._manifests.get(component_id, {})
            item = {
                "id": component.id,
                "type": component.type,
                "role": component.role,
                "name": manifest.get("name") or component.name or component.id,
                "version": manifest.get("version") or component.version or "",
                "enabled": manifest.get("enabled", True),
                "capabilities": list(manifest.get("capabilities", [])),
                "config": manifest.get("config"),
            }
            if include_health:
                try:
                    item["health"] = component.health()
                except Exception as exc:  # health must not take down /api/health
                    item["health"] = {"id": component.id, "type": component.type, "ok": False, "error": str(exc)}
            result.append(item)
        return result


def _load_entrypoint(manifest_path: Path, manifest: dict[str, Any]) -> type[Component]:
    entry = manifest.get("entry")
    if not isinstance(entry, str) or ":" not in entry:
        raise ValueError("manifest entry must be 'module.py:ClassName'")
    module_name, class_name = entry.split(":", 1)
    provider_root = manifest_path.parent.resolve()
    provider_path = (provider_root / module_name).resolve()
    try:
        provider_path.relative_to(provider_root)
    except ValueError as exc:
        raise ValueError("entry module escapes the provider package") from exc
    if not provider_path.is_file():
        raise ValueError(f"entry module not found: {provider_path.name}")

    safe_id = str(manifest.get("id", manifest_path.parent.name)).replace("-", "_")
    import_name = f"dice_arena_component_{safe_id}_{manifest_path.parent.name}"
    spec = importlib.util.spec_from_file_location(import_name, provider_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot create module spec for {provider_path}")
    module = importlib.util.module_from_spec(spec)
    # Keep the module alive for imported dataclasses/decorators and so a
    # provider's module-level locks/caches remain process-wide.
    sys.modules[import_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(import_name, None)
        raise
    provider = getattr(module, class_name, None)
    if not inspect.isclass(provider) or not issubclass(provider, Component):
        raise ValueError(f"entry {entry} is not a Component subclass")
    return provider


def build_registry(*, log: bool = True) -> ComponentRegistry:
    """Scan ``backend/components/*/manifest.json`` and instantiate providers."""
    def report(message: str) -> None:
        if log:
            print(message, flush=True)
    registry = ComponentRegistry(migration_logger=report)
    components_root = Path(__file__).resolve().parents[1] / "components"
    if not components_root.is_dir():
        report("[components] components/ directory not found")
        return registry

    for manifest_path in sorted(components_root.glob("*/manifest.json")):
        try:
            manifest = _validate_manifest(
                manifest_path,
                json.loads(manifest_path.read_text(encoding="utf-8")),
            )
            component_id = manifest["id"]
            if component_id in COMPONENT_ID_ALIASES:
                report(
                    f"[components] skip legacy package {component_id}; "
                    f"use {COMPONENT_ID_ALIASES[component_id]}"
                )
                continue
            if manifest.get("enabled", True) is False:
                report(f"[components] disabled {component_id}")
                continue
            provider_class = _load_entrypoint(manifest_path, manifest)
            parameters = inspect.signature(provider_class).parameters
            accepts_manifest = "manifest" in parameters or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
            instance = provider_class(manifest=manifest) if accepts_manifest else provider_class()
            if instance.id != component_id:
                raise ValueError(f"entry id {instance.id!r} does not match manifest id {component_id!r}")
            if instance.type != manifest["type"]:
                raise ValueError(
                    f"entry type {instance.type!r} does not match manifest type {manifest['type']!r}"
                )
            manifest_role = str(manifest.get("role") or "")
            if instance.role != manifest_role:
                raise ValueError(
                    f"entry role {instance.role!r} does not match manifest role {manifest_role!r}"
                )
            registry.register(instance, manifest)
            type_label = (
                f"{instance.type}/{instance.role}" if instance.role else instance.type
            )
            report(
                f"[components] registered {instance.id} ({type_label}) "
                f"from {manifest_path.parent.name}"
            )
        except Exception as exc:
            report(f"[components] skip {manifest_path}: {exc}")
    return registry
