"""Factory helpers for instantiating access control providers."""

from __future__ import annotations

import importlib
from typing import Any

from loguru import logger

from genai_graph.kg.access.base import BaseAccessControlProvider, DefaultPublicAccessControlProvider
from genai_graph.kg.access.yaml_provider import YamlAccessControlProvider


def create_access_control_provider(
    provider_spec: str | None = None,
    options: dict[str, Any] | None = None,
) -> BaseAccessControlProvider:
    """Instantiate an AccessControlProvider from a qualified name or alias.

    Args:
        provider_spec: Qualified Python class name (e.g. 'genai_graph.kg.access.yaml_provider.YamlAccessControlProvider')
            or shorthand alias ('yaml', 'default', 'public').
        options: Dictionary of provider keyword arguments.

    Returns:
        Configured BaseAccessControlProvider instance.
    """
    opts = options or {}
    if not provider_spec or provider_spec in ("default", "public"):
        return DefaultPublicAccessControlProvider(**opts)

    if provider_spec == "yaml":
        return YamlAccessControlProvider(**opts)

    # Resolve qualified Python class path
    try:
        module_path, class_name = provider_spec.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        if not issubclass(cls, BaseAccessControlProvider):
            raise TypeError(f"Class '{provider_spec}' is not a subclass of BaseAccessControlProvider")
        return cls(**opts)
    except Exception as exc:
        logger.error(
            "Failed instantiating AccessControlProvider '{}': {}. Falling back to default public provider.",
            provider_spec,
            exc,
        )
        return DefaultPublicAccessControlProvider(**opts)
