"""Registry for available knowledge graph subgraphs.

This module centralizes registration and lookup of EKG subgraphs so that
commands and core logic do not need hard dependencies on a particular
subgraph implementation module.
"""

from __future__ import annotations

from typing import Any

from genai_tk.utils.config_mngr import global_config, import_from_qualified
from genai_tk.utils.singleton import once
from loguru import logger
from pydantic import BaseModel, Field

from genai_graph.core.subgraph import Subgraph


class GraphRegistry(BaseModel):
    """Singleton registry for knowledge graph subgraphs."""

    subgraphs: dict[str, Subgraph] = Field(default_factory=dict)

    model_config = {
        "arbitrary_types_allowed": True,
    }

    def model_post_init(self, _context: Any) -> None:
        """Load and register configured subgraph providers.

        Subgraph registration functions can optionally accept a
        :class:`GraphRegistry` instance as their single argument. If they
        don't, they will be called with no arguments for backward
        compatibility.
        """
        modules = global_config().get_list("subgraphs", value_type=str)
        for module in modules:
            try:
                logger.info(f"import {module}")
                imported = import_from_qualified(module)
                try:
                    # Preferred signature: register(registry: GraphRegistry) -> None
                    imported(self)
                except TypeError:
                    # Backward-compatible signature: register() -> None
                    imported()
            except Exception as ex:
                logger.warning(f"Cannot load module {module}: {ex}")

    @once
    def get_instance() -> "GraphRegistry":
        """Get the global GraphRegistry instance."""
        return GraphRegistry()

    def register_subgraph(self, name: str, subgraph: Subgraph) -> None:
        """Register a subgraph implementation under the given name."""
        self.subgraphs[name] = subgraph

    def get_subgraph(self, name: str) -> Subgraph:
        """Get a subgraph instance by name.

        Args:
            name: Name of the subgraph to retrieve.

        Returns:
            Subgraph instance.

        Raises:
            ValueError: If subgraph name is not found.
        """
        if name not in self.subgraphs:
            available = ", ".join(sorted(self.subgraphs.keys())) or "<none>"
            raise ValueError(f"Unknown subgraph '{name}'. Available: {available}")
        return self.subgraphs[name]

    def listsubgraphs(self) -> list[str]:
        """List names of all registered subgraphs."""
        return sorted(self.subgraphs.keys())


def register_subgraph(name: str, subgraph: Subgraph, registry: "GraphRegistry" | None = None) -> None:
    """Convenience wrapper to register a subgraph on the global registry.

    The optional ``registry`` argument allows explicit control over
    which registry instance receives the registration and avoids
    recursive calls to :meth:`GraphRegistry.get_instance` during
    initialisation.
    """
    target = registry if registry is not None else GraphRegistry.get_instance()
    target.register_subgraph(name, subgraph)


def get_subgraph(name: str) -> Subgraph:
    """Convenience wrapper to retrieve a subgraph from the global registry."""
    return GraphRegistry.get_instance().get_subgraph(name)


_ = GraphRegistry.get_instance()
