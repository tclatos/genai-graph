"""Registry for available knowledge graph subgraphs.

This module centralizes registration and lookup of EKG subgraphs so that
commands and core logic do not need hard dependencies on a particular
subgraph implementation module.
"""

from __future__ import annotations

from typing import Any

from genai_tk.utils.singleton import once
from pydantic import BaseModel, Field


class GraphRegistry(BaseModel):
    """Singleton registry for knowledge graph subgraphs."""

    subgraphs: dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "arbitrary_types_allowed": True,
    }

    @once
    def get_instance() -> GraphRegistry:
        """Get the global GraphRegistry instance."""
        return GraphRegistry()

    def register_subgraph(self, name: str, subgraph: Any) -> None:
        """Register a subgraph implementation under the given name."""
        self.subgraphs[name] = subgraph

    def get_subgraph(self, name: str) -> Any:
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


def register_subgraph(name: str, subgraph: Any) -> None:
    """Convenience wrapper to register a subgraph on the global registry."""
    GraphRegistry.get_instance().register_subgraph(name, subgraph)


def get_subgraph(name: str) -> Any:
    """Convenience wrapper to retrieve a subgraph from the global registry."""
    return GraphRegistry.get_instance().get_subgraph(name)
