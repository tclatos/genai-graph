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

from genai_graph.core.graph_schema import GraphSchema
from genai_graph.core.subgraph import Subgraph


class GraphRegistry(BaseModel):
    """Singleton registry for knowledge graph subgraphs.

    In addition to managing individual subgraphs, the registry can also
    build *combined* graph schemas that merge multiple subgraphs into a
    single :class:`GraphSchema`. This is useful for commands that should
    operate on a logical union of several subgraphs.
    """

    subgraphs: dict[str, Subgraph] = Field(default_factory=dict)

    model_config = {
        "arbitrary_types_allowed": True,
    }

    def model_post_init(self, _context: Any) -> None:
        """Load and register configured subgraph providers.

        Each entry in the ``subgraphs`` configuration should resolve to one of
        the following:

        * A :class:`Subgraph` subclass (preferred) – it will be instantiated
          with default arguments and registered.
        * A callable returning a :class:`Subgraph` instance – the instance
          will be registered.
        * A legacy ``register`` function that accepts an optional
          :class:`GraphRegistry` instance; this path is kept for backward
          compatibility and is responsible for calling
          :func:`register_subgraph` manually.
        """
        providers = global_config().get_list("subgraphs", value_type=str)
        for provider in providers:
            try:
                logger.info(f"import {provider}")
                imported = import_from_qualified(provider)

                subgraph: Subgraph | None = None

                # Already-instantiated Subgraph instance
                if isinstance(imported, Subgraph):
                    subgraph = imported
                # Subgraph subclass – instantiate with defaults
                elif isinstance(imported, type) and issubclass(imported, Subgraph):
                    subgraph = imported()
                else:
                    # Callable provider – may be a factory returning a Subgraph
                    # or a legacy register(registry) function.
                    try:
                        candidate = imported(self)
                    except TypeError:
                        candidate = imported()

                    if isinstance(candidate, Subgraph):
                        subgraph = candidate
                    else:
                        # Legacy path: callable handled registration itself.
                        continue

                if subgraph is not None:
                    subgraph.register(self)

            except Exception as ex:
                logger.warning(f"Cannot load subgraph provider {provider}: {ex}")

    @once
    def get_instance() -> "GraphRegistry":
        """Get the global GraphRegistry instance."""
        return GraphRegistry()

    def register_subgraph(self, name: str, subgraph: Subgraph) -> None:
        """Register a subgraph implementation under the given name."""
        self.subgraphs[name] = subgraph

    def build_combined_schema(self, subgraph_names: list[str] | None = None) -> GraphSchema:
        """Build a combined :class:`GraphSchema` from one or more subgraphs.

        Args:
            subgraph_names: Optional list of subgraph names to combine. If
                omitted or empty, all registered subgraphs are used.

        Returns:
            A new :class:`GraphSchema` instance whose nodes and relations are
            the union of the selected subgraphs.

        Notes:
            - Node configurations are de-duplicated by their ``baml_class``.
            - Relationship configurations are de-duplicated by the
              ``(from_node, to_node, name)`` triple.
            - The ``root_model_class`` of the first selected subgraph is used
              for the combined schema; this is sufficient for documentation
              and visualization use cases where we only need the union of
              nodes/relations.
        """
        if not self.subgraphs:
            raise ValueError("No subgraphs are registered in the GraphRegistry")

        # Default to all registered subgraphs when none are explicitly provided
        if not subgraph_names:
            subgraph_names = sorted(self.subgraphs.keys())

        schemas: list[GraphSchema] = []
        for name in subgraph_names:
            if name not in self.subgraphs:
                available = ", ".join(sorted(self.subgraphs.keys())) or "<none>"
                raise ValueError(f"Unknown subgraph '{name}'. Available: {available}")
            schemas.append(self.subgraphs[name].build_schema())

        if not schemas:
            raise ValueError("No schemas could be built from the selected subgraphs")

        # Use the root_model_class of the first schema; other schemas may use
        # different roots but their node/relationship configurations are still
        # meaningful when merged.
        root_model_class = schemas[0].root_model_class

        # Merge nodes, de-duplicating by underlying Pydantic class
        merged_nodes: list[Any] = []
        seen_node_classes: set[type] = set()
        for schema in schemas:
            for node in schema.nodes:
                node_class = node.baml_class
                if node_class in seen_node_classes:
                    continue
                seen_node_classes.add(node_class)
                merged_nodes.append(node)

        # Merge relations, de-duplicating by (from_node, to_node, name)
        merged_relations: list[Any] = []
        seen_relations: set[tuple[type, type, str]] = set()
        for schema in schemas:
            for rel in schema.relations:
                key = (rel.from_node, rel.to_node, rel.name)
                if key in seen_relations:
                    continue
                seen_relations.add(key)
                merged_relations.append(rel)

        return GraphSchema(root_model_class=root_model_class, nodes=merged_nodes, relations=merged_relations)

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


def register_subgraph(name: str, subgraph: Subgraph, registry: "GraphRegistry | None" = None) -> None:
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
