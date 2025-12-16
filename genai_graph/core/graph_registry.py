"""Registry for available knowledge graph subgraphs.

This module centralizes registration and lookup of EKG subgraphs so that
commands and core logic do not need hard dependencies on a particular
subgraph implementation module.
"""

from __future__ import annotations

import typing
from typing import Any

from genai_tk.utils.config_mngr import global_config, import_from_qualified
from genai_tk.utils.singleton import once
from loguru import logger
from pydantic import BaseModel, Field

from genai_graph.core.graph_schema import GraphSchema
from genai_graph.core.subgraph_factories import SubgraphFactory

if typing.TYPE_CHECKING:
    from genai_graph.core.graph_registry import GraphRegistry

from beartype import BeartypeConf, beartype

beartype_nop = beartype(conf=BeartypeConf(claw_decoration_position_funcs=None))


class GraphRegistry(BaseModel):
    """Singleton registry for knowledge graph subgraphs.

    In addition to managing individual subgraphs, the registry can also
    build *combined* graph schemas that merge multiple subgraphs into a
    single :class:`GraphSchema`. This is useful for commands that should
    operate on a logical union of several subgraphs.
    """

    subgraphs: dict[str, SubgraphFactory] = Field(default_factory=dict)

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
        # Load subgraph providers from YAML config (config/ekg.yaml)
        cfg_name = global_config().get("kg_config", default=global_config().get("default_kg_config", default="db_only"))

        # Get subgraphs: [{factory: "module:Class", initial_load: [...]}, ...]
        try:
            subgraphs = global_config().get_list(f"kg_configs.{cfg_name}.subgraphs")
        except Exception:
            subgraphs = []

        # Extract and import factory classes
        for subgraph_cfg in subgraphs:
            if not isinstance(subgraph_cfg, dict) or "factory" not in subgraph_cfg:
                continue

            factory = subgraph_cfg["factory"]
            try:
                logger.debug(f"import {factory}")
                imported = import_from_qualified(factory)

                subgraph: SubgraphFactory | None = None

                # Already-instantiated Subgraph instance
                if isinstance(imported, SubgraphFactory):
                    subgraph = imported
                # Subgraph subclass – instantiate with config parameters
                elif isinstance(imported, type) and issubclass(imported, SubgraphFactory):
                    # Prepare constructor kwargs from YAML config (excluding factory, initial_load, trigger)
                    constructor_kwargs = {
                        k: v for k, v in subgraph_cfg.items() if k not in ["factory", "initial_load", "trigger"]
                    }
                    subgraph = imported(**constructor_kwargs)  # type: ignore[call-arg]
                else:
                    # Callable provider – may be a factory returning a Subgraph
                    # or a legacy register(registry) function.
                    try:
                        candidate = imported(self)
                    except TypeError:
                        candidate = imported()

                    if isinstance(candidate, SubgraphFactory):
                        subgraph = candidate
                    else:
                        # Legacy path: callable handled registration itself.
                        continue

                if subgraph is not None:
                    subgraph.register(self)

            except Exception as ex:
                import traceback

                logger.warning(f"Cannot load subgraph provider {factory}: {ex}")
                logger.debug(traceback.format_exc())

    @once
    def get_instance() -> "GraphRegistry":
        """Get the global GraphRegistry instance."""
        return GraphRegistry()

    def register_subgraph(self, name: str, subgraph: SubgraphFactory) -> None:
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
            - Node configurations are de-duplicated by their underlying Pydantic ``node_class``.
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
            schema = self.subgraphs[name].build_schema()
            schemas.append(schema)

        if not schemas:
            raise ValueError("No schemas could be built from the selected subgraphs")

        # Use the root_model_class of the first schema; other schemas may use
        # different roots but their node/relationship configurations are still
        # meaningful when merged.
        root_model_class = schemas[0].root_model_class

        # Track all root model classes from all schemas for validation
        merged_root_classes = [schema.root_model_class for schema in schemas]

        # Merge nodes, de-duplicating by underlying Pydantic class
        merged_nodes: list[Any] = []
        seen_node_classes: set[type] = set()
        for schema in schemas:
            for node in schema.nodes:
                node_class = node.node_class
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

        return GraphSchema(
            root_model_class=root_model_class,
            nodes=merged_nodes,
            relations=merged_relations,
            merged_root_classes=merged_root_classes,
        )

    def get_subgraph(self, name: str) -> SubgraphFactory:
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


# @beartype_nop
def register_subgraph(
    name: str, subgraph: SubgraphFactory, registry: Any = None
) -> None:  # registry is "Optional[GraphRegistry]"
    """Convenience wrapper to register a subgraph on the global registry.

    The optional ``registry`` argument allows explicit control over
    which registry instance receives the registration and avoids
    recursive calls to :meth:`GraphRegistry.get_instance` during
    initialisation.
    """
    target = registry if registry is not None else GraphRegistry.get_instance()
    target.register_subgraph(name, subgraph)


def get_subgraph(name: str) -> SubgraphFactory:
    """Convenience wrapper to retrieve a subgraph from the global registry."""
    return GraphRegistry.get_instance().get_subgraph(name)


_ = GraphRegistry.get_instance()
