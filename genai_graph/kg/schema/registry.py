"""Registry for available knowledge graph factories.

This module centralizes registration and lookup of KG graph factories so that
commands and core logic do not need hard dependencies on a particular
graph implementation module.
"""

from __future__ import annotations

import typing
from typing import Any

from genai_tk.config_mgmt.import_utils import ImportResolver
from genai_tk.utils.singleton import once
from loguru import logger
from pydantic import BaseModel, Field

from genai_graph.kg.manager import get_kg_manager
from genai_graph.kg.schema.core import GraphRelation, GraphSchema

if typing.TYPE_CHECKING:
    from genai_graph.kg.factories.base import KgFactory

from beartype import BeartypeConf, beartype

beartype_nop = beartype(conf=BeartypeConf(claw_decoration_position_funcs=None))


class GraphRegistry(BaseModel):
    """Singleton registry for knowledge graph factories.

    In addition to managing individual graph factories, the registry can also
    build *combined* graph schemas that merge multiple graphs into a
    single :class:`GraphSchema`. This is useful for commands that should
    operate on a logical union of several graphs.
    """

    graphs: dict[str, KgFactory] = Field(default_factory=dict)

    model_config = {
        "arbitrary_types_allowed": True,
    }

    def model_post_init(self, _context: Any) -> None:
        """Load and register configured graph factory providers.

        Each entry in the ``graphs`` configuration should resolve to one of
        the following:

        * A :class:`KgFactory` subclass (preferred) – it will be instantiated
          with default arguments and registered.
        * A callable returning a :class:`KgFactory` instance – the instance
          will be registered.
        """
        from genai_graph.kg.factories.base import KgFactory

        # Load graph factory providers for the active KG profile via KgManager
        manager = get_kg_manager()
        profile_cfg = manager.get_profile_dict()

        # Get graphs: [{factory: "module.Class", initial_load: [...]}, ...]
        graph_configs = profile_cfg.get("graphs", []) or []
        self._failed_factories: list[tuple[str, str]] = []  # (factory_path, error)

        # Extract and import factory classes
        for graph_cfg in graph_configs:
            if not isinstance(graph_cfg, dict) or "factory" not in graph_cfg:
                continue

            factory = graph_cfg["factory"]
            try:
                logger.debug(f"import {factory}")
                imported = ImportResolver.import_from_qualified(factory)

                graph_impl: KgFactory | None = None

                # Already-instantiated KgFactory instance
                if isinstance(imported, KgFactory):
                    graph_impl = imported
                # KgFactory subclass – instantiate with config parameters
                elif isinstance(imported, type) and issubclass(imported, KgFactory):
                    # Prepare constructor kwargs from YAML config (excluding factory, initial_load, trigger)
                    constructor_kwargs = {
                        k: v for k, v in graph_cfg.items() if k not in ["factory", "initial_load", "trigger"]
                    }
                    graph_impl = imported(**constructor_kwargs)  # type: ignore[call-arg]
                else:
                    # Callable provider – may be a factory returning a KgFactory.
                    try:
                        candidate = imported(self)
                    except TypeError:
                        candidate = imported()

                    if isinstance(candidate, KgFactory):
                        graph_impl = candidate
                    else:
                        continue

                if graph_impl is not None:
                    graph_impl.register(self)

            except Exception as ex:
                import traceback

                self._failed_factories.append((factory, str(ex)))
                logger.warning(f"Cannot load graph factory {factory}: {ex}")
                logger.debug(traceback.format_exc())

    @staticmethod
    @once
    def get_instance() -> GraphRegistry:
        """Get the global GraphRegistry instance."""
        # Rebuild model to resolve forward reference to KgFactory
        # This must happen before instantiation, and after factories are importable
        from genai_graph.kg.factories.base import KgFactory

        GraphRegistry.model_rebuild(_types_namespace={"KgFactory": KgFactory})
        return GraphRegistry()

    def register_graph(self, name: str, graph: KgFactory) -> None:
        """Register a graph factory under the given name."""
        self.graphs[name] = graph

    def build_combined_schema(self, graph_names: list[str] | None = None) -> GraphSchema:
        """Build a combined :class:`GraphSchema` from one or more graphs.

        Args:
            graph_names: Optional list of graph names to combine. If
                omitted or empty, all registered graphs are used.

        Returns:
            A new :class:`GraphSchema` instance whose nodes and relations are
            the union of the selected graphs.

        Notes:
            - Node configurations are de-duplicated by their underlying Pydantic ``node_class``.
            - Relationship configurations are de-duplicated by the
              ``(from_node, to_node, name)`` triple.
            - The ``root_model_class`` of the first selected graph is used
              for the combined schema; this is sufficient for documentation
              and visualization use cases where we only need the union of
              nodes/relations.
        """
        if not self.graphs:
            failed = getattr(self, "_failed_factories", [])
            if failed:
                details = "\n".join(f"  - {f}: {err}" for f, err in failed)
                raise ValueError(
                    f"No graphs registered — all {len(failed)} graph factory import(s) failed:\n"
                    f"{details}\n"
                    f"Fix the errors above and retry."
                )
            raise ValueError(
                "No graphs are registered in the GraphRegistry. "
                "Check that your KG profile config has a 'graphs' section with valid factory entries."
            )

        # Default to all registered graphs when none are explicitly provided
        if not graph_names:
            graph_names = sorted(self.graphs.keys())

        schemas: list[GraphSchema] = []
        for name in graph_names:
            if name not in self.graphs:
                available = ", ".join(sorted(self.graphs.keys())) or "<none>"
                raise ValueError(f"Unknown graph '{name}'. Available: {available}")
            schema = self.graphs[name].build_schema()
            schemas.append(schema)

        if not schemas:
            raise ValueError("No schemas could be built from the selected graphs")

        # Use the root_model_class of the first schema; other schemas may use
        # different roots but their node/relationship configurations are still
        # meaningful when merged.
        root_model_class = schemas[0].root_model_class

        # Track all root model classes from all schemas for validation
        # Filter out None values from schemas without root_model_class
        merged_root_classes = [schema.root_model_class for schema in schemas if schema.root_model_class is not None]

        # Merge nodes, de-duplicating by label (which may be table_name or class __name__).
        # When multiple nodes share the same label, prefer the one that appears first
        # (typically from more authoritative sources like Neo4j or database imports).
        merged_nodes: list[Any] = []
        seen_node_names: set[str] = set()
        for schema in schemas:
            for node in schema.nodes:
                node_name = node.label
                if node_name in seen_node_names:
                    continue
                seen_node_names.add(node_name)
                merged_nodes.append(node)

        # Merge relations, de-duplicating by (from_node_name, to_node_name, rel_name).
        # Use class names since different factories may use different classes with the same name.
        merged_relations: list[GraphRelation] = []
        seen_relations: set[tuple[str, str, str]] = set()
        for schema in schemas:
            for rel in schema.relations:
                key = (rel.from_node.label, rel.to_node.label, rel.name)
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

    def get_graph(self, name: str) -> KgFactory:
        """Get a graph factory by name.

        Args:
            name: Name of the graph to retrieve.

        Returns:
            KgFactory instance.

        Raises:
            ValueError: If graph name is not found.
        """
        if name not in self.graphs:
            available = ", ".join(sorted(self.graphs.keys())) or "<none>"
            raise ValueError(f"Unknown graph '{name}'. Available: {available}")
        return self.graphs[name]

    def list_graphs(self) -> list[str]:
        """List names of all registered graphs."""
        return sorted(self.graphs.keys())


def get_graph_registry() -> GraphRegistry:
    """Get the global GraphRegistry instance."""
    return GraphRegistry.get_instance()


def register_graph(name: str, graph: KgFactory, registry: Any = None) -> None:
    """Convenience wrapper to register a graph on the global registry.

    The optional ``registry`` argument allows explicit control over
    which registry instance receives the registration and avoids
    recursive calls to :meth:`GraphRegistry.get_instance` during
    initialisation.
    """
    target = registry if registry is not None else GraphRegistry.get_instance()
    target.register_graph(name, graph)


def get_graph(name: str) -> KgFactory:
    """Convenience wrapper to retrieve a graph from the global registry."""
    return GraphRegistry.get_instance().get_graph(name)
