"""Abstract base class for Knowledge Graph factories.

A KgFactory loads data and provides a GraphSchema for extraction.
The actual graph is built via extract_graph_data() → merge_nodes_batch().
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel
from rich.console import Console

from genai_graph.kg.schema.core import GraphSchema

console = Console()


class KgFactory(ABC, BaseModel):
    """Abstract base class for KG factory implementations.

    A KgFactory provides:
    - A GraphSchema defining node types and relationships
    - A method to load structured data by key for graph extraction
    """

    # Optional class constant - set for factories with a single root model type.
    TOP_CLASS: type[BaseModel] | None = None

    @property
    def name(self) -> str:
        """Name of this graph factory.

        Derived from TOP_CLASS if set, otherwise from build_schema().root_model_class.
        """
        if self.TOP_CLASS is not None:
            return self.TOP_CLASS.__name__
        # Fallback to root_model_class from schema
        schema = self.build_schema()
        if schema.root_model_class is not None:
            return schema.root_model_class.__name__
        return self.__class__.__name__

    @abstractmethod
    def get_struct_data_by_key(self, key: str) -> BaseModel | None:
        """Load structured data for the given key.

        The returned Pydantic model is then processed by extract_graph_data()
        according to the schema from build_schema().
        """
        ...

    @abstractmethod
    def build_schema(self) -> GraphSchema:
        """Build and return the graph schema configuration.

        The schema defines node types, relationships, and how to extract data
        from Pydantic models.
        """
        ...

    def get_node_labels(self) -> dict[str, str]:
        """Get mapping of node types to human-readable descriptions from schema."""
        schema = self.build_schema()
        return {node.node_class.__name__: node.description for node in schema.nodes}

    def get_relationship_labels(self) -> dict[str, tuple[str, str]]:
        """Get mapping of relationship types to (direction, meaning) tuples from schema."""
        schema = self.build_schema()
        result = {}
        for relation in schema.relations:
            direction = f"{relation.from_node.label} → {relation.to_node.label}"
            result[relation.name] = (direction, relation.description)
        return result

    def get_sample_queries(self) -> list[str]:
        """Get list of sample Cypher queries for this graph."""
        return []

    def config_fingerprint(self) -> str:
        """Compute a hex digest of this factory's configuration.

        Hashes all Pydantic-serialisable fields (excluding private attrs
        and the schema itself) so that changes in factory configuration
        (e.g. ``data_root``, ``include``/``exclude`` patterns) invalidate
        cached outputs.

        Returns:
            Hex string (xxh3_64).
        """
        from genai_tk.utils.hashing import buffer_digest

        # Dump only the config fields (not the schema or model methods)
        config_data = self.model_dump(mode="json", exclude_none=True)
        # Sort keys for determinism
        import json

        payload = json.dumps(config_data, sort_keys=True, default=str)
        return buffer_digest(payload.encode())

    def register(self, registry: Any = None) -> None:
        """Register this graph factory.

        If ``registry`` is not provided, the global :class:`GraphRegistry`
        instance is used.
        """
        from genai_graph.kg.schema.registry import register_graph

        register_graph(self.name, self, registry=registry)
