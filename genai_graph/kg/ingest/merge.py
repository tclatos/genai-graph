"""Merge nodes and relationships into the graph database.

This module provides utilities for adding nodes and edges to the graph,
handling automatic merging based on key fields (typically 'name').

Uses Ladybug's LOAD FROM capability with PyArrow tables for efficient batch operations:
- LOAD FROM arrow_table MERGE (n:NodeType {key: key}) for batch node merging
- LOAD FROM arrow_table MATCH ... CREATE for batch relationship creation
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

import pyarrow as pa
from loguru import logger
from pydantic import BaseModel, Field

from genai_graph.kg.backend import KgBackend
from genai_graph.kg.ingest.arrow_utils import (
    arrow_type_contains_struct,
    pydantic_annotation_to_arrow,
)
from genai_graph.kg.manager import KgManager

if TYPE_CHECKING:
    from genai_graph.kg.ingest.extract import RelationshipRecord
    from genai_graph.kg.schema.core import GraphNode


# =============================================================================
# Type definitions for node data structures
# =============================================================================


def _build_node_arrow_schema(
    node_class: type,
    primary_key_field: str = "id",
    excluded_fields: set[str] | None = None,
    embedded_struct_classes: list[type] | None = None,
    embedding_field_dimensions: dict[str, int] | None = None,
) -> pa.Schema:
    """Build a ``pa.Schema`` for a Pydantic node class that mirrors the Ladybug table.

    Column rules (applied in order):

    1. **Sentinel columns** — ``primary_key_field``, ``"name"``, ``"_original_name"``
       are always prepended as ``pa.string()``, deduplicated when
       ``primary_key_field == "name"``.
    2. **Excluded fields** — skipped (``p_*_`` edge-property sentinels and
       relationship-target sub-model fields from
       ``GraphSchema._compute_excluded_fields``).
    3. **Embedded structs** — sub-models in ``embedded_struct_classes`` become
       ``pa.struct(...)`` columns in model-definition order.
    4. **Embedding vectors** — ``*_embedding`` fields become ``pa.list_(pa.float64())``.
    5. **Un-listed struct-typed fields** — skipped (safety net for rel targets).
    6. **Everything else** — type from :func:`~arrow_utils.pydantic_annotation_to_arrow`.
    7. **Timestamp sentinels** — ``_created_at`` / ``_updated_at`` appended if absent.

    Args:
        node_class: Pydantic class whose ``model_fields`` define the columns.
        primary_key_field: Primary key column name.
        excluded_fields: Column names to skip (p_*_ sentinels + rel targets).
        embedded_struct_classes: Sub-model classes stored inline as STRUCT columns.
        embedding_field_dimensions: Field name → vector dimension (flags a field as
            an embedding; stored type is always ``list<float64>``).

    Returns:
        ``pa.Schema`` ready for LOAD FROM MERGE operations.
    """
    from genai_graph.kg.schema.core import find_embedded_field_for_class

    excluded = excluded_fields or set()
    struct_map: dict[str, type] = {
        field_name: emb_cls
        for emb_cls in (embedded_struct_classes or [])
        if (field_name := find_embedded_field_for_class(node_class, emb_cls))
    }
    emb_dims = embedding_field_dimensions or {}
    fields: list[pa.Field] = []
    seen: set[str] = set()

    for name in dict.fromkeys([primary_key_field, "name", "_original_name"]):
        if name not in excluded:
            fields.append(pa.field(name, pa.string()))
            seen.add(name)

    for field_name, field_info in getattr(node_class, "model_fields", {}).items():
        if field_name in seen or field_name in excluded:
            continue
        seen.add(field_name)
        if field_name in struct_map:
            emb_cls = struct_map[field_name]
            fields.append(
                pa.field(
                    field_name,
                    pa.struct(
                        [
                            pa.field(n, pydantic_annotation_to_arrow(fi.annotation))
                            for n, fi in emb_cls.model_fields.items()
                        ]
                    ),
                )
            )
        elif field_name in emb_dims or field_name.endswith("_embedding"):
            fields.append(pa.field(field_name, pa.list_(pa.float64())))
        else:
            arrow_type = pydantic_annotation_to_arrow(field_info.annotation)
            if not arrow_type_contains_struct(arrow_type):
                fields.append(pa.field(field_name, arrow_type))

    for ts in ("_created_at", "_updated_at"):
        if ts not in seen:
            fields.append(pa.field(ts, pa.string()))

    return pa.schema(fields)


# A single node's properties as a dictionary
NodeProperties = dict[str, Any]

# A list of nodes of the same type
NodeList = list[NodeProperties]


class NodeDataCollection(BaseModel):
    """Collection of nodes grouped by their type.

    This provides a typed wrapper around the common pattern of
    `dict[str, list[dict[str, Any]]]` used throughout the graph creation code.

    Each key is a node type name (e.g., "Person", "Opportunity"),
    and each value is a list of property dictionaries for nodes of that type.

    Example:
        ```python
        nodes = NodeDataCollection()
        nodes.add("Person", {"name": "Alice", "age": 30})
        nodes.add("Person", {"name": "Bob", "age": 25})
        nodes.add("Company", {"name": "Acme", "industry": "Tech"})

        # Access all persons
        for person in nodes.get("Person"):
            print(person["name"])

        # Get total count
        print(nodes.total_count())  # 3
        ```
    """

    data: dict[str, NodeList] = Field(default_factory=dict)

    def add(self, node_type: str, properties: NodeProperties) -> None:
        """Add a node with the given properties to the collection."""
        if node_type not in self.data:
            self.data[node_type] = []
        self.data[node_type].append(properties)

    def get(self, node_type: str) -> NodeList:
        """Get all nodes of a given type (empty list if none)."""
        return self.data.get(node_type, [])

    def ensure_type(self, node_type: str) -> None:
        """Ensure a node type exists in the collection (creates empty list if not)."""
        if node_type not in self.data:
            self.data[node_type] = []

    def types(self) -> list[str]:
        """Get all node types in this collection."""
        return list(self.data.keys())

    def items(self) -> list[tuple[str, NodeList]]:
        """Iterate over (node_type, node_list) pairs."""
        return list(self.data.items())

    def total_count(self) -> int:
        """Get total node count across all types."""
        return sum(len(nodes) for nodes in self.data.values())

    def __contains__(self, node_type: str) -> bool:
        return node_type in self.data

    def __getitem__(self, node_type: str) -> NodeList:
        return self.data[node_type]

    def __setitem__(self, node_type: str, nodes: NodeList) -> None:
        self.data[node_type] = nodes

    def __len__(self) -> int:
        return len(self.data)

    @classmethod
    def from_dict(cls, data: dict[str, list[dict[str, Any]]]) -> NodeDataCollection:
        """Create a NodeDataCollection from a raw dictionary."""
        return cls(data=data)

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        """Convert to a raw dictionary (for backward compatibility)."""
        return self.data


# =============================================================================
# Parquet Collector for capturing DataFrames during merge
# =============================================================================


class ParquetCollector(BaseModel):
    """Collects Arrow tables during merge operations for parquet export.

    This allows capturing the exact data being merged into the graph,
    avoiding the need to query it back out (which can hit Kuzu bugs).

    Thread-safe: all mutations are protected by a lock so that
    concurrent bundle preparation tasks can safely append data.
    """

    nodes: dict[str, pa.Table] = Field(default_factory=dict)
    relationships: dict[str, pa.Table] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    _lock: Any = None  # threading.Lock, lazily initialised

    def model_post_init(self, _context: Any) -> None:
        import threading

        object.__setattr__(self, "_lock", threading.Lock())

    def add_nodes(self, node_type: str, table: pa.Table) -> None:
        """Add or append node data for a node type (thread-safe)."""
        lock = object.__getattribute__(self, "_lock")
        with lock:
            if node_type in self.nodes:
                self.nodes[node_type] = pa.concat_tables([self.nodes[node_type], table], promote_options="default")
            else:
                self.nodes[node_type] = table

    def add_relationships(self, rel_type: str, table: pa.Table) -> None:
        """Add or append relationship data for a relationship type (thread-safe)."""
        lock = object.__getattribute__(self, "_lock")
        with lock:
            if rel_type in self.relationships:
                self.relationships[rel_type] = pa.concat_tables(
                    [self.relationships[rel_type], table], promote_options="default"
                )
            else:
                self.relationships[rel_type] = table

    def get_node_count(self) -> int:
        """Get total node count across all types."""
        return sum(t.num_rows for t in self.nodes.values())

    def get_relationship_count(self) -> int:
        """Get total relationship count across all types."""
        return sum(t.num_rows for t in self.relationships.values())


# Global collector instance - set by KG creation flow
_parquet_collector: ParquetCollector | None = None


def set_parquet_collector(collector: ParquetCollector | None) -> None:
    """Set the global parquet collector for the current KG creation."""
    global _parquet_collector
    _parquet_collector = collector


def get_parquet_collector() -> ParquetCollector | None:
    """Get the global parquet collector."""
    return _parquet_collector


# =============================================================================
# Data Classes for structured return types
# =============================================================================


class MergeStats(BaseModel):
    """Statistics for a single node type merge operation."""

    created: int = 0
    matched: int = 0
    total: int = 0

    def __str__(self) -> str:
        return f"created={self.created}, matched={self.matched}, total={self.total}"


class NodeIdMapping(BaseModel):
    """Mapping from original node IDs to merged database IDs.

    For non-AUTO_ID nodes: maps (node_type, key_value) -> key_value
    For AUTO_ID nodes: maps (node_type, name) -> name (used for relationship matching)
    """

    mapping_data: dict[str, str] = Field(default_factory=dict, alias="_mapping")

    def _make_key(self, node_type: str, original_id: str) -> str:
        """Create a string key from node_type and original_id."""
        return f"{node_type}::{original_id}"

    def add(self, node_type: str, original_id: str, merged_id: str) -> None:
        """Add a mapping entry."""
        self.mapping_data[self._make_key(node_type, original_id)] = merged_id

    def get(self, node_type: str, original_id: str, default: str | None = None) -> str:
        """Get the merged ID for an original ID."""
        result = self.mapping_data.get(self._make_key(node_type, str(original_id)))
        if result is not None:
            return result
        return default if default is not None else str(original_id)

    def __contains__(self, key: tuple[str, str]) -> bool:
        return self._make_key(key[0], key[1]) in self.mapping_data

    def __len__(self) -> int:
        return len(self.mapping_data)

    def items(self) -> list[tuple[tuple[str, str], str]]:
        """Return all mapping items."""
        result = []
        for k, v in self.mapping_data.items():
            parts = k.split(":.", 1)
            if len(parts) == 2:
                result.append(((parts[0], parts[1]), v))
        return result


class NodeMergeResult(BaseModel):
    """Result of a batch node merge operation."""

    stats: dict[str, MergeStats] = Field(default_factory=dict)
    id_mapping: NodeIdMapping = Field(default_factory=NodeIdMapping)

    def get_stats(self, node_type: str) -> MergeStats:
        """Get stats for a node type, creating if needed."""
        if node_type not in self.stats:
            self.stats[node_type] = MergeStats()
        return self.stats[node_type]

    def total_nodes(self) -> int:
        """Return total nodes across all types."""
        return sum(s.total for s in self.stats.values())


class NodeTypeConfig(BaseModel):
    """Configuration for how a node type should be merged.

    Encapsulates the primary key field and the explicit Arrow schema for merge operations.
    The schema is derived from the Pydantic node class and serves as the single source
    of truth for column names, types, struct field order, and embedding columns.
    """

    model_config = {"arbitrary_types_allowed": True, "extra": "forbid"}

    node_type: str
    primary_key_field: str = "id"
    # Explicit Arrow schema: one field per node property in schema-definition order.
    # Struct columns carry pa.struct(...) types with sub-fields in the exact order
    # Ladybug expects; embedding columns carry pa.list_(pa.float64()).
    # None only for dynamically-created configs (e.g. Neo4j imports with no Pydantic model).
    arrow_schema: pa.Schema | None = None

    @property
    def field_names(self) -> set[str]:
        """Set of all field names defined in the Arrow schema."""
        if self.arrow_schema is None:
            return set()
        return set(self.arrow_schema.names)

    @property
    def struct_field_names(self) -> set[str]:
        """Set of field names whose Arrow type is a struct."""
        if self.arrow_schema is None:
            return set()
        return {f.name for f in self.arrow_schema if pa.types.is_struct(f.type)}

    @classmethod
    def from_graph_node(cls, node: GraphNode) -> NodeTypeConfig:
        """Create config from a GraphNode definition."""
        node_type = node.node_class.__name__
        key_from = node.key_from
        primary_key_field = "id" if (key_from == "AUTO_ID" or callable(key_from)) else key_from

        arrow_schema = _build_node_arrow_schema(
            node_class=node.node_class,
            primary_key_field=primary_key_field,
            excluded_fields=set(node.excluded_fields),
            embedded_struct_classes=list(getattr(node, "embedded_struct_classes", None) or []),
            embedding_field_dimensions=dict(getattr(node, "_embedding_field_dimensions", None) or {}),
        )

        return cls(
            node_type=node_type,
            primary_key_field=primary_key_field,
            arrow_schema=arrow_schema,
        )


class NodeTypeRegistry(BaseModel):
    """Registry of node type configurations for merge operations."""

    configs: dict[str, NodeTypeConfig] = Field(default_factory=dict, alias="_configs")

    def register(self, config: NodeTypeConfig) -> None:
        """Register a node type configuration."""
        self.configs[config.node_type] = config

    def add_type(self, node_type: str, key_field: str = "id") -> None:
        """Add a node type with default configuration.

        This is a convenience method for dynamic schema creation where
        we don't have GraphNode definitions.

        Args:
            node_type: The node type name (table name)
            key_field: The primary key field name (default: "id")
        """
        config = NodeTypeConfig(
            node_type=node_type,
            primary_key_field=key_field,
        )
        self.register(config)

    def get(self, node_type: str) -> NodeTypeConfig:
        """Get config for a node type, with sensible defaults."""
        if node_type in self.configs:
            return self.configs[node_type]
        # Return default config if not registered
        return NodeTypeConfig(node_type=node_type)

    def __contains__(self, node_type: str) -> bool:
        return node_type in self.configs

    @classmethod
    def from_graph_nodes(cls, nodes: list[GraphNode]) -> NodeTypeRegistry:
        """Build registry from a list of GraphNode definitions."""
        registry = cls()
        for node in nodes:
            registry.register(NodeTypeConfig.from_graph_node(node))
        return registry


def _format_value_for_cypher(value: Any) -> str:
    """Format a Python value for use in Cypher-like queries.

    Handles strings (with escaping), lists, dicts (as MAP/STRUCT), None,
    booleans, and numbers according to Cypher syntax requirements.

    This representation is compatible with Kuzu (STRUCT fields) and can also
    be interpreted as nested map properties by future Neo4j backends.

    Args:
        value: Python value to format

    Returns:
        Formatted string ready for Cypher query insertion
    """
    # Check for TypedNull first (must import to check type)
    if hasattr(value, "__class__") and value.__class__.__name__ == "TypedNull":
        # Return the CAST expression directly
        return repr(value)
    elif value is None:
        return "NULL"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, str):
        # Empty strings should be NULL to avoid type inference issues in STRUCT_PACK
        if value.strip() == "":
            return "NULL"
        # Escape single quotes for Cypher
        escaped = value.replace("'", "\\'")
        return f"'{escaped}'"
    elif isinstance(value, list):
        # Empty lists should be NULL for STRUCT compatibility
        if len(value) == 0:
            return "NULL"
        # Recursively format list elements
        formatted_items = [_format_value_for_cypher(item) for item in value]
        return f"[{', '.join(formatted_items)}]"
    elif isinstance(value, dict):
        # Map / struct literal: {key: value, ...}
        # Empty dicts cannot be represented in Cypher, use NULL instead
        if not value:
            return "NULL"

        # Check if all values are None, empty, or TypedNull - if so, use NULL for the whole struct
        # This avoids Kuzu creating a struct with only NULL fields
        def is_null_like(v: Any) -> bool:
            return (
                v is None
                or (isinstance(v, str) and v.strip() == "")
                or (isinstance(v, list) and len(v) == 0)
                or (hasattr(v, "__class__") and v.__class__.__name__ == "TypedNull")
            )

        if all(is_null_like(v) for v in value.values()):
            return "NULL"

        # Format each value - TypedNull and NULLs will be handled appropriately
        items = [f"{k}: {_format_value_for_cypher(v)}" for k, v in value.items()]
        return "{" + ", ".join(items) + "}"
    elif isinstance(value, (int, float)):
        return str(value)
    elif hasattr(value, "value"):  # Enum types
        escaped = str(value.value).replace("'", "\\'")
        return f"'{escaped}'"
    else:
        # Complex objects - convert to string
        escaped = str(value).replace("'", "\\'")
        return f"'{escaped}'"


# =============================================================================
# Arrow table construction for batch merge
# =============================================================================


def _prepare_node_arrow_table(
    node_list: list[dict[str, Any]],
    config: NodeTypeConfig,
) -> pa.Table:
    """Build a PyArrow Table from node dicts using the config's explicit Arrow schema.

    The schema drives everything:
    - Column selection and order come from ``config.arrow_schema``
    - Struct columns are built with the exact sub-field types and order from the schema
    - Numeric null handling is derived from the Arrow type (float64 → NaN, int64 → NaN)
    - No runtime type inference — the schema is the authoritative contract

    If ``config.arrow_schema`` is None (dynamic/unregistered node type), columns are
    inferred from the data directly with no type coercion.

    Args:
        node_list: List of node data dictionaries
        config: NodeTypeConfig carrying the Arrow schema and primary key field

    Returns:
        PyArrow Table ready for ``LOAD FROM arrow_table`` MERGE operation
    """
    from genai_graph.kg.ingest.extract import TypedNull

    schema = config.arrow_schema

    def _arrow_null_for(arrow_type: pa.DataType) -> Any:
        """Return the appropriate Python null value for an Arrow type."""
        if pa.types.is_floating(arrow_type) or pa.types.is_integer(arrow_type):
            return float("nan")
        if pa.types.is_list(arrow_type):
            return []
        return None

    def clean_value(value: Any, arrow_type: pa.DataType | None) -> Any:
        """Recursively clean a value to match the expected Arrow type."""
        if isinstance(value, Enum):
            return value.value

        if value is None or isinstance(value, TypedNull):
            return _arrow_null_for(arrow_type) if arrow_type is not None else None

        if isinstance(value, dict):
            if arrow_type is not None and pa.types.is_struct(arrow_type):
                # Reorder dict keys to match schema-defined struct field order
                return {f.name: clean_value(value.get(f.name), f.type) for f in arrow_type}
            # Unknown dict (schema stores it as STRING) — serialise to JSON
            return json.dumps(value, default=str) if value else None

        if isinstance(value, list):
            inner_type = arrow_type.value_type if (arrow_type is not None and pa.types.is_list(arrow_type)) else None
            return [clean_value(item, inner_type) for item in value]

        return value

    if not node_list:
        return pa.table({}) if schema is None else pa.table({f.name: pa.array([], type=f.type) for f in schema})

    excluded = {"created_at", "updated_at", "dedup_key"}
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    # Clean all rows
    cleaned_nodes: list[dict[str, Any]] = []
    for node_data in node_list:
        cleaned: dict[str, Any] = {}
        for key, value in node_data.items():
            if key in excluded:
                continue
            arrow_type = schema.field(key).type if (schema is not None and key in schema.names) else None
            cleaned[key] = clean_value(value, arrow_type)
        if "_created_at" not in cleaned:
            cleaned["_created_at"] = timestamp
        if "_updated_at" not in cleaned:
            cleaned["_updated_at"] = timestamp
        cleaned_nodes.append(cleaned)

    if schema is not None:
        # Schema-driven path: iterate schema fields in order; include only schema fields
        # plus any extra *_embedding columns added dynamically (not in the Pydantic model).
        extra_embedding_cols = [
            k for k in cleaned_nodes[0].keys() if k.endswith("_embedding") and k not in schema.names
        ]

        arrays: list[pa.Array] = []
        names: list[str] = []

        for field in schema:
            values = [node.get(field.name) for node in cleaned_nodes]
            arrays.append(pa.array(values, type=field.type, from_pandas=True))
            names.append(field.name)

        # Dynamically-added embedding columns (index_fields injected by extract pipeline)
        for col in extra_embedding_cols:
            values = [node.get(col) for node in cleaned_nodes]
            arrays.append(pa.array(values, type=pa.list_(pa.float64())))
            names.append(col)

        return pa.table(dict(zip(names, arrays, strict=False)))

    # Fallback: no schema — infer types from data (dynamic Neo4j imports)
    all_columns = list(dict.fromkeys(k for node in cleaned_nodes for k in node))

    inferred_arrays: list[pa.Array] = []
    for col in all_columns:
        values = [node.get(col) for node in cleaned_nodes]
        # Still give embedding columns the correct list<float64> type
        if col.endswith("_embedding"):
            inferred_arrays.append(pa.array(values, type=pa.list_(pa.float64())))
        else:
            inferred_arrays.append(pa.array(values))

    return pa.table(dict(zip(all_columns, inferred_arrays, strict=False)))


def _get_columns_for_set_clause(
    columns: list[str],
    key_field: str,
    exclude_on_match: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Get column names for ON CREATE SET and ON MATCH SET clauses.

    Args:
        columns: List of column names
        key_field: Primary key field (excluded from SET)
        exclude_on_match: Fields to exclude from ON MATCH SET (like _created_at)

    Returns:
        Tuple of (on_create_columns, on_match_columns)
    """
    if exclude_on_match is None:
        exclude_on_match = {"_created_at"}  # Don't update creation timestamp on match

    all_columns = [c for c in columns if c != key_field]
    # Also exclude *_embedding columns from ON MATCH SET: Ladybug (and Kuzu) forbid
    # updating a property in-place when it is covered by a vector index.
    # Embeddings don't change unless the source text changes, so skipping them on
    # match is correct; a delete_first=true run will recreate them anyway.
    on_match_columns = [c for c in all_columns if c not in exclude_on_match and not c.endswith("_embedding")]

    return all_columns, on_match_columns


def merge_nodes_batch(
    conn: KgBackend,
    nodes: NodeDataCollection,
    registry: NodeTypeRegistry,
    context: KgManager | None = None,
) -> NodeMergeResult:
    """Merge multiple nodes using DataFrame-based batch operations.

    Uses Kuzu's LOAD FROM df MERGE capability for efficient batch inserts.
    This is significantly faster than individual MERGE queries.

    Args:
        conn: Graph database connection (kuzu.Connection or KgBackend)
        nodes: Node data collection
        registry: Node type configuration registry
        context: Optional KgManager for collecting warnings

    Returns:
        NodeMergeResult containing statistics and ID mappings
    """
    result = NodeMergeResult()

    for node_type, node_list in nodes.items():
        if not node_list:
            continue

        config = registry.get(node_type)
        primary_key_field = config.primary_key_field

        logger.debug(f"Merging {len(node_list)} {node_type} nodes via Arrow table...")

        type_stats = MergeStats(total=len(node_list))

        # Prepare Arrow table using the config's explicit schema
        arrow_table = _prepare_node_arrow_table(node_list, config)

        if arrow_table.num_rows == 0:
            result.stats[node_type] = type_stats
            continue

        # The schema-driven path already limits columns to those defined in the Arrow
        # schema plus dynamically-added *_embedding columns. No additional filtering
        # needed when a schema is present. For the schema-less fallback (dynamic Neo4j
        # imports), also skip filtering to preserve all data.

        # Get columns for SET clauses
        on_create_cols, on_match_cols = _get_columns_for_set_clause(arrow_table.column_names, primary_key_field)

        # Get the Kuzu connection (handle both KgBackend and raw connection)
        kuzu_conn = conn.conn if hasattr(conn, "conn") else conn  # type: ignore[union-attr]

        try:
            # Build MERGE query with ON CREATE/ON MATCH SET
            on_create_set = ", ".join([f"n.{c} = {c}" for c in on_create_cols])
            on_match_set = ", ".join([f"n.{c} = {c}" for c in on_match_cols])

            # Update the timestamp for ON MATCH
            timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            if "_updated_at" in on_match_cols:
                col_idx = arrow_table.column_names.index("_updated_at")
                arrow_table = arrow_table.set_column(
                    col_idx, "_updated_at", pa.array([timestamp] * arrow_table.num_rows)
                )

            merge_query = f"""
                LOAD FROM arrow_table
                MERGE (n:{node_type} {{{primary_key_field}: {primary_key_field}}})
                ON CREATE SET {on_create_set}
                ON MATCH SET {on_match_set}
            """

            # NOTE: arrow_table is read by name from this frame by Ladybug's LOAD FROM scanner.
            kuzu_conn.execute(merge_query)

            # Collect Arrow table for parquet export if collector is active.
            # Struct field order is already baked into the Arrow schema.
            collector = get_parquet_collector()
            if collector is not None:
                collector.add_nodes(node_type, arrow_table)

            # Stats - we can't easily distinguish created vs matched in batch mode
            type_stats.created = arrow_table.num_rows  # Approximation

            # Build ID mapping from Arrow columns
            key_values = arrow_table.column(primary_key_field).to_pylist()
            name_values = arrow_table.column("name").to_pylist() if "name" in arrow_table.column_names else None

            for i, key_value in enumerate(key_values):
                if key_value:
                    key_str = str(key_value)
                    result.id_mapping.add(node_type, key_str, key_str)

                    # For AUTO_ID nodes, also map name → UUID
                    if primary_key_field == "id" and name_values is not None:
                        name_value = str(name_values[i])
                        if name_value and name_value != key_str:
                            result.id_mapping.add(node_type, name_value, key_str)

        except Exception as e:
            error_msg = str(e)
            # Enhance error message with context — wrapped in its own try/except
            # so a bug in this block can never mask the original exception.
            try:
                if "Cannot find property" in error_msg:
                    match = re.search(r"Cannot find property (\w+)", error_msg)
                    if match:
                        missing_prop = match.group(1)
                        schema_fields = list(config.field_names)[:10]
                        logger.error(
                            f"Schema mismatch for {node_type}: property '{missing_prop}' not in database schema. "
                            f"Schema fields: {', '.join(schema_fields)}. "
                            f"This usually means the field exists in data but wasn't defined in the node's Pydantic model."
                        )
                else:
                    logger.error(f"Error in batch merge for {node_type}: {e}")
            except Exception as fmt_exc:  # noqa: BLE001
                # Error-formatting failure: log it separately, then re-raise the original
                logger.error(f"Error in batch merge for {node_type}: {e}")
                logger.warning(f"(Error formatter also failed: {fmt_exc})")
            raise

        result.stats[node_type] = type_stats
        logger.debug(f"  {node_type}: {type_stats.total} processed via batch merge")

    return result


def _count_relationships(conn: KgBackend, rel_name: str) -> int:
    """Return the current row count of a relationship table (0 on a fresh/absent table)."""
    try:
        df = conn.execute_get_as_df(
            f"MATCH ()-[r:{rel_name}]->() RETURN count(r) AS c", None, union=False
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("count({}) failed: {}", rel_name, exc)
        return 0
    if df is None or df.empty:
        return 0
    return int(df.iloc[0, 0])


def merge_relationships_batch(
    conn: KgBackend,
    relationships: list[RelationshipRecord],
    registry: NodeTypeRegistry,
    id_mapping: NodeIdMapping,
) -> int:
    """Merge relationships using DataFrame-based batch operations.

    Groups relationships by type and uses ``LOAD FROM df MATCH ... MERGE`` for
    efficient batch relationship creation. No-property relationships use a
    single ``LOAD FROM`` with inline ``{key: col}`` node patterns (point
    lookups, no cross product); relationships with properties fall back to
    row-by-row parameterized ``MERGE ... SET``.

    Args:
        conn: Graph database connection
        relationships: List of RelationshipRecord objects
        registry: Node type configuration registry
        id_mapping: Mapping from (node_type, original_id) to merged_id

    Returns:
        Number of relationships actually created (before/after count delta per
        rel type, so idempotent MERGEs and the previously-silent no-property
        path no longer over-report).
    """
    if not relationships:
        return 0

    # Group relationships by (from_type, to_type, rel_name) for batch processing
    rel_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    for rel in relationships:
        from_type = rel.from_type
        from_id = rel.from_id
        to_type = rel.to_type
        to_id = rel.to_id
        rel_name = rel.name
        properties = rel.properties or {}

        # Translate IDs using mapping
        merged_from_id = id_mapping.get(from_type, str(from_id))
        merged_to_id = id_mapping.get(to_type, str(to_id))

        # Determine match fields based on whether nodes use AUTO_ID
        from_config = registry.get(from_type)
        to_config = registry.get(to_type)
        from_key_field = from_config.primary_key_field
        to_key_field = to_config.primary_key_field

        group_key = (from_type, to_type, rel_name)
        if group_key not in rel_groups:
            rel_groups[group_key] = []

        rel_data = {
            "from_id": merged_from_id,
            "to_id": merged_to_id,
            "from_key_field": from_key_field,
            "to_key_field": to_key_field,
            **properties,
        }
        rel_groups[group_key].append(rel_data)

    # Get the Kuzu connection
    kuzu_conn = conn.conn if hasattr(conn, "conn") else conn  # type: ignore[union-attr]

    total_created = 0

    for (from_type, to_type, rel_name), rel_list in rel_groups.items():
        if not rel_list:
            continue

        logger.debug(f"Creating {len(rel_list)} {rel_name} relationships ({from_type} -> {to_type})...")

        # Get the key fields from first relationship (all should be the same)
        from_key_field = rel_list[0]["from_key_field"]
        to_key_field = rel_list[0]["to_key_field"]

        # Build row data - remove key field info
        row_data: list[dict[str, Any]] = []
        property_cols: set[str] = set()
        for rel_data in rel_list:
            row: dict[str, Any] = {
                "from_id": rel_data["from_id"],
                "to_id": rel_data["to_id"],
            }
            for k, v in rel_data.items():
                if k not in ("from_id", "to_id", "from_key_field", "to_key_field"):
                    row[k] = v
                    property_cols.add(k)
            row_data.append(row)

        # Filter out properties that have all None/empty values
        non_empty_prop_cols: set[str] = set()
        for col in property_cols:
            for row in row_data:
                val = row.get(col)
                if val is not None and val != "":
                    non_empty_prop_cols.add(col)
                    break

        property_cols = non_empty_prop_cols

        # Use MERGE for relationships to avoid duplicates when the same relationship
        # is created from multiple sources (e.g., both BAML extraction and Neo4j import).
        # This ensures (from)-[r:REL]->(to) is only created once per node pair.
        # Count before/after so the total reflects rows actually created (MERGE is
        # idempotent, so the delta is exact) instead of the input row count, which
        # over-reports when rows already exist or — as the old no-property path did —
        # the MERGE silently matches nothing.
        before = _count_relationships(conn, rel_name)
        try:
            if property_cols:
                # Kuzu's LOAD FROM doesn't support inline property assignment
                # in MERGE for relationships. Use row-by-row creation with SET.
                prop_cols_list = sorted(property_cols)
                for row in row_data:
                    from_id_val = row["from_id"]
                    to_id_val = row["to_id"]
                    merge_q = (
                        f"MATCH (from:{from_type} {{{from_key_field}: $from_id}}), "
                        f"(to:{to_type} {{{to_key_field}: $to_id}}) "
                        f"MERGE (from)-[r:{rel_name}]->(to)"
                    )
                    set_parts = []
                    params: dict[str, Any] = {
                        "from_id": from_id_val,
                        "to_id": to_id_val,
                    }
                    for col in prop_cols_list:
                        val = row.get(col)
                        if val is not None and val != "":
                            # Convert lists to JSON for storage
                            if isinstance(val, list):
                                val = json.dumps(val)
                            param_name = f"p_{col}"
                            set_parts.append(f"r.{col} = ${param_name}")
                            params[param_name] = val
                    if set_parts:
                        merge_q += " SET " + ", ".join(set_parts)
                    kuzu_conn.execute(merge_q, parameters=params)
            else:
                # No properties: a single LOAD FROM with inline ``{key: col}``
                # node patterns — each endpoint is a point lookup, so there is no
                # CROSS PRODUCT (the concern that motivated the old two-stage
                # form). NOTE: arrow_rel_table is read by name from this frame by
                # Ladybug's LOAD FROM scanner.
                #
                # The previous two-stage form (``MATCH (from) WHERE from.k=from_id
                # WITH from, to_id MATCH (to) WHERE to.k=to_id``) silently created
                # 0 rows: ``WITH from, to_id`` dropped the LOAD FROM column binding
                # after the first MATCH, so the second MATCH matched nothing and
                # HAS_SECTION/HAS_SUBSECTION tables stayed empty despite the build
                # reporting a non-zero relationship count.
                arrow_rel_table = pa.table(  # noqa: F841
                    {
                        "from_id": [row["from_id"] for row in row_data],
                        "to_id": [row["to_id"] for row in row_data],
                    }
                )
                merge_rel_query = f"""
                    LOAD FROM arrow_rel_table
                    MATCH (from:{from_type} {{{from_key_field}: from_id}}),
                          (to:{to_type} {{{to_key_field}: to_id}})
                    MERGE (from)-[:{rel_name}]->(to)
                """
                try:
                    kuzu_conn.execute(merge_rel_query)
                except Exception as batch_err:
                    logger.warning(
                        f"Batch LOAD FROM failed for {rel_name} ({len(row_data)} rows): {batch_err}; falling back to point merges"
                    )
                    for r in row_data:
                        f_id = r["from_id"]
                        t_id = r["to_id"]
                        point_q = f"""
                            MATCH (from:{from_type} {{{from_key_field}: $from_id}}),
                                  (to:{to_type} {{{to_key_field}: $to_id}})
                            MERGE (from)-[:{rel_name}]->(to)
                        """
                        kuzu_conn.execute(point_q, parameters={"from_id": f_id, "to_id": t_id})
            total_created += max(0, _count_relationships(conn, rel_name) - before)

            # Collect Arrow table for parquet export if collector is active
            collector = get_parquet_collector()
            if collector is not None:
                export_cols: dict[str, list[Any]] = {
                    "from_id": [row["from_id"] for row in row_data],
                    "to_id": [row["to_id"] for row in row_data],
                }
                for col in property_cols:
                    export_cols[col] = [row.get(col) for row in row_data]
                export_cols["_from_type"] = [from_type] * len(row_data)
                export_cols["_to_type"] = [to_type] * len(row_data)
                export_cols["_from_key_field"] = [from_key_field] * len(row_data)
                export_cols["_to_key_field"] = [to_key_field] * len(row_data)
                collector.add_relationships(rel_name, pa.table(export_cols))

        except Exception as e:
            logger.error(f"Error in batch relationship creation for {rel_name}: {e}")
            logger.error(f"Query failed for {rel_name} ({from_type} -> {to_type})")
            raise

    return total_created
