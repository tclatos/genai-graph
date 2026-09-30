"""Graph data extraction from Pydantic models.

This module provides functions for extracting graph data from Pydantic models
and creating the graph schema and nodes/relationships in the database.
"""

import contextlib
import json
from datetime import UTC, datetime
from typing import Any, NamedTuple, Union

from genai_tk.config_mgmt.config_mngr import global_config
from genai_tk.core.factories.embeddings_factory import EmbeddingsFactory
from loguru import logger
from pydantic import BaseModel
from rich.console import Console

from genai_graph.kg.backend import KgBackend, create_in_memory_backend
from genai_graph.kg.embeddings_handler import EmbeddingsHandler
from genai_graph.kg.ingest.extra_fields import apply_extra_fields
from genai_graph.kg.ingest.merge import (
    NodeDataCollection,
    NodeTypeRegistry,
    merge_nodes_batch,
    merge_relationships_batch,
)
from genai_graph.kg.manager import KgManager
from genai_graph.kg.schema.core import GraphNode, GraphRelation, GraphSchema, _find_embedded_field_for_class


class TypedNull:
    """Marker for NULL values with explicit type information for Kuzu STRUCT fields."""

    def __init__(self, type_name: str) -> None:
        self.type_name = type_name

    def __repr__(self) -> str:
        return f"CAST(NULL AS {self.type_name})"


class NodeRecord(NamedTuple):
    """Structured representation of a graph node.

    Attributes:
        node_id: Unique identifier for the node.
        properties: Node properties dictionary.
    """

    node_id: str
    properties: dict[str, Any]


class RelationshipRecord(NamedTuple):
    """Structured representation of a graph relationship.

    Attributes:
        from_type: Source node label (table name).
        from_id: Source node primary key value.
        to_type: Target node label (table name).
        to_id: Target node primary key value.
        name: Relationship type name.
        properties: Edge properties dictionary.
    """

    from_type: str
    from_id: str
    to_type: str
    to_id: str
    name: str
    properties: dict[str, Any]


# Import new schema types

console = Console()


# Database helpers


def _get_kuzu_type(annotation: Any) -> str:
    """Map Python type annotation to Kuzu type string.

    This is a low-level helper used by the Kuzu-backed implementation to pick an
    appropriate scalar type. Structured (MAP/STRUCT) fields for embedded models
    are handled separately in ``create_schema``.

    Args:
        annotation: Python type annotation

    Returns:
        Kuzu type string
    """
    import types
    import typing
    from typing import get_origin

    if annotation is None:
        return "STRING"

    origin = get_origin(annotation)
    actual_type = annotation

    # Handle Optional[...] types by unwrapping to get the inner type
    # Supports both typing.Union and types.UnionType (Python 3.10+)
    if origin is typing.Union or origin is types.UnionType:
        args = typing.get_args(annotation)
        # Optional[X] is Union[X, None], so extract X
        if len(args) == 2 and type(None) in args:
            actual_type = args[0] if args[1] is type(None) else args[1]
            origin = get_origin(actual_type)

    # Check if it's a list type (after unwrapping Optional)
    if origin is list:
        # Check element type for list[float] -> FLOAT[]
        args = typing.get_args(actual_type)
        if args and args[0] is float:
            return "FLOAT[]"
        return "STRING[]"
    elif actual_type is int:
        return "INT64"
    elif actual_type is float:
        return "DOUBLE"
    else:
        # Fallback for strings, enums and complex types that are not marked as embedded
        return "STRING"


def _add_embedded_fields(
    parent_data: dict[str, Any], root_model: BaseModel, _all_nodes: list[GraphNode], parent_node: GraphNode
) -> None:
    """Add embedded struct fields to the parent record as nested maps.

    Embedded structs are configured via ``GraphNode.extra_classes`` using
    plain Pydantic models (non-:class:`ExtraFields` subclasses). For each such
    class we locate the corresponding field on ``parent_node.node_class`` and
    copy its ``model_dump()`` into the parent data.
    """
    if not parent_node.embedded_struct_classes:
        return

    # Locate the parent instance under the root model using the primary
    # field path selected for this node configuration.
    field_path = getattr(parent_node, "_field_path", parent_node.field_paths[0] if parent_node.field_paths else None)
    parent_instance = get_field_by_path(root_model, field_path) if field_path else root_model

    if parent_instance is None:
        return

    for embedded_cls in parent_node.embedded_struct_classes:
        field_name = _find_embedded_field_for_class(parent_node.node_class, embedded_cls)
        if not field_name:
            continue

        embedded_data = getattr(parent_instance, field_name, None)
        if embedded_data is None:
            continue

        if hasattr(embedded_data, "model_dump"):
            embedded_dict = embedded_data.model_dump()
        elif isinstance(embedded_data, dict):
            embedded_dict = embedded_data
        else:
            embedded_dict = dict(getattr(embedded_data, "__dict__", {}))

        # Normalize the embedded dict to match expected schema types
        embedded_dict = _normalize_embedded_dict(embedded_dict, embedded_cls)

        parent_data[field_name] = embedded_dict


def _normalize_embedded_dict(data: dict[str, Any], model_class: type[BaseModel]) -> dict[str, Any]:
    """Normalize embedded dict values to match Pydantic model schema types.

    This handles common type mismatches and ensures all fields are present:
    - Empty strings -> TypedNull for numeric fields
    - String values -> [string] for list fields
    - Missing fields -> TypedNull (ensures complete STRUCT for Kuzu)
    - None values -> TypedNull with proper type for Kuzu

    Args:
        data: Dictionary to normalize
        model_class: The Pydantic model class defining the expected schema

    Returns:
        Normalized dictionary with type-consistent values and all fields present
    """
    from typing import get_args, get_origin

    if not hasattr(model_class, "model_fields"):
        return data

    normalized = {}

    # Ensure ALL fields from the model are present in the output
    # This is critical for Kuzu's STRUCT type which requires complete structs
    for field_name, field_info in model_class.model_fields.items():
        value = data.get(field_name)

        # Get the field type annotation
        annotation = field_info.annotation
        origin = get_origin(annotation)

        # Unwrap Optional types
        is_optional = False
        if origin is Union or (hasattr(annotation, "__args__") and type(None) in get_args(annotation)):
            is_optional = True
            args = get_args(annotation)
            # Get the non-None type
            actual_type = next((arg for arg in args if arg is not type(None)), None)
            if actual_type:
                origin = get_origin(actual_type)
                annotation = actual_type

        # Determine Kuzu type name for NULL casting
        kuzu_type = None
        if annotation is int or (origin and origin is int):
            kuzu_type = "INT64"
        elif annotation is float or (origin and origin is float):
            kuzu_type = "DOUBLE"
        elif annotation is str or (origin and origin is str):
            kuzu_type = "STRING"
        elif annotation is bool or (origin and origin is bool):
            kuzu_type = "BOOL"

        # Handle None/empty values with typed NULL
        if value is None or (isinstance(value, str) and value.strip() == ""):
            if kuzu_type and is_optional:
                # Use TypedNull for proper Kuzu STRUCT initialization
                normalized[field_name] = TypedNull(kuzu_type)
            else:
                normalized[field_name] = None
            continue

        # Handle type mismatches for list fields
        if origin is list:
            if isinstance(value, str):
                # Convert single string to list
                if value.strip():
                    normalized[field_name] = [value]
                else:
                    # Empty string for a list field -> typed NULL
                    normalized[field_name] = TypedNull("STRING[]") if is_optional else None
            elif isinstance(value, list):
                # Empty list -> typed NULL for consistency
                if len(value) == 0 and is_optional:
                    normalized[field_name] = TypedNull("STRING[]")
                else:
                    normalized[field_name] = value
            else:
                # Fallback: try to convert to list
                try:
                    converted = list(value) if value else None
                    if converted is None and is_optional:
                        normalized[field_name] = TypedNull("STRING[]")
                    else:
                        normalized[field_name] = converted
                except (TypeError, ValueError):
                    if is_optional:
                        normalized[field_name] = TypedNull("STRING[]")
                    else:
                        normalized[field_name] = None
        else:
            # Keep value as-is, but if missing, set to TypedNull
            if field_name not in data and is_optional and kuzu_type:
                normalized[field_name] = TypedNull(kuzu_type)
            else:
                normalized[field_name] = value

    return normalized


def restart_database() -> KgBackend:
    """Restart the database by creating a fresh in-memory backend.

    Returns:
        KgBackend instance connected to an in-memory database
    """

    backend = create_in_memory_backend()
    logger.debug("Database restarted - all tables cleared")
    return backend


# Schema


def _add_missing_columns(backend: KgBackend, table_name: str, fields: list[str]) -> None:
    """Add columns to an already-existing node table that are missing from its DB schema.

    Handles schema evolution: `CREATE TABLE IF NOT EXISTS` is a no-op on a table
    that already exists, so a Pydantic model field added after a table was first
    created (e.g. `Folder.parent_folder_id`) would otherwise silently never reach
    the database and later fail at merge time with "Cannot find property ...".

    Args:
        backend: Connected KgBackend.
        table_name: Node table name (matches the Pydantic model's class name).
        fields: `"field_name TYPE"` declarations, as passed to `CREATE TABLE`.
    """
    try:
        existing_cols = {str(row[1]) for row in backend.execute(f"CALL table_info('{table_name}') RETURN *")}
    except Exception as exc:
        logger.debug(f"Could not introspect columns for {table_name}: {exc}")
        return

    for field_decl in fields:
        field_name = field_decl.split(" ", 1)[0]
        if field_name in existing_cols:
            continue
        try:
            backend.execute(f"ALTER TABLE {table_name} ADD {field_decl}")
            logger.info(f"Schema evolution: added missing column '{field_name}' to existing table {table_name}")
        except Exception as alter_exc:
            logger.warning(f"Could not add missing column '{field_name}' to {table_name}: {alter_exc}")


def create_schema(
    backend: KgBackend, nodes: list[GraphNode], relations: list[GraphRelation], context: KgManager | None = None
) -> None:
    """Create node and relationship tables in the graph database (idempotent).

    Creates CREATE NODE TABLE IF NOT EXISTS and CREATE REL TABLE IF NOT EXISTS statements
    based on GraphNode and GraphRelationConfig. This function is safe to call
    multiple times - it will not drop existing tables, allowing incremental additions.
    Embedded nodes have their fields merged into parent tables.

    Args:
        backend: KgBackend instance
        nodes: List of GraphNode objects
        relations: List of GraphRelationConfig objects
        context: Optional KgContext for collecting warnings
    """
    # TODO: Handle schema evolution by detecting new node or relationship types dynamically
    # and creating missing tables on the fly. This would allow adding new document types
    # with extended schemas without requiring database restarts.

    # Create node tables
    created_tables: set[str] = set()
    # For embedded configuration, we represent each embedded class as a
    # single MAP/STRUCT-typed column on the parent node.
    embedded_struct_fields_by_parent: dict[str, list[tuple[str, str]]] = {}

    # First, collect embedded struct definitions for each parent
    for node in nodes:
        if not node.embedded_struct_classes:
            continue

        parent_name = node.node_class.__name__
        if parent_name not in embedded_struct_fields_by_parent:
            embedded_struct_fields_by_parent[parent_name] = []

        parent_model_fields = getattr(node.node_class, "model_fields", {})

        for embedded_class in node.embedded_struct_classes:
            field_name = _find_embedded_field_for_class(node.node_class, embedded_class)
            if not field_name:
                continue

            # Validate that the embedded field exists on the parent model
            if field_name not in parent_model_fields:
                warning_msg = f"Embedded field '{field_name}' is not defined on {parent_name}"
                logger.warning(warning_msg)
                if context:
                    context.add_warning(warning_msg)
                continue

            # Ensure we can introspect the embedded class
            embedded_model_fields = getattr(embedded_class, "model_fields", None)
            if embedded_model_fields is None:
                warning_msg = (
                    f"Embedded class {embedded_class!r} for field '{field_name}' "
                    f"on {parent_name} has no model_fields; skipping STRUCT generation"
                )
                logger.warning(warning_msg)
                if context:
                    context.add_warning(warning_msg)
                continue

            # Build STRUCT(field1 TYPE, field2 TYPE, ...) definition
            struct_parts: list[str] = []
            for emb_field_name, emb_field_info in embedded_model_fields.items():
                kuzu_type = _get_kuzu_type(emb_field_info.annotation)
                struct_parts.append(f"{emb_field_name} {kuzu_type}")

            if not struct_parts:
                warning_msg = (
                    f"Embedded class {embedded_class.__name__} for field "
                    f"'{field_name}' on {parent_name} has no fields; skipping"
                )
                logger.warning(warning_msg)
                if context:
                    context.add_warning(warning_msg)
                continue

            struct_type = f"STRUCT({', '.join(struct_parts)})"
            embedded_struct_fields_by_parent[parent_name].append((field_name, struct_type))

    # Create node tables
    for node in nodes:
        table_name = node.node_class.__name__
        if table_name in created_tables:
            continue

        # Determine the primary key field
        key_from = node.key_from
        if key_from == "AUTO_ID":
            key_field = "id"
        elif isinstance(key_from, str):
            key_field = key_from  # Use the specified field
        else:
            # Callable - store computed key in 'id' field
            key_field = "id"

        fields: list[str] = []
        field_names: set[str] = set()
        model_fields = node.node_class.model_fields

        # Add primary key field first if needed
        if key_from == "AUTO_ID" or callable(key_from) or key_from == "id":
            # AUTO_ID generates UUID, callable computes key, or explicit 'id' field
            # - all stored as STRING in the 'id' column
            fields.append("id STRING")
            field_names.add("id")
        # If key_from is a different field name, that field will be added from model_fields

        # Add other metadata fields
        fields.append("name STRING")  # Node name from name_from (user-chosen)
        fields.append("_original_name STRING")  # Original Pydantic 'name' field if it existed
        fields.append("_created_at STRING")  # ISO timestamp
        fields.append("_updated_at STRING")  # ISO timestamp
        field_names.update({"name", "_original_name", "_created_at", "_updated_at"})

        # Resolve embedded struct field types for this table, if any
        embedded_struct_fields = dict(embedded_struct_fields_by_parent.get(table_name, []))

        # Metadata field names that are handled separately and should not be added from model_fields
        metadata_field_names = {"id", "name", "created_at", "updated_at"}

        # Add regular fields (excluding any specified excluded_fields).
        # If a field is declared as embedded, we override its scalar type with
        # a STRUCT(...) definition so it becomes a MAP/STRUCT column.
        for field_name, field_info in model_fields.items():
            # Skip metadata fields (they're added separately with _ prefix)
            if field_name in metadata_field_names:
                continue
            if field_name not in node.excluded_fields:
                if field_name in embedded_struct_fields:
                    kuzu_type = embedded_struct_fields[field_name]
                elif field_name == "metadata":
                    # Persist metadata as a JSON string for maximum flexibility.
                    kuzu_type = "STRING"
                else:
                    kuzu_type = _get_kuzu_type(field_info.annotation)
                    # For pre-computed list[float] fields with a known embedding dimension,
                    # use FLOAT[N] so Kuzu can build a vector index on them.
                    if kuzu_type == "FLOAT[]" and field_name in node.embedding_field_dimensions:
                        kuzu_type = f"FLOAT[{node.embedding_field_dimensions[field_name]}]"
                fields.append(f"{field_name} {kuzu_type}")
                field_names.add(field_name)

        # Add embedding columns for index_fields when embeddings are enabled.
        if node.index_fields:
            # Pre-fetch all known embedding model infos for dimension lookup (no API key needed).
            try:
                all_models = {item.id: item for item in EmbeddingsFactory.known_list()}
            except Exception:
                all_models = {}
            try:
                default_model_id: str | None = global_config().get_str("kg_build.embeddings.default")
            except Exception:
                default_model_id = None

            for field_name, model_override in node.index_field_specs:
                model_id = model_override or default_model_id
                embedding_field = f"{field_name}_embedding"
                embedding_dim: int | None = None
                if model_id:
                    info = all_models.get(model_id)
                    if info:
                        embedding_dim = info.dimension
                if embedding_dim is None:
                    logger.warning(
                        f"Cannot determine embedding dimension for {table_name}.{embedding_field} "
                        f"(model={model_id}); skipping embedding column"
                    )
                    continue
                kuzu_embedding_type = f"FLOAT[{embedding_dim}]"
                if embedding_field in field_names:
                    # The field already exists (e.g. as a pre-computed list[float]
                    # Pydantic field like L3.description_embedding added by the
                    # model_fields loop as FLOAT[]).  Upgrade it to FLOAT[N] so
                    # Kuzu can build a vector index on it.
                    for i, f in enumerate(fields):
                        if f == f"{embedding_field} FLOAT[]":
                            fields[i] = f"{embedding_field} {kuzu_embedding_type}"
                            break
                else:
                    fields.append(f"{embedding_field} {kuzu_embedding_type}")
                    field_names.add(embedding_field)

        fields_str = ", ".join(fields)
        create_sql = f"CREATE NODE TABLE IF NOT EXISTS {table_name}({fields_str}, PRIMARY KEY({key_field}))"

        logger.debug(f"Creating node table: {create_sql}")
        backend.execute(create_sql)
        _add_missing_columns(backend, table_name, fields)
        created_tables.add(table_name)

    # Create relationship tables with properties from p_*_ fields or relation.properties
    for relation in relations:
        from_table = relation.from_node.label
        to_table = relation.to_node.label
        rel_name = relation.name

        # Check if properties are explicitly defined (e.g., from Neo4j mappings)
        rel_properties = []
        if hasattr(relation, "properties") and relation.properties:
            # Use properties from GraphRelation (Fix 2: Support Neo4j property mappings)
            for prop_name, prop_type in relation.properties.items():
                kuzu_type = _get_kuzu_type(prop_type)
                rel_properties.append(f"{prop_name} {kuzu_type}")
        elif hasattr(relation.to_node.node_class, "model_fields"):
            # Fallback: Find p_*_ properties from the to_node class
            for field_name, field_info in relation.to_node.node_class.model_fields.items():
                if field_name.startswith("p_") and field_name.endswith("_"):
                    # Extract the property name without p_ prefix and _ suffix
                    prop_name = field_name[2:-1]
                    kuzu_type = _get_kuzu_type(field_info.annotation)
                    rel_properties.append(f"{prop_name} {kuzu_type}")

        if rel_properties:
            props_str = ", ".join(rel_properties)
            create_rel_sql = f"CREATE REL TABLE IF NOT EXISTS {rel_name}(FROM {from_table} TO {to_table}, {props_str})"
        else:
            create_rel_sql = f"CREATE REL TABLE IF NOT EXISTS {rel_name}(FROM {from_table} TO {to_table})"

        logger.debug(f"Creating relationship table: {create_rel_sql}")
        backend.execute(create_rel_sql)


# Extraction helpers


def get_field_by_path(obj: BaseModel, path: str) -> Any:
    """Get an attribute by a dot-separated path.

    Args:
        obj: Root object or dict
        path: Dot path like a.b.c

    Returns:
        Value at that path or None if not found
    """
    try:
        current = obj
        for part in path.split("."):
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current
    except (AttributeError, KeyError, TypeError):
        return None


def extract_graph_data(
    model: BaseModel,
    nodes: list[GraphNode],
    relations: list[GraphRelation],
    source_key: str | None = None,
) -> tuple[NodeDataCollection, list[RelationshipRecord]]:
    """Generic extraction of nodes and relationships from any Pydantic model.

    Args:
        model: Pydantic model instance
        nodes: List of GraphNode objects
        relations: List of GraphRelation objects
        source_key: Optional source identifier for provenance tracking

    Returns:
        nodes_data: Collection of nodes grouped by type
        relationships: List of :class:`RelationshipRecord` instances
    """
    nodes_data = NodeDataCollection()
    relationships: list[RelationshipRecord] = []
    node_registry: dict[str, set[str]] = {}  # For deduplication: node_type -> set of dedup values
    id_registry: dict[str, dict[str, str]] = {}  # For relationships: node_type -> {dedup_value: _id}

    # Field paths are already set as _field_path in create_graph

    # Init buckets
    for node_info in nodes:
        node_type = node_info.node_class.__name__
        nodes_data.ensure_type(node_type)
        node_registry[node_type] = set()
        id_registry[node_type] = {}

    # Nodes
    embeddings_handlers: dict[str, EmbeddingsHandler] = {}
    for node_info in nodes:
        node_type = node_info.node_class.__name__

        # Process ALL field paths for this node type, not just the first one
        field_paths_to_process = node_info.field_paths or [None]

        # Skip nodes that have no paths in the root model and are not the root
        # model itself.  Such nodes (e.g. Document) are created by dedicated
        # Prefect tasks and must not be extracted from the root model data.
        if not node_info.field_paths and node_info.node_class is not type(model):
            continue

        for field_path in field_paths_to_process:
            field_data = get_field_by_path(model, field_path) if field_path else model
            if field_data is None:
                continue

            # Check if this field path represents a list. Guard against a
            # ``None`` field_path to keep type-checkers happy – the
            # is_list_at_paths mapping is keyed by concrete path strings.
            if field_path is not None and hasattr(node_info, "is_list_at_paths"):
                is_list = node_info.is_list_at_paths.get(field_path, False)
            else:
                is_list = False
            items = field_data if is_list else [field_data]

            for item in items:
                if item is None:
                    continue

                if hasattr(item, "model_dump"):
                    item_data = item.model_dump()  # type: ignore
                elif isinstance(item, dict):
                    item_data = item.copy()
                else:
                    continue

                # Preserve original 'name' field if it exists
                if "name" in item_data:
                    item_data["_original_name"] = item_data["name"]

                # Set 'name' from name_from using get_name_value (user-chosen node name)
                item_data["name"] = node_info.get_name_value(item_data, node_type)

                # Determine which field to use as the primary key in the database
                key_from = node_info.key_from
                if key_from == "AUTO_ID" or callable(key_from):
                    # AUTO_ID generates UUID, callable computes key - both stored in 'id' field
                    primary_key_field = "id"
                    key_value = node_info.get_key_value(item_data, node_type)
                    if key_value is None:
                        continue  # key_from returned None — skip this item
                    item_data[primary_key_field] = key_value
                else:
                    # Use the specified field as primary key
                    primary_key_field = key_from
                    key_value = node_info.get_key_value(item_data, node_type)
                    item_data[primary_key_field] = key_value

                # Filter out excluded fields to avoid complex data issues
                if node_info.excluded_fields:
                    for excluded_field in node_info.excluded_fields:
                        item_data.pop(excluded_field, None)

                # Add embedded fields to this parent record
                _add_embedded_fields(item_data, model, nodes, node_info)

                # Normalise legacy `metadata` and attach provenance.
                with contextlib.suppress(Exception):
                    # Defensive: do not break extraction if helper fails
                    apply_extra_fields(item_data, node_info, model, item, source_key)

                # When persisting to the database, store metadata as a JSON string
                # so that arbitrary keys/values are supported without schema
                # evolution on the Kuzu side.
                try:
                    if "metadata" in item_data and isinstance(item_data["metadata"], dict):
                        item_data["metadata"] = json.dumps(item_data["metadata"])
                except Exception:
                    # Best-effort only; falling back to the original value is fine.
                    pass

                # Add timestamps
                now = datetime.now(UTC).isoformat()
                item_data["_created_at"] = now
                item_data["_updated_at"] = now

                # Compute embeddings for index_fields when enabled.
                if node_info.index_fields:
                    try:
                        default_embeddings_id: str | None = global_config().get_str("kg_build.embeddings.default")
                    except Exception:
                        default_embeddings_id = None

                    for field_name, model_override in node_info.index_field_specs:
                        embeddings_id = model_override or default_embeddings_id
                        if not embeddings_id:
                            logger.debug(f"Skipping embedding for {node_type}.{field_name}; no model configured")
                            continue
                        embedding_field = f"{field_name}_embedding"
                        if embedding_field in item_data:
                            continue  # already pre-computed (e.g. Stratnav description_embedding)
                        field_value = item_data.get(field_name)
                        if not field_value or not isinstance(field_value, str):
                            continue
                        handler = embeddings_handlers.get(embeddings_id)
                        if handler is None:
                            try:
                                handler = EmbeddingsHandler(embeddings_id=embeddings_id)
                                embeddings_handlers[embeddings_id] = handler
                            except Exception as e:
                                logger.warning(f"Failed to initialize embeddings handler for {embeddings_id}: {e}")
                                continue
                        try:
                            item_data[embedding_field] = handler.compute_embeddings(field_value)
                        except Exception as e:
                            logger.warning(f"Failed to compute embedding for {node_type}.{field_name}: {e}")

                # Use primary key for deduplication
                if key_value not in node_registry[node_type]:
                    nodes_data.add(node_type, item_data)
                    node_registry[node_type].add(key_value)
                    # Register lookup key for relationship creation
                    # For AUTO_ID/callable keys, use 'name' field as both lookup key AND value
                    # since we'll match by name in the relationship MATCH query
                    # For field-based keys, use the actual key value
                    if key_from == "AUTO_ID" or callable(key_from):
                        lookup_key = item_data.get("name", key_value)
                        # Store name as the value too, since we'll match by name
                        id_registry[node_type][lookup_key] = lookup_key
                    else:
                        id_registry[node_type][key_value] = key_value

    # Relationships
    for relation_info in relations:
        from_type = relation_info.from_node.label
        to_type = relation_info.to_node.label

        # Skip relationships involving node types not in the nodes list
        if from_type not in node_registry or to_type not in node_registry:
            continue
        # The GraphNode instances on the relation ARE the node configs
        from_node_info = relation_info.from_node
        to_node_info = relation_info.to_node

        # Get field paths from relation config
        from_field_path = getattr(relation_info, "_from_field_path", None)
        to_field_path = getattr(relation_info, "_to_field_path", None)

        from_data = get_field_by_path(model, from_field_path) if from_field_path else model
        to_data = get_field_by_path(model, to_field_path) if to_field_path else None
        if from_data is None or to_data is None:
            # Skip if we couldn't find the target data
            continue

        # Handle from_data as list (iterate through each item)
        from_items = from_data if isinstance(from_data, list) else [from_data]

        for from_item in from_items:
            if from_item is None:
                continue

            # Get dedup value for from_node to lookup id
            raw_from = from_item.model_dump() if hasattr(from_item, "model_dump") else from_item
            # Fallback: best-effort conversion for unexpected types
            from_dict: dict[str, Any] = (
                raw_from if isinstance(raw_from, dict) else dict(getattr(raw_from, "__dict__", {}))
            )

            # Get lookup key for from_node - use name for AUTO_ID/callable, field value otherwise
            from_key_from = from_node_info.key_from
            if from_key_from == "AUTO_ID" or callable(from_key_from):
                # Use name field for lookup (must match what was stored in id_registry)
                from_lookup_key = from_node_info.get_name_value(from_dict, from_type)
            else:
                # Use the primary key field value
                from_lookup_key = from_dict.get(from_key_from)
            from_id = id_registry[from_type].get(from_lookup_key) if from_lookup_key else None

            if not from_id:
                continue  # Skip if we can't find the from node id

            to_items = to_data if isinstance(to_data, list) else [to_data]
            for to_item in to_items:
                if to_item is None:
                    continue
                raw_to = to_item.model_dump() if hasattr(to_item, "model_dump") else to_item
                to_dict: dict[str, Any]
                to_dict = raw_to if isinstance(raw_to, dict) else dict(getattr(raw_to, "__dict__", {}))

                # Get lookup key for to_node - use name for AUTO_ID/callable, field value otherwise
                to_key_from = to_node_info.key_from
                if to_key_from == "AUTO_ID" or callable(to_key_from):
                    # Use name field for lookup (must match what was stored in id_registry)
                    to_lookup_key = to_node_info.get_name_value(to_dict, to_type)
                else:
                    # Use the primary key field value
                    to_lookup_key = to_dict.get(to_key_from)
                to_id = id_registry[to_type].get(to_lookup_key) if to_lookup_key else None

                if to_id:
                    # Extract p_*_ properties from to_item for edge properties
                    edge_properties = {}
                    if hasattr(relation_info.to_node.node_class, "model_fields"):
                        for field_name in relation_info.to_node.node_class.model_fields:
                            if field_name.startswith("p_") and field_name.endswith("_"):
                                prop_name = field_name[2:-1]  # Remove p_ prefix and _ suffix
                                prop_value = to_dict.get(field_name)
                                if prop_value is not None:
                                    edge_properties[prop_name] = prop_value

                    # Use id values for relationships with properties
                    relationships.append(
                        RelationshipRecord(
                            from_type=from_type,
                            from_id=from_id,
                            to_type=to_type,
                            to_id=to_id,
                            name=relation_info.name,
                            properties=edge_properties,
                        )
                    )

    return nodes_data, relationships


# Loading


def load_graph_data(
    backend: KgBackend,
    nodes: list[GraphNode],
    nodes_data: NodeDataCollection,
    relationships: list[RelationshipRecord],
    context: KgManager | None = None,
) -> None:
    """Load nodes and relationships into the graph database.

    Uses Kuzu's DataFrame MERGE for efficient batch operations:
    - LOAD FROM df MERGE for batch node merging
    - LOAD FROM df MATCH ... CREATE for batch relationship creation

    Args:
        backend: KgBackend instance
        nodes: List of GraphNode configurations
        nodes_data: Collection of nodes grouped by type
        relationships: List of RelationshipRecord instances
        context: Optional KgManager for collecting warnings
    """
    # Build node type registry from GraphNode configurations
    registry = NodeTypeRegistry.from_graph_nodes(nodes)

    # Merge nodes using DataFrame-based batch operations
    logger.debug("Merging nodes into graph...")
    merge_result = merge_nodes_batch(
        conn=backend,
        nodes=nodes_data,
        registry=registry,
        context=context,
    )

    # Create relationships using DataFrame-based batch operations
    logger.debug(f"Creating {len(relationships)} relationships...")
    edge_props_count = sum(1 for r in relationships if r.properties)
    if edge_props_count > 0:
        logger.debug(f"  {edge_props_count} relationships have properties")

    relationships_created = merge_relationships_batch(
        conn=backend,
        relationships=relationships,
        registry=registry,
        id_mapping=merge_result.id_mapping,
    )
    logger.debug(f"Created {relationships_created} relationships")


# Orchestration


def import_neo4j_data(
    backend: KgBackend,
    nodes_data: NodeDataCollection,
    relationships: list[RelationshipRecord],
    context: KgManager | None = None,
    key_fields: dict[str, str] | None = None,
) -> tuple[NodeDataCollection, list[RelationshipRecord]]:
    """Import pre-built nodes and relationships directly into the graph database.

    This function is designed for Neo4j imports and other scenarios where data
    is already in the right format and doesn't need hierarchical extraction.
    It creates the necessary schema tables dynamically based on the data provided.

    Args:
        backend: Graph database backend
        nodes_data: Pre-built NodeDataCollection with nodes grouped by type
        relationships: Pre-built list of RelationshipRecord instances
        context: Optional KgManager for collecting warnings
        key_fields: Optional mapping of node_type -> primary key field name.
                    Defaults to using the actual key field from existing DB tables,
                    or 'id' for new tables.

    Returns:
        Tuple of (nodes_data, relationships) that were loaded into the graph
    """
    from genai_graph.kg.ingest.merge import (
        NodeTypeRegistry,
        merge_nodes_batch,
        merge_relationships_batch,
    )

    logger.info(f"Importing {nodes_data.total_count()} nodes and {len(relationships)} relationships")

    # Create dynamic schema for all node types in the data
    _create_dynamic_schema_for_nodes(backend, nodes_data, relationships, key_fields=key_fields)

    # Build registry: use provided key_fields or default to 'id'
    registry = NodeTypeRegistry()
    for node_type in nodes_data.types():
        kf = (key_fields or {}).get(node_type, "id")
        registry.add_type(node_type, key_field=kf)

    # Merge nodes using DataFrame-based batch operations
    logger.debug("Merging nodes into graph...")
    merge_result = merge_nodes_batch(
        conn=backend,
        nodes=nodes_data,
        registry=registry,
        context=context,
    )

    # Create relationships
    logger.debug(f"Creating {len(relationships)} relationships...")
    relationships_created = merge_relationships_batch(
        conn=backend,
        relationships=relationships,
        registry=registry,
        id_mapping=merge_result.id_mapping,
    )
    logger.info(f"Import complete: {nodes_data.total_count()} nodes, {relationships_created} relationships")

    return nodes_data, relationships


def _create_dynamic_schema_for_nodes(
    backend: KgBackend,
    nodes_data: NodeDataCollection,
    relationships: list[RelationshipRecord],
    key_fields: dict[str, str] | None = None,
) -> None:
    """Create Kuzu schema tables dynamically based on node data structure.

    This inspects the actual data to determine field types and creates appropriate
    node and relationship tables.

    Args:
        backend: Graph database backend
        nodes_data: Collection of nodes grouped by type
        relationships: List of relationships (used to determine rel table schema)
        key_fields: Optional mapping of node_type -> primary key field name.
                    Defaults to 'id' for each node type.
    """

    def _infer_kuzu_type(value: Any) -> str:
        """Infer Kuzu type from a Python value."""
        if value is None:
            return "STRING"
        if isinstance(value, bool):
            return "BOOL"
        if isinstance(value, int):
            return "INT64"
        if isinstance(value, float):
            return "DOUBLE"
        if isinstance(value, list):
            # Peek at first element to distinguish float arrays from string arrays
            if value and isinstance(value[0], (int, float)) and not isinstance(value[0], bool):
                return "FLOAT[]"
            return "STRING[]"
        # Check for string boolean values from Neo4j exports
        if isinstance(value, str):
            lower_val = value.lower()
            if lower_val in ("true", "false"):
                return "BOOL"
            # Check if it's a JSON-encoded float array (e.g. "[0.1, 0.2, ...]")
            stripped = value.strip()
            if stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list) and parsed and isinstance(parsed[0], (int, float)):
                        return "FLOAT[]"
                except (ValueError, AttributeError):
                    pass
        return "STRING"

    def _coerce_value(value: Any, kuzu_type: str) -> Any:
        """Coerce a value to match the expected Kuzu type."""
        if value is None:
            return None
        if kuzu_type == "BOOL":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() == "true"
            return bool(value)
        if kuzu_type == "INT64":
            if isinstance(value, int):
                return value
            try:
                return int(float(value))
            except (ValueError, TypeError):
                return None
        if kuzu_type == "DOUBLE":
            if isinstance(value, float):
                return value
            try:
                return float(value)
            except (ValueError, TypeError):
                return None
        if kuzu_type == "FLOAT[]":
            # Deserialize JSON-encoded string arrays (e.g. "[0.1, 0.2, ...]")
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return [float(v) for v in parsed]
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
                return []
            if isinstance(value, list):
                try:
                    return [float(v) for v in value]
                except (ValueError, TypeError):
                    return []
            return []
        return value

    def _get_field_types_from_data(data_list: list[dict[str, Any]]) -> dict[str, str]:
        """Analyze data to determine field types."""
        field_types: dict[str, str] = {}

        for item in data_list:
            for field_name, value in item.items():
                if field_name not in field_types:
                    field_types[field_name] = _infer_kuzu_type(value)
                elif value is not None:
                    current = field_types[field_name]
                    # Upgrade type if we see a more specific type
                    if current in ("STRING", "STRING[]"):
                        inferred = _infer_kuzu_type(value)
                        if inferred not in ("STRING", "STRING[]"):
                            field_types[field_name] = inferred

        return field_types

    # Create node tables
    for node_type, node_list in nodes_data.items():
        if not node_list:
            continue

        # Infer field types from data
        field_types = _get_field_types_from_data(node_list)

        # Coerce node data to match inferred types
        for node in node_list:
            for field_name, kuzu_type in field_types.items():
                if field_name in node:
                    node[field_name] = _coerce_value(node[field_name], kuzu_type)

        # Build field definitions
        fields: list[str] = []

        # Determine primary key for this node type
        pk_field = (key_fields or {}).get(node_type, "id")

        # Ensure PK field and 'name' are always present
        if pk_field not in field_types:
            field_types[pk_field] = "STRING"
        if "name" not in field_types:
            field_types["name"] = "STRING"

        # Add standard metadata fields
        metadata_fields = {
            "_created_at": "STRING",
            "_updated_at": "STRING",
        }

        for field_name, kuzu_type in field_types.items():
            fields.append(f"{field_name} {kuzu_type}")

        for field_name, kuzu_type in metadata_fields.items():
            if field_name not in field_types:
                fields.append(f"{field_name} {kuzu_type}")

        fields_str = ", ".join(fields)
        create_sql = f"CREATE NODE TABLE IF NOT EXISTS {node_type}({fields_str}, PRIMARY KEY({pk_field}))"

        logger.debug(f"Creating dynamic node table: {node_type}")
        try:
            backend.execute(create_sql)
        except Exception as e:
            logger.warning(f"Failed to create node table {node_type}: {e}")

    # Create relationship tables
    # Group relationships by (from_type, to_type, name) to create unique rel tables
    rel_schemas: dict[tuple[str, str, str], dict[str, str]] = {}

    for rel in relationships:
        key = (rel.from_type, rel.to_type, rel.name)
        if key not in rel_schemas:
            rel_schemas[key] = {}

        # Collect property types
        for prop_name, prop_value in rel.properties.items():
            if prop_name not in rel_schemas[key]:
                rel_schemas[key][prop_name] = _infer_kuzu_type(prop_value)
            elif prop_value is not None and rel_schemas[key][prop_name] == "STRING":
                # Upgrade type if we see a more specific type
                inferred = _infer_kuzu_type(prop_value)
                if inferred != "STRING":
                    rel_schemas[key][prop_name] = inferred

    # Coerce relationship properties to match inferred types
    # RelationshipRecord is a NamedTuple, so we need to create new records with coerced props
    coerced_relationships: list[RelationshipRecord] = []
    for rel in relationships:
        key = (rel.from_type, rel.to_type, rel.name)
        if key in rel_schemas:
            prop_types = rel_schemas[key]
            coerced_props = {}
            for prop_name, prop_value in rel.properties.items():
                if prop_name in prop_types:
                    coerced_props[prop_name] = _coerce_value(prop_value, prop_types[prop_name])
                else:
                    coerced_props[prop_name] = prop_value
            # Create new relationship record with coerced properties
            coerced_relationships.append(
                RelationshipRecord(
                    from_type=rel.from_type,
                    from_id=rel.from_id,
                    to_type=rel.to_type,
                    to_id=rel.to_id,
                    name=rel.name,
                    properties=coerced_props,
                )
            )
        else:
            coerced_relationships.append(rel)

    # Replace original relationships with coerced ones
    relationships.clear()
    relationships.extend(coerced_relationships)

    for (from_type, to_type, rel_name), prop_types in rel_schemas.items():
        if prop_types:
            props_str = ", ".join(f"{name} {kuzu_type}" for name, kuzu_type in prop_types.items())
            create_rel_sql = f"CREATE REL TABLE IF NOT EXISTS {rel_name}(FROM {from_type} TO {to_type}, {props_str})"
        else:
            create_rel_sql = f"CREATE REL TABLE IF NOT EXISTS {rel_name}(FROM {from_type} TO {to_type})"

        logger.debug(f"Creating dynamic rel table: {rel_name}")
        try:
            backend.execute(create_rel_sql)
        except Exception as e:
            logger.warning(f"Failed to create rel table {rel_name}: {e}")


def create_graph(
    backend: KgBackend,
    model: BaseModel,
    schema_config: GraphSchema,
    source_key: str | None = None,
    context: KgManager | None = None,
) -> tuple[NodeDataCollection, list[RelationshipRecord]]:
    """Create a knowledge graph from a Pydantic model in the configured graph database.

    Args:
        backend: Graph database backend
        model: Root instance to convert
        schema_config: GraphSchema object with node and relationship configurations
        source_key: Optional source key for provenance tracking
        context: Optional KgContext for collecting warnings

    Returns:
        Tuple of (nodes_data, relationships) that were used to populate the graph
    """
    # Check if this is the new GraphSchema format
    if not hasattr(schema_config, "nodes") or not hasattr(schema_config, "relations"):
        raise ValueError("create_graph now only accepts GraphSchema objects. Please update your configuration.")

    schema = schema_config
    logger.debug("Using GraphSchema format")

    # Print schema summary
    try:
        schema.print_schema_summary()
    except Exception:
        logger.debug(f"Schema with {len(schema.nodes)} nodes and {len(schema.relations)} relations")

    logger.debug("Creating database schema...")

    # Prepare nodes with computed is_list flags
    for node_config in schema.nodes:
        field_paths = node_config.field_paths or []
        field_path = field_paths[0] if field_paths else None

        # Check if field is a list by looking at the model field annotation
        is_list = False
        if field_path and hasattr(model, "model_fields"):
            try:
                field_obj = get_field_by_path(model, field_path)
                if isinstance(field_obj, list):
                    is_list = True
                # Also check the field annotation in the model
                parts = field_path.split(".")
                current_model = type(model)
                for part in parts[:-1]:
                    if hasattr(current_model, "model_fields") and part in current_model.model_fields:
                        field_info = current_model.model_fields[part]
                        if hasattr(field_info.annotation, "__origin__"):
                            current_model = field_info.annotation.__args__[0]
                # Check final field
                if hasattr(current_model, "model_fields") and parts[-1] in current_model.model_fields:
                    field_info = current_model.model_fields[parts[-1]]
                    if hasattr(field_info.annotation, "__origin__") and field_info.annotation.__origin__ is list:
                        is_list = True
            except Exception:
                pass

        # Store is_list as a dynamic attribute
        node_config._is_list = is_list  # type: ignore
        node_config._field_path = field_path  # type: ignore

    # Prepare relations with field paths
    for relation_config in schema.relations:
        if relation_config.field_paths:
            from_path, to_path = relation_config.field_paths[0]
            relation_config._from_field_path = from_path  # type: ignore
            relation_config._to_field_path = to_path  # type: ignore
        else:
            relation_config._from_field_path = None  # type: ignore
            relation_config._to_field_path = None  # type: ignore

    logger.debug("Creating database tables...")
    create_schema(backend, schema.nodes, schema.relations)

    logger.debug("Extracting and loading data...")
    nodes_data, relationships = extract_graph_data(model, schema.nodes, schema.relations, source_key=source_key)

    load_graph_data(backend, schema.nodes, nodes_data, relationships, context)

    logger.debug("Graph creation complete")
    logger.debug(f"Total nodes: {nodes_data.total_count()}")
    logger.debug(f"Total relationships: {len(relationships)}")

    return nodes_data, relationships
