import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, NamedTuple

from pydantic import BaseModel
from rich.console import Console

from genai_graph.core.extra_fields_utils import apply_extra_fields
from genai_graph.core.graph_backend import GraphBackend, create_in_memory_backend
from genai_graph.core.graph_merge import merge_nodes_batch
from genai_graph.core.graph_schema import GraphNode, GraphRelation, GraphSchema, _find_embedded_field_for_class


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
    import typing

    if annotation is None:
        return "STRING"

    origin = getattr(annotation, "__origin__", None)
    actual_type = annotation

    # Handle Optional[...] types by unwrapping to get the inner type
    if origin is typing.Union:
        args = typing.get_args(annotation)
        # Optional[X] is Union[X, None], so extract X
        if len(args) == 2 and type(None) in args:
            actual_type = args[0] if args[1] is type(None) else args[1]
            origin = getattr(actual_type, "__origin__", None)

    # Check if it's a list type
    if origin is list:
        return "STRING[]"
    elif actual_type in (float,):
        return "DOUBLE"
    elif actual_type in (int,):
        return "INT64"
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

        parent_data[field_name] = embedded_dict


def restart_database() -> GraphBackend:
    """Restart the database by creating a fresh in-memory backend.

    Returns:
        GraphBackend instance connected to an in-memory database
    """

    backend = create_in_memory_backend()
    console.print("[yellow]🔄 Database restarted - all tables cleared[/yellow]")
    return backend


# Schema


def create_schema(backend: GraphBackend, nodes: list[GraphNode], relations: list[GraphRelation]) -> None:
    """Create node and relationship tables in the graph database (idempotent).

    Creates CREATE NODE TABLE IF NOT EXISTS and CREATE REL TABLE IF NOT EXISTS statements
    based on GraphNode and GraphRelationConfig. This function is safe to call
    multiple times - it will not drop existing tables, allowing incremental additions.
    Embedded nodes have their fields merged into parent tables.

    Args:
        conn: Kuzu database connection
        nodes: List of GraphNode objects
        relations: List of GraphRelationConfig objects
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
                    console.print(
                        f"[yellow]Warning: embedded field '{field_name}' is not defined on {parent_name}[/yellow]"
                    )
                    continue

                # Ensure we can introspect the embedded class
                embedded_model_fields = getattr(embedded_class, "model_fields", None)
                if embedded_model_fields is None:
                    console.print(
                        f"[yellow]Warning: embedded class {embedded_class!r} for field '{field_name}' "
                        f"on {parent_name} has no model_fields; skipping STRUCT generation[/yellow]"
                    )
                    continue

                # Build STRUCT(field1 TYPE, field2 TYPE, ...) definition
                struct_parts: list[str] = []
                for emb_field_name, emb_field_info in embedded_model_fields.items():
                    kuzu_type = _get_kuzu_type(emb_field_info.annotation)
                    struct_parts.append(f"{emb_field_name} {kuzu_type}")

                if not struct_parts:
                    console.print(
                        f"[yellow]Warning: embedded class {embedded_class.__name__} for field "
                        f"'{field_name}' on {parent_name} has no fields; skipping[/yellow]"
                    )
                    continue

                struct_type = f"STRUCT({', '.join(struct_parts)})"
                embedded_struct_fields_by_parent[parent_name].append((field_name, struct_type))

    # Collect ExtraFields-based struct fields per parent node
    extra_struct_fields_by_parent: dict[str, list[tuple[str, str]]] = {}
    for node in nodes:
        extras = getattr(node, "extra_field_classes", []) or []
        if not extras:
            continue
        parent_name = node.node_class.__name__
        if parent_name not in extra_struct_fields_by_parent:
            extra_struct_fields_by_parent[parent_name] = []

        for extra_cls in extras:
            # Introspect pydantic model fields for the extra class
            extra_fields = getattr(extra_cls, "model_fields", None)
            if not extra_fields:
                continue
            struct_parts = []
            for ef_name, ef_info in extra_fields.items():
                kuzu_type = _get_kuzu_type(ef_info.annotation)
                struct_parts.append(f"{ef_name} {kuzu_type}")
            if struct_parts:
                field_name = "".join(["_" + c.lower() if c.isupper() else c for c in extra_cls.__name__]).lstrip("_")
                struct_type = f"STRUCT({', '.join(struct_parts)})"
                extra_struct_fields_by_parent[parent_name].append((field_name, struct_type))

    # Create node tables
    for node in nodes:
        table_name = node.node_class.__name__
        if table_name in created_tables:
            continue

        key_field = node.key
        fields: list[str] = []
        model_fields = node.node_class.model_fields

        # Add ExtraFields-based struct fields (synthetic extras)
        extra_struct_fields = dict(extra_struct_fields_by_parent.get(table_name, []))
        for field_name, struct_type in extra_struct_fields.items():
            fields.append(f"{field_name} {struct_type}")

        # Add metadata fields first
        fields.append("id STRING")  # UUID primary key
        fields.append("_name STRING")  # Human-readable name from name_from
        fields.append("_created_at STRING")  # ISO timestamp
        fields.append("_updated_at STRING")  # ISO timestamp
        # Unified deduplication key used for MERGE semantics
        fields.append("_dedup_key STRING")
        # Optional list of alternate names captured when merging nodes that
        # share the same dedup key but have different human-readable names
        fields.append("alternate_names STRING[]")

        # Resolve embedded struct field types for this table, if any
        embedded_struct_fields = dict(embedded_struct_fields_by_parent.get(table_name, []))

        # Add regular fields (excluding any specified excluded_fields).
        # If a field is declared as embedded, we override its scalar type with
        # a STRUCT(...) definition so it becomes a MAP/STRUCT column.
        for field_name, field_info in model_fields.items():
            if field_name not in node.excluded_fields:
                if field_name in embedded_struct_fields:
                    kuzu_type = embedded_struct_fields[field_name]
                elif field_name == "metadata":
                    # Create metadata as a STRUCT type only when not handled by an ExtraFields class
                    metadata_handled = any(
                        getattr(ec, "__name__", "") == "FileMetadata"
                        for ec in getattr(node, "extra_field_classes", []) or []
                    )
                    if metadata_handled:
                        # Skip creating legacy metadata column when FileMetadata is used
                        continue
                    kuzu_type = "STRUCT(source STRING)"
                else:
                    kuzu_type = _get_kuzu_type(field_info.annotation)
                fields.append(f"{field_name} {kuzu_type}")

        fields_str = ", ".join(fields)
        create_sql = f"CREATE NODE TABLE IF NOT EXISTS {table_name}({fields_str}, PRIMARY KEY({key_field}))"
        console.print(f"[cyan]Creating node table:[/cyan] {create_sql}")
        backend.execute(create_sql)
        created_tables.add(table_name)

    # Create relationship tables with properties from p_*_ fields
    for relation in relations:
        from_table = relation.from_node.__name__
        to_table = relation.to_node.__name__
        rel_name = relation.name

        # Find p_*_ properties from the to_node class
        rel_properties = []
        if hasattr(relation.to_node, "model_fields"):
            for field_name, field_info in relation.to_node.model_fields.items():
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

        console.print(f"[cyan]Creating relationship table:[/cyan] {create_rel_sql}")
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
) -> tuple[dict[str, list[dict[str, Any]]], list[RelationshipRecord]]:
    """Generic extraction of nodes and relationships from any Pydantic model.

    Args:
        model: Pydantic model instance
        nodes: List of GraphNode objects
        relations: List of GraphRelation objects

    Returns:
        nodes_dict: Mapping of node type to list of property dicts
        relationships: List of :class:`RelationshipRecord` instances
    """
    nodes_dict: dict[str, list[dict[str, Any]]] = {}
    relationships: list[RelationshipRecord] = []
    node_registry: Dict[str, set[str]] = {}  # For deduplication: node_type -> set of dedup values
    id_registry: Dict[str, Dict[str, str]] = {}  # For relationships: node_type -> {dedup_value: _id}

    # Field paths are already set as _field_path in create_graph

    # Init buckets
    for node_info in nodes:
        node_type = node_info.node_class.__name__
        nodes_dict[node_type] = []
        node_registry[node_type] = set()
        id_registry[node_type] = {}

    # Nodes
    for node_info in nodes:
        node_type = node_info.node_class.__name__

        # Process ALL field paths for this node type, not just the first one
        field_paths_to_process = node_info.field_paths if node_info.field_paths else [None]

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

                # Generate metadata fields FIRST, before removing excluded fields
                # Generate UUID for id
                item_data["id"] = str(uuid.uuid4())

                # Get _name from name_from using get_name_value
                item_data["_name"] = node_info.get_name_value(item_data, node_type)

                # Filter out excluded fields to avoid complex data issues
                if node_info.excluded_fields:
                    for excluded_field in node_info.excluded_fields:
                        item_data.pop(excluded_field, None)

                # Add embedded fields to this parent record
                _add_embedded_fields(item_data, model, nodes, node_info)

                # Normalize legacy `metadata` and populate any configured ExtraFields
                # Use the central helper to keep extraction logic consistent.
                try:
                    apply_extra_fields(item_data, node_info, model, item, source_key)
                except Exception:
                    # Defensive: do not break extraction if helper fails
                    pass

                # Add timestamps
                now = datetime.utcnow().isoformat() + "Z"
                item_data["_created_at"] = now
                item_data["_updated_at"] = now

                # Deduplication: use unified helper so that extraction, relationship
                # wiring, and DB merges all agree on a single canonical key.
                dedup_value = node_info.get_dedup_value(item_data, node_type)
                dedup_str: str | None = str(dedup_value) if dedup_value is not None else None

                # Always populate a _dedup_key field so downstream loaders can rely on it.
                if dedup_str:
                    item_data["_dedup_key"] = dedup_str
                else:
                    # Fallback: use _name, or the generated id as a last resort
                    fallback = item_data.get("_name") or item_data["id"]
                    dedup_str = str(fallback)
                    item_data["_dedup_key"] = dedup_str

                if dedup_str not in node_registry[node_type]:
                    nodes_dict[node_type].append(item_data)
                    node_registry[node_type].add(dedup_str)
                    # Register id for relationship lookups
                    id_registry[node_type][dedup_str] = item_data["id"]

    # Relationships
    for relation_info in relations:
        from_type = relation_info.from_node.__name__
        to_type = relation_info.to_node.__name__

        # Skip relationships involving node classes that are not configured
        from_node_info = next((n for n in nodes if n.node_class.__name__ == from_type), None)
        to_node_info = next((n for n in nodes if n.node_class.__name__ == to_type), None)

        if not from_node_info or not to_node_info:
            continue

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
            from_dict: Dict[str, Any]
            if isinstance(raw_from, dict):
                from_dict = raw_from
            else:
                # Fallback: best-effort conversion for unexpected types
                from_dict = dict(getattr(raw_from, "__dict__", {}))

            # Get dedup value for from node using the same helper as extraction
            from_dedup_value = from_node_info.get_dedup_value(from_dict, from_type)
            from_dedup_str = str(from_dedup_value) if from_dedup_value else None
            from_id = id_registry[from_type].get(from_dedup_str) if from_dedup_str else None

            if not from_id:
                continue  # Skip if we can't find the from node id

            to_items = to_data if isinstance(to_data, list) else [to_data]
            for to_item in to_items:
                if to_item is None:
                    continue
                raw_to = to_item.model_dump() if hasattr(to_item, "model_dump") else to_item
                to_dict: Dict[str, Any]
                if isinstance(raw_to, dict):
                    to_dict = raw_to
                else:
                    to_dict = dict(getattr(raw_to, "__dict__", {}))

                # Get dedup value for to node using the same helper as extraction
                to_dedup_value = to_node_info.get_dedup_value(to_dict, to_type)
                to_dedup_str = str(to_dedup_value) if to_dedup_value else None
                to_id = id_registry[to_type].get(to_dedup_str) if to_dedup_str else None

                if to_id:
                    # Extract p_*_ properties from to_item for edge properties
                    edge_properties = {}
                    if hasattr(relation_info.to_node, "model_fields"):
                        for field_name in relation_info.to_node.model_fields.keys():
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

    return nodes_dict, relationships


# Loading


def load_graph_data(
    backend: GraphBackend,
    nodes_dict: dict[str, list[dict[str, Any]]],
    relationships: list[RelationshipRecord] | list[tuple[Any, ...]],
) -> None:
    """Load nodes and relationships into the graph database using MERGE semantics.

    Uses MERGE statements to insert or update nodes, preserving _created_at timestamps
    and updating _updated_at on matches. Relationships are created using MATCH + CREATE.

    Args:
        conn: Kuzu database connection
        nodes_dict: Dictionary mapping node types to list of node data dicts
        relationships: List of relationship tuples
        nodes: List of GraphNode objects
    """

    # Merge nodes using MERGE statements (creates new or updates existing).
    # We now merge on the unified _dedup_key field so that deduplication
    # semantics are driven entirely by GraphNode.deduplication_key
    # (or name_from when that is not set).
    console.print("[bold]Merging nodes into graph...[/bold]")
    _merge_stats, id_mapping = merge_nodes_batch(
        conn=backend,
        nodes_dict=nodes_dict,
        merge_on_field="_dedup_key",
    )

    # Normalise relationships into RelationshipRecord instances so that
    # downstream code can rely on named attributes rather than tuple
    # indexing. This also preserves backwards compatibility with any
    # older callers that might still pass raw tuples.
    normalised_rels: list[RelationshipRecord] = []
    for rel in relationships:
        if isinstance(rel, RelationshipRecord):
            normalised_rels.append(rel)
            continue

        # Tuple fallback – support both legacy 5-field and 6-field formats
        if not isinstance(rel, tuple):
            continue

        if len(rel) == 6:
            from_type, from_id, to_type, to_id, rel_name, edge_properties = rel
        elif len(rel) == 5:
            from_type, from_id, to_type, to_id, rel_name = rel
            edge_properties = {}
        else:
            # Unsupported legacy shape
            continue

        normalised_rels.append(
            RelationshipRecord(
                from_type=str(from_type),
                from_id=str(from_id),
                to_type=str(to_type),
                to_id=str(to_id),
                name=str(rel_name),
                properties=dict(edge_properties or {}),
            )
        )

    # Relationships - use merged IDs for all node references
    # TODO: Implement relationship deduplication to avoid duplicate edges between same node pairs.
    # Currently, relationships are created even if they already exist. A future enhancement
    # could use MERGE for relationships as well, matching on (from_node, to_node, rel_type)
    # and optionally updating edge properties.

    console.print(f"[bold]Creating {len(normalised_rels)} relationships...[/bold]")
    edge_props_count = sum(1 for r in normalised_rels if r.properties)
    if edge_props_count > 0:
        console.print(f"[cyan]  {edge_props_count} relationships have properties[/cyan]")

    relationships_created = 0
    for rel in normalised_rels:
        # Translate original IDs to merged IDs using id_mapping
        merged_from_id = id_mapping.get((rel.from_type, rel.from_id), rel.from_id)
        merged_to_id = id_mapping.get((rel.to_type, rel.to_id), rel.to_id)

        # Ensure we have strings before calling replace (defensive)
        merged_from_id_str = "" if merged_from_id is None else str(merged_from_id)
        merged_to_id_str = "" if merged_to_id is None else str(merged_to_id)

        from_id_escaped = merged_from_id_str.replace("'", "\\'")
        to_id_escaped = merged_to_id_str.replace("'", "\\'")

        # Build properties string for edge
        props_str = ""
        edge_properties = rel.properties or {}
        if edge_properties:
            prop_parts = []
            for key, value in edge_properties.items():
                if value is None:
                    prop_parts.append(f"{key}: NULL")
                elif isinstance(value, str):
                    escaped = value.replace("'", "\\'")
                    prop_parts.append(f"{key}: '{escaped}'")
                elif isinstance(value, (int, float)):
                    prop_parts.append(f"{key}: {value}")
                else:
                    escaped = str(value).replace("'", "\\'")
                    prop_parts.append(f"{key}: '{escaped}'")
            if prop_parts:
                props_str = " {" + ", ".join(prop_parts) + "}"

        match_sql = f"""
        MATCH (from:{rel.from_type}), (to:{rel.to_type})
        WHERE from.id = '{from_id_escaped}'
          AND to.id = '{to_id_escaped}'
        CREATE (from)-[:{rel.name}{props_str}]->(to)
        """
        try:
            backend.execute(match_sql)
            relationships_created += 1
        except Exception as e:
            console.print(f"[red]Error creating {rel.name} relationship:[/red] {e}")
            console.print(f"[dim]SQL: {match_sql}[/dim]")

    console.print(f"[green]✓ Created {relationships_created} relationships[/green]")


# Orchestration
# Orchestration


def create_graph(
    backend: GraphBackend,
    model: BaseModel,
    schema_config: GraphSchema,
    source_key: str | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], list[RelationshipRecord]]:
    """Create a knowledge graph from a Pydantic model in the configured graph database.

    Args:
        backend: Graph database backend
        model: Root instance to convert
        schema_config: GraphSchema object with node and relationship configurations
        relations: Ignored (kept for compatibility)

    Returns:
        nodes_dict and relationships that were used to populate the graph
    """
    # Check if this is the new GraphSchema format
    if not hasattr(schema_config, "nodes") or not hasattr(schema_config, "relations"):
        raise ValueError("create_graph now only accepts GraphSchema objects. Please update your configuration.")

    schema = schema_config
    console.print("[green]Using GraphSchema format[/green]")

    # Print schema summary
    try:
        schema.print_schema_summary()
    except Exception:
        console.print(f"[yellow]Schema with {len(schema.nodes)} nodes and {len(schema.relations)} relations[/yellow]")

    console.print("[bold]Creating database schema...[/bold]")

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

    console.print("[cyan]Creating database tables...[/cyan]")
    create_schema(backend, schema.nodes, schema.relations)

    console.print("[cyan]Extracting and loading data...[/cyan]")
    nodes_dict, relationships = extract_graph_data(model, schema.nodes, schema.relations, source_key=source_key)

    load_graph_data(backend, nodes_dict, relationships)

    console.print("\n[bold green]Graph creation complete![/bold green]")
    total_nodes = sum(len(node_list) for node_list in nodes_dict.values())
    console.print(f"[green]Total nodes:[/green] {total_nodes}")
    console.print(f"[green]Total relationships:[/green] {len(relationships)}")

    return nodes_dict, relationships


def build_visualization_nodes(nodes_dict: dict[str, list[dict[str, Any]]]) -> list[tuple[str, dict[str, Any]]]:
    """Build a flat (id, properties) list suitable for HTML visualizers.

    The returned node IDs are taken from each record's ``id`` field when
    present; a synthetic value is generated as a fallback.
    """
    flattened: list[tuple[str, dict[str, Any]]] = []
    for node_type, items in nodes_dict.items():
        for item in items:
            node_id_val = item.get("id")
            node_id = str(node_id_val) if node_id_val is not None else f"{node_type}_{uuid.uuid4()}"
            flattened.append((node_id, item))
    return flattened


def build_visualization_links_from_relationships(
    relationships: Iterable[RelationshipRecord],
) -> list[tuple[str, str, str, dict[str, Any]]]:
    """Convert :class:`RelationshipRecord` instances into HTML link tuples.

    Returns a list of ``(from_id, to_id, name, properties)`` tuples that can
    be consumed by visualization helpers without knowing about
    :class:`RelationshipRecord` internals.
    """
    links: list[tuple[str, str, str, dict[str, Any]]] = []
    for rel in relationships:
        props = dict(rel.properties or {})
        links.append((rel.from_id, rel.to_id, rel.name, props))
    return links
