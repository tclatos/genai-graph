"""Generic graph node merging utilities for incremental knowledge graph construction.

This module provides utilities to merge nodes into a Kuzu graph database using pure
Cypher MERGE statements. It supports incremental addition of documents by merging
nodes based on a configurable key field (default: _name) and preserving creation
timestamps while updating modification timestamps.

The merging strategy follows these principles:
- Nodes are matched by a merge_on_field (default: _name)
- On first creation, all properties plus _created_at and _updated_at are set
- On subsequent matches, only _updated_at is refreshed
- No APOC dependencies - uses pure Cypher only
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from genai_tk.core.prompts import dedent_ws

from genai_graph.core.graph_backend import GraphBackend

if TYPE_CHECKING:
    pass
from loguru import logger
from rich.console import Console

console = Console()


def _should_update_value(value: Any) -> bool:
    """Return True when a value should overwrite an existing node property."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return True


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
    if value is None:
        return "NULL"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, str):
        # Escape single quotes for Cypher
        escaped = value.replace("'", "\\'")
        return f"'{escaped}'"
    elif isinstance(value, list):
        # Recursively format list elements
        formatted_items = [_format_value_for_cypher(item) for item in value]
        return f"[{', '.join(formatted_items)}]"
    elif isinstance(value, dict):
        # Map / struct literal: {key: value, ...}
        # Empty dicts cannot be represented in Cypher, use NULL instead
        if not value:
            return "NULL"
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


def build_merge_query(
    node_type: str,
    node_data: dict[str, Any],
    key_field: str = "id",
    merge_on_field: str = "_name",
) -> tuple[str, str]:
    """Build queries for upserting a node using MATCH + conditional CREATE.

    Since Kuzu doesn't support MERGE with ON CREATE/ON MATCH, we use two queries:
    1. Check if node exists (MATCH)
    2. Either UPDATE or CREATE based on existence

    Args:
        node_type: Label/type of the node
        node_data: Dictionary of node properties
        key_field: Primary key field name
        merge_on_field: Field to match on for merging

    Returns:
        Tuple of (check_query, upsert_query)
    """
    # Get the merge value
    merge_value = node_data.get(merge_on_field)
    if merge_value is None:
        raise ValueError(f"Node data missing required merge field '{merge_on_field}'")

    # Format merge value
    merge_value_formatted = _format_value_for_cypher(merge_value)

    # Query 1: Check if node exists and get its id plus current naming info
    check_query = dedent_ws(f"""
        MATCH (n:{node_type} {{{merge_on_field}: {merge_value_formatted}}})
        RETURN n.{key_field} as id, n._created_at as created_at, n._name as name, n.alternate_names as alternate_names
        LIMIT 1
        """)

    # Query 2a: Update existing node (timestamp only)
    # Query 2b: Create new node with all properties
    # We'll return a template that the caller will use based on check results

    # Metadata fields from the Pydantic model that are renamed with _ prefix in the schema
    # These should be excluded since they're already in node_data with the _ prefix
    # For example: "name" -> "_name", "created_at" -> "_created_at"
    excluded_metadata_fields = {"name", "created_at", "updated_at", "dedup_key"}

    # Build properties for CREATE
    create_props = []
    for key, value in node_data.items():
        # Skip metadata fields without _ prefix (they're duplicates)
        if key in excluded_metadata_fields:
            continue
        # Generic handling for dicts / struct-like values is sufficient
        # We format dicts as STRUCT literals. Empty dicts are mapped to NULL.
        formatted_value = _format_value_for_cypher(value)
        create_props.append(f"{key}: {formatted_value}")

    props_str = ", ".join(create_props)

    # Return both check query and upsert info
    # The caller will decide which operation to perform
    return check_query.strip(), props_str


def merge_node_in_graph(
    conn: GraphBackend,
    node_type: str,
    node_data: dict[str, Any],
    merge_on_field: str = "_name",
) -> tuple[bool, str]:
    """Merge a single node into the graph database.

    Executes a check-then-upsert operation: first checks if node exists,
    then either updates timestamp or creates new node.

    Args:
        conn: Graph database connection (kuzu.Connection or similar)
        node_type: Node label/type
        node_data: Node properties dictionary
        schema_config: Optional schema configuration
        merge_on_field: Field to match nodes on

    Returns:
        Tuple of (was_created: bool, node_id: str)
    """
    try:
        timestamp = datetime.utcnow().isoformat() + "Z"
        # merge_value = node_data.get(merge_on_field)
        # merge_value_formatted = _format_value_for_cypher(merge_value)

        # Build queries
        check_query, props_str = build_merge_query(
            node_type=node_type,
            node_data=node_data,
            key_field="id",
            merge_on_field=merge_on_field,
        )

        # Step 1: Check if node exists
        result = conn.execute(check_query)
        df = result.get_as_df()

        if not df.empty:
            # Node exists - update timestamp and optionally maintain alternate names
            row = df.iloc[0]
            existing_id = str(row["id"])
            existing_name = row.get("name")
            existing_alternates = row.get("alternate_names")

            new_name = node_data.get("_name")
            updated_alternates = None

            # Only track alternate names when the new name is different from the
            # canonical one and not already present in the alternates list.
            if new_name and new_name != existing_name:
                current_list: list[str] = []
                if isinstance(existing_alternates, list):
                    current_list = [str(v) for v in existing_alternates if v is not None]
                elif isinstance(existing_alternates, str) and existing_alternates:
                    # If the backend already stored a single string value, normalise
                    # it into a one-element list. Ignore non-string NaN/None values.
                    current_list = [existing_alternates]

                if new_name not in current_list:
                    current_list.append(new_name)
                    updated_alternates = current_list

            # Build SET clause dynamically. On matches, update non-empty
            # properties from the incoming node_data so later sources (e.g. DB
            # pulls) can take precedence over earlier ones.
            set_clauses = [f"n._updated_at = '{timestamp}'"]

            if updated_alternates is not None:
                alternates_formatted = _format_value_for_cypher(updated_alternates)
                set_clauses.append(f"n.alternate_names = {alternates_formatted}")

            excluded_update_fields = {
                "id",
                "name",
                "created_at",
                "updated_at",
                "dedup_key",
                "_created_at",
                "_updated_at",
                "_name",
                "_dedup_key",
                "alternate_names",
            }

            for key, value in node_data.items():
                if key in excluded_update_fields:
                    continue
                if not _should_update_value(value):
                    continue
                formatted = _format_value_for_cypher(value)
                set_clauses.append(f"n.{key} = {formatted}")

            set_sql = ", ".join(set_clauses)
            update_query = dedent_ws(f"""
                MATCH (n:{node_type})
                WHERE n.id = '{existing_id.replace("'", "\\'")}'
                SET {set_sql}
                RETURN n.id as id
                """)
            conn.execute(update_query)
            return False, existing_id
        else:
            # Node doesn't exist - create it
            create_query = f"CREATE (n:{node_type} {{{props_str}}}) RETURN n.id as id"
            result = conn.execute(create_query)
            df = result.get_as_df()

            if df.empty:
                logger.warning(f"CREATE returned no ID for {node_type}")
                return True, ""

            node_id = str(df.iloc[0]["id"])
            return True, node_id

    except Exception as e:
        import traceback as tb

        logger.error(f"Error merging {node_type} node: {e}")
        logger.error(f"Node data: {node_data.get(merge_on_field, 'unknown')}")
        logger.error(tb.format_exc())
        raise


def merge_nodes_batch(
    conn: GraphBackend,
    nodes_dict: dict[str, list[dict[str, Any]]],
    merge_on_field: str = "_name",
) -> tuple[dict[str, dict[str, int]], dict[tuple[str, str], str]]:
    """Merge multiple nodes into the graph database in batch.

    Processes all nodes by type, tracking statistics and building an ID mapping
    for relationship creation.

    Args:
        conn: Graph database connection (kuzu.Connection or similar)
        nodes_dict: Mapping of node_type to list of node data dicts
        schema_config: Optional schema configuration
        merge_on_field: Field to match nodes on

    Returns:
        Tuple of:
        - Statistics dict: {node_type: {created, matched, total}}
        - ID mapping: {(node_type, original_id): merged_global_id}
    """
    stats: dict[str, dict[str, int]] = {}
    id_mapping: dict[tuple[str, str], str] = {}

    for node_type, node_list in nodes_dict.items():
        if not node_list:
            continue

        logger.debug(f"Merging {len(node_list)} {node_type} nodes...")

        type_stats = {"created": 0, "matched": 0, "total": len(node_list)}

        for node_data in node_list:
            # Get original ID before merge
            original_id = node_data.get("id", "")

            # Merge the node
            was_created, merged_id = merge_node_in_graph(
                conn=conn,
                node_type=node_type,
                node_data=node_data,
                merge_on_field=merge_on_field,
            )

            # Update statistics
            if was_created:
                type_stats["created"] += 1
            else:
                type_stats["matched"] += 1

            # Register ID mapping for relationship creation
            if original_id and merged_id:
                id_mapping[(node_type, original_id)] = merged_id

        stats[node_type] = type_stats
        logger.debug(f"  {node_type}: {type_stats['created']} created, {type_stats['matched']} matched")

    return stats, id_mapping
