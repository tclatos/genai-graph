"""Document management for the Knowledge Graph.

Handles the logic for adding documents to the EKG database, including:
- Loading data from subgraph implementations
- Creating graph nodes and relationships
- Managing Document nodes and SOURCE relationships
- Tracking ingestion statistics
"""

from typing import Any

REL_TO_DOCUMENT_NODE = "SOURCE"


class DocumentStats:
    """Statistics for document ingestion operations."""

    def __init__(self) -> None:
        self.total_processed = 0
        self.total_failed = 0
        self.nodes_created = 0
        self.relationships_created = 0

    def to_dict(self) -> dict[str, int]:
        """Convert stats to dictionary."""
        return {
            "total_processed": self.total_processed,
            "total_failed": self.total_failed,
            "nodes_created": self.nodes_created,
            "relationships_created": self.relationships_created,
        }


def get_document_relationship_name() -> str:
    """Return the relationship name for linking to Document nodes."""
    return REL_TO_DOCUMENT_NODE


def ensure_document_tables(backend: Any) -> None:
    """Ensure the Document node table exists in the database.

    Args:
        backend: GraphBackend instance for database operations

    Raises:
        Exception: If table creation fails (though defensive errors are logged)
    """
    backend.create_node_table(
        table_name="Document",
        fields={
            "uuid": "STRING",
            # Arbitrary key/value metadata attached to the document.
            # We currently serialise it as a JSON string. Initially
            # this is the empty map "{}".
            "metadata": "STRING",
        },
        primary_key="uuid",
    )


def ensure_source_relationship_table(backend: Any, root_type: str) -> None:
    """Ensure the SOURCE relationship table exists.

    Args:
        backend: GraphBackend instance
        root_type: The root node type this SOURCE relationship originates from

    Raises:
        Exception: If relationship table creation fails
    """
    backend.create_relationship_table(
        rel_name=get_document_relationship_name,
        from_table=root_type,
        to_table="Document",
        properties=None,
    )


def create_or_ensure_document_node(backend: Any, key: str) -> bool:
    """Create or ensure a Document node exists for the given key.

    Args:
        backend: GraphBackend instance
        key: The unique document key/uuid

    Returns:
        True if the document node was created, False if it already existed

    Raises:
        Exception: If database operations fail
    """
    escaped_key = key.replace("'", "\\'")
    check_result = backend.execute(f"MATCH (d:Document {{uuid: '{escaped_key}'}}) RETURN count(d) as count")
    check_df = check_result.get_as_df()
    doc_exists = False

    if not check_df.empty:
        try:
            doc_exists = int(check_df.iloc[0]["count"]) > 0
        except Exception:  # pragma: no cover - defensive
            doc_exists = False

    if not doc_exists:
        # Store metadata as a JSON-encoded string; we
        # initialise it to the empty map "{}".
        backend.execute(f"CREATE (d:Document {{uuid: '{escaped_key}', metadata: '{{}}'}})")
        return True

    return False


def create_source_relationship(backend: Any, root_type: str, root_node: dict[str, Any], key: str) -> bool:
    """Create a SOURCE relationship from root node to Document node.

    Args:
        backend: GraphBackend instance
        root_type: The type of the root node
        root_node: The root node dictionary containing _dedup_key or _name
        key: The document key/uuid

    Returns:
        True if the relationship was created, False if root node or dedup value not found

    Raises:
        Exception: If database operations fail
    """
    if root_node is None:
        return False

    dedup_value = root_node.get("_dedup_key") or root_node.get("_name")
    if dedup_value is None:
        return False

    escaped_key = key.replace("'", "\\'")
    dedup_str = str(dedup_value).replace("'", "\\'")
    query = (
        f"MATCH (root:{root_type} {{_dedup_key: '{dedup_str}'}}), "
        f"(doc:Document {{uuid: '{escaped_key}'}}) "
        "CREATE (root)-[:SOURCE]->(doc)"
    )
    backend.execute(query)
    return True


def process_single_document(
    backend: Any,
    subgraph_impl: Any,
    key: str,
    schema: Any,
    root_type: str,
) -> tuple[dict[str, int], list[tuple[str, Any, str, Any, str]]]:
    """Process a single document: load data, create graph, link to Document node.

    Args:
        backend: GraphBackend instance
        subgraph_impl: Subgraph implementation with load_data and get_entity_name_from_data methods
        key: Document key to load
        schema: Graph schema for creating nodes
        root_type: Root node type for SOURCE relationships

    Returns:
        Tuple of (nodes_dict, relationships_list)

    Raises:
        Exception: If document loading or graph creation fails
    """
    from genai_graph.core.graph_core import create_graph

    # Load data
    data = subgraph_impl.load_data(key)
    if not data:
        raise ValueError(f"No data found for key: {key}")

    # Create graph nodes and relationships
    nodes_dict, relationships = create_graph(backend, data, schema)

    # Create/attach a generic Document node for this ingestion key
    try:
        # 1) Ensure a Document node exists for the key (idempotent)
        create_or_ensure_document_node(backend, key)

        # 2) Link the top-level entity node to the Document via SOURCE
        root_nodes = nodes_dict.get(root_type, [])
        root_node = root_nodes[0] if root_nodes else None

        if root_node is not None:
            if create_source_relationship(backend, root_type, root_node, key):
                # Count this extra relationship
                relationships.append((root_type, None, "Document", None, get_document_relationship_name))
    except Exception:
        # Defensive: don't break ingestion if Document linking fails
        pass

    return nodes_dict, relationships


def add_documents_to_graph(
    keys: list[str],
    subgraph_impl: Any,
    backend: Any,
    schema: Any,
) -> DocumentStats:
    """Add one or more documents to the knowledge graph.

    Handles the complete workflow of processing multiple documents:
    - Ensures required tables exist
    - Loads data for each document
    - Creates graph nodes and relationships
    - Links documents via SOURCE relationships
    - Accumulates statistics

    Args:
        keys: List of document keys to add
        subgraph_impl: Subgraph implementation
        backend: GraphBackend instance
        schema: Graph schema

    Returns:
        DocumentStats object with processing results
    """
    stats = DocumentStats()

    # Determine the logical root node type for this subgraph
    root_type = schema.root_model_class.__name__

    # Ensure the generic Document node table exists
    try:
        ensure_document_tables(backend)
    except Exception:  # pragma: no cover - defensive, backend should support this
        pass

    # Ensure SOURCE relationship table exists
    try:
        ensure_source_relationship_table(backend, root_type)
    except Exception:  # pragma: no cover - defensive
        pass

    # Process each key sequentially
    for key in keys:
        try:
            # Process document
            nodes_dict, relationships = process_single_document(
                backend,
                subgraph_impl,
                key,
                schema,
                root_type,
            )

            # Accumulate statistics
            doc_nodes = sum(len(node_list) for node_list in nodes_dict.values())
            doc_rels = len(relationships)

            stats.nodes_created += doc_nodes
            stats.relationships_created += doc_rels
            stats.total_processed += 1

        except Exception:
            # Document processing failed, continue with next
            stats.total_failed += 1
            continue

    return stats
