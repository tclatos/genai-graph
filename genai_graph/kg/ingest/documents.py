"""Document ingestion helpers.

This module provides helpers that the CLI and Prefect flow use to add
structured data from various sources into the knowledge graph.

For JSON file-backed sources:
- Use add_documents_to_graph() with a JsonFileBackedFactory
- Document provenance is tracked via Document graph nodes (see DocumentMixin)
  and CONTAINS relationships, created by create_document_nodes_task.

For Neo4j imports:
- Use add_neo4j_data_to_graph() with a Neo4jImportFactory
- This bypasses hierarchical extraction and directly loads pre-built nodes/relationships
"""

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel

from genai_graph.kg.backend import KgBackend
from genai_graph.kg.factories.base import KgFactory
from genai_graph.kg.manager import KgManager
from genai_graph.kg.schema.core import GraphSchema

if TYPE_CHECKING:
    from genai_graph.kg.factories.neo4j_factory import Neo4jImportFactory


class DocumentStats(BaseModel):
    """Statistics from document processing."""

    total_processed: int = 0
    total_failed: int = 0
    nodes_created: int = 0
    relationships_created: int = 0


def add_neo4j_data_to_graph(
    graph_impl: "Neo4jImportFactory",
    backend: KgBackend,
    context: KgManager | None = None,
) -> DocumentStats:
    """Add Neo4j data directly to the knowledge graph.

    This is a specialized path for Neo4j imports that bypasses the hierarchical
    model extraction. The factory provides pre-built nodes and relationships
    which are loaded directly.

    Args:
        graph_impl: Neo4jImportFactory instance with build_nodes_and_relationships()
        backend: KgBackend instance
        context: Optional KgManager for collecting warnings

    Returns:
        DocumentStats instance summarising processing results
    """
    from genai_graph.kg.ingest.extract import import_neo4j_data

    stats = DocumentStats()

    try:
        logger.info(f"Building nodes and relationships from {graph_impl.name}")

        # Get pre-built nodes and relationships from the factory
        nodes_data, relationships = graph_impl.build_nodes_and_relationships()

        logger.info(f"Loaded {nodes_data.total_count()} nodes and {len(relationships)} relationships")

        if nodes_data.total_count() == 0:
            logger.warning("No nodes to import")
            return stats

        # Use the direct import function
        # Pass key_field mapping from the factory so import uses correct PKs
        key_fields: dict[str, str] = {}
        if hasattr(graph_impl, "get_node_mappings"):
            for mapping in graph_impl.get_node_mappings():
                key_fields[mapping.target_label] = mapping.key_field

        nodes_data, relationships = import_neo4j_data(
            backend=backend,
            nodes_data=nodes_data,
            relationships=relationships,
            context=context,
            key_fields=key_fields or None,
        )

        stats.nodes_created = nodes_data.total_count()
        stats.relationships_created = len(relationships)
        stats.total_processed = 1

    except Exception as e:
        logger.error(f"Failed to import Neo4j data: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        stats.total_failed += 1

    return stats


def add_documents_to_graph(
    keys: list[str],
    graph_impl: KgFactory,
    backend: KgBackend,
    schema: GraphSchema,
    context: KgManager | None = None,
) -> DocumentStats:
    """Add one or more documents to the knowledge graph.

    Args:
        keys: List of keys/file paths to load via the subgraph implementation.
              For JsonFileBackedFactory, these are file paths.
              For other factories, these are abstract keys.
        graph_impl: Subgraph factory providing data loading methods
        backend: KgBackend instance
        schema: GraphSchema instance
        context: Optional KgManager for collecting warnings

    Returns:
        DocumentStats instance summarising processing results
    """
    from genai_graph.kg.ingest.extract import create_graph

    stats = DocumentStats()

    root_class = getattr(schema, "root_model_class", None)
    if root_class is None:
        raise ValueError(
            f"Schema for subgraph '{graph_impl.name}' does not have root_model_class set. "
            "Document processing requires a root model class."
        )

    for key in keys:
        try:
            logger.debug(f"Loading key {key} for subgraph {graph_impl.name}")
            data = graph_impl.get_struct_data_by_key(key)
            logger.debug(f"Loaded? {bool(data)}")
            if not data:
                stats.total_failed += 1
                continue

            # For file-based sources, use relative path as source_key for cleaner provenance
            from genai_graph.kg.factories import JsonFileBackedFactory

            if isinstance(graph_impl, JsonFileBackedFactory):
                # Extract relative path from full file path for cleaner source tracking
                from genai_tk.config_mgmt.file_patterns import resolve_config_path

                file_path = Path(key)
                try:
                    data_root = resolve_config_path(graph_impl.data_root)
                    root_path = Path(data_root)
                    source_key = str(file_path.relative_to(root_path))
                except (ValueError, AttributeError):
                    # Fallback to filename if relative path fails
                    source_key = file_path.name
            else:
                source_key = key

            # create_graph will attach source_key into the extracted root nodes
            nodes_data, relationships = create_graph(backend, data, schema, source_key=source_key, context=context)

            nodes_created = nodes_data.total_count() if nodes_data else 0
            rels_created = len(relationships) if relationships is not None else 0

            stats.nodes_created += nodes_created
            stats.relationships_created += rels_created
            stats.total_processed += 1

        except Exception as e:
            import traceback

            error_msg = str(e)
            hint = ""
            # Provide helpful context based on error type
            if "Key field" in error_msg and "not found or empty" in error_msg:
                hint = " 💡 Consider key_from='AUTO_ID' if this field may be absent."
            elif "Cannot find property" in error_msg:
                hint = " 💡 Schema mismatch — field in data not in node definition."

            full_msg = f"Failed to process key {key}: {error_msg}{hint}"
            logger.error(full_msg)
            # Always log the full traceback at ERROR so it's visible without debug logging
            logger.error("Full traceback:\n" + traceback.format_exc())

            # Propagate to KgManager so failures appear in the warnings report
            if context:
                context.add_warning(full_msg)

            stats.total_failed += 1
            continue

    return stats
