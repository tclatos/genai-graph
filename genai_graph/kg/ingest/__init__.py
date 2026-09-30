"""Data ingestion for Knowledge Graph construction.

This package provides:
- Graph data extraction from Pydantic models
- Node/relationship merging into the graph database
- Document ingestion helpers
- Data lineage tracking
"""

from genai_graph.kg.ingest.documents import (
    DocumentStats,
    add_documents_to_graph,
    add_neo4j_data_to_graph,
)
from genai_graph.kg.ingest.extract import (
    NodeRecord,
    RelationshipRecord,
    _get_kuzu_type,
    create_graph,
    create_schema,
    extract_graph_data,
    restart_database,
)
from genai_graph.kg.ingest.merge import (
    MergeStats,
    NodeDataCollection,
    NodeTypeRegistry,
    ParquetCollector,
    get_parquet_collector,
    merge_nodes_batch,
    merge_relationships_batch,
    set_parquet_collector,
)

__all__ = [
    # Documents
    "DocumentStats",
    "MergeStats",
    # Merge
    "NodeDataCollection",
    # Extract
    "NodeRecord",
    "NodeTypeRegistry",
    "ParquetCollector",
    "RelationshipRecord",
    "_get_kuzu_type",
    "add_documents_to_graph",
    "add_neo4j_data_to_graph",
    "create_graph",
    "create_schema",
    "extract_graph_data",
    "get_parquet_collector",
    "merge_nodes_batch",
    "merge_relationships_batch",
    "restart_database",
    "set_parquet_collector",
]
