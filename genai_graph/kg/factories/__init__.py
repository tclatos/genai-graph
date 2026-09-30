"""Data source factories for Knowledge Graph construction.

This package provides factories that load data from various sources:
- KgFactory: Abstract base class for all factories
- JsonFileBackedFactory: Load data from JSON files
- TableBackedFactory: Load data from SQL database tables
- Neo4jFactory: Load data from Neo4j JSONL exports
- Neo4jImportFactory: Extended factory with schema-based mappings
- Neo4jNodeMapping: Configuration for node type mappings
- Neo4jRelationMapping: Configuration for relationship type mappings
- SimilarityFactory: Abstract base for embedding-similarity relationship factories
- SimilaritySpec: Configuration for one similarity field pair (from/to/relationship/iterate_over)
- SimilarityResult: Statistics from a similarity computation run
"""

from genai_graph.kg.factories.base import KgFactory
from genai_graph.kg.factories.document_factory import DocumentDirectoryFactory
from genai_graph.kg.factories.document_graph_factory import DocumentGraphBundle, DocumentGraphFactory
from genai_graph.kg.factories.document_mixin import DocumentMixin
from genai_graph.kg.factories.json_factory import JsonFileBackedFactory
from genai_graph.kg.factories.markdown_baml_factory import MarkdownBamlFactory
from genai_graph.kg.factories.neo4j_factory import (
    Neo4jFactory,
    Neo4jImportFactory,
    Neo4jNodeMapping,
    Neo4jRelationMapping,
)
from genai_graph.kg.factories.similarity import (
    SimilarityFactory,
    SimilarityResult,
    SimilaritySpec,
)
from genai_graph.kg.factories.table_factory import TableBackedFactory

__all__ = [
    "DocumentDirectoryFactory",
    "DocumentGraphBundle",
    "DocumentGraphFactory",
    "DocumentMixin",
    "JsonFileBackedFactory",
    "KgFactory",
    "MarkdownBamlFactory",
    "Neo4jFactory",
    "Neo4jImportFactory",
    "Neo4jNodeMapping",
    "Neo4jRelationMapping",
    "SimilarityFactory",
    "SimilarityResult",
    "SimilaritySpec",
    "TableBackedFactory",
]
