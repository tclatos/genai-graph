"""Schema definitions for Knowledge Graph construction.

This package provides:
- GraphNode: Node configuration for graph extraction
- GraphRelation: Relationship configuration
- GraphSchema: Complete schema definition
- GraphRegistry: Registry for graph factories
- generate_schema_description: LLM-friendly schema documentation
- ResolvedSchema: Canonical enriched schema with all render methods
- compiler: Standalone compilation functions (build_model_field_map, etc.)
"""

from genai_graph.kg.schema._helpers import _get_kuzu_type_for_field
from genai_graph.kg.schema.compiler import (
    build_model_field_map,
    compile_schema,
    compute_excluded_fields,
    deduce_node_field_paths,
    deduce_relation_field_paths,
    validate_schema_coherence,
)
from genai_graph.kg.schema.core import (
    GraphNode,
    GraphRelation,
    GraphSchema,
    _find_embedded_field_for_class,
    find_embedded_field_for_class,
)
from genai_graph.kg.schema.doc_generator import (
    generate_schema_description,
)
from genai_graph.kg.schema.registry import (
    GraphRegistry,
    get_graph,
    get_graph_registry,
    register_graph,
)
from genai_graph.kg.schema.resolved import (
    ResolvedSchema,
    VectorIndexInfo,
)

__all__ = [
    "GraphNode",
    "GraphRegistry",
    "GraphRelation",
    "GraphSchema",
    "ResolvedSchema",
    "VectorIndexInfo",
    "_find_embedded_field_for_class",
    "_get_kuzu_type_for_field",
    # Compiler
    "build_model_field_map",
    "compile_schema",
    "compute_excluded_fields",
    "deduce_node_field_paths",
    "deduce_relation_field_paths",
    "find_embedded_field_for_class",
    "generate_schema_description",
    "get_graph",
    "get_graph_registry",
    "register_graph",
    "validate_schema_coherence",
]
