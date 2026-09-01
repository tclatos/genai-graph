"""Knowledge Graph construction and querying package.

This package provides a unified API for:
- Defining graph schemas (kg.schema)
- Loading data from various sources (kg.factories)
- Ingesting data into the graph (kg.ingest)
- Exporting graph artifacts (kg.export)
- Querying the graph (kg.query)

The package uses consistent "kg" naming throughout:
- KgManager: Central configuration and artifact management
- KgBackend: Graph database backend abstraction
- KgFactory: Base class for data source factories

Example usage:
    from genai_graph.kg import KgManager, get_kg_manager, KgBackend
    from genai_graph.kg.schema import GraphSchema, GraphNode, GraphRelation
    from genai_graph.kg.factories import JsonFileBackedFactory, TableBackedFactory
"""

from genai_graph.kg.backend import (
    KgBackend,
    KuzuBackend,
    LadybugBackend,
    create_backend,
    create_backend_from_config,
    create_in_memory_backend,
)
from genai_graph.kg.manager import KgManager, get_kg_manager
from genai_graph.kg.parallel import SharedKuzuParallel, SharedLadybugParallel

__all__ = [
    "KgManager",
    "get_kg_manager",
    "KgBackend",
    "KuzuBackend",
    "LadybugBackend",
    "create_backend",
    "create_backend_from_config",
    "create_in_memory_backend",
    "SharedKuzuParallel",
    "SharedLadybugParallel",
]
