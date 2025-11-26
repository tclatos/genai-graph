"""Document ingestion helpers.

This module provides a small wrapper that the CLI can call to add one or
more documents (keys) to the graph. The previous implementation relied on
separate Document nodes + SOURCE relationships. We now attach provenance
into the root model's `metadata` map field (key: ``source``).

Behavior:
- Validate that the subgraph root model exposes a `metadata` field whose
  annotation is either ``dict`` or ``Optional[dict]``.
- For each key, load the pydantic model using the provided subgraph
  implementation and call ``create_graph(..., source_key=key)`` which will
  populate the created root node(s) `metadata.source` value. If the
  extracted model instance has `metadata` == None it is replaced with a
  dict by the extraction/creation code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List


@dataclass
class DocumentStats:
    total_processed: int = 0
    total_failed: int = 0
    nodes_created: int = 0
    relationships_created: int = 0


def _has_metadata_map(root_class: Any) -> bool:
    """Return True if root_class defines a `metadata` model field typed as dict or Optional[dict]."""
    try:
        from typing import get_args, get_origin

        if not hasattr(root_class, "model_fields"):
            return False
        if "metadata" not in root_class.model_fields:
            return False
        ann = root_class.model_fields["metadata"].annotation
        # Direct dict
        if ann is dict:
            return True
        origin = get_origin(ann)
        if origin is dict:
            return True
        # Optional / Union[...] handling
        if origin is None and hasattr(ann, "__args__"):
            origin = get_origin(ann)
        if origin is None:
            return False
        # If origin is Union, check args
        if origin.__name__ == "Union" or origin is tuple:
            for a in get_args(ann):
                if a is dict:
                    return True
                if get_origin(a) is dict:
                    return True
        return False
    except Exception:
        return False


def add_documents_to_graph(keys: List[str], subgraph_impl: Any, backend: Any, schema: Any) -> DocumentStats:
    """Add one or more documents to the knowledge graph.

    Args:
        keys: list of keys to load via the subgraph implementation
        subgraph_impl: subgraph module providing `load_data` and `get_entity_name_from_data`
        backend: GraphBackend instance
        schema: GraphSchema instance

    Returns:
        DocumentStats instance summarising processing results
    """
    from genai_graph.core.graph_core import create_graph

    stats = DocumentStats()

    root_class = getattr(schema, "root_model_class", None)
    if root_class is None:
        raise ValueError("schema does not expose root_model_class")

    # Validate presence of metadata map field (allow Optional[dict])
    if not _has_metadata_map(root_class):
        raise ValueError(
            f"Subgraph root model '{root_class.__name__}' must expose a 'metadata' map field (dict or Optional[dict])"
        )

    for key in keys:
        try:
            data = subgraph_impl.load_data(key)
            if not data:
                stats.total_failed += 1
                continue

            # create_graph will attach source_key into the extracted root nodes
            nodes_dict, relationships = create_graph(backend, data, schema, source_key=key)

            nodes_created = sum(len(v) for v in nodes_dict.values()) if nodes_dict else 0
            rels_created = len(relationships) if relationships is not None else 0

            stats.nodes_created += nodes_created
            stats.relationships_created += rels_created
            stats.total_processed += 1

        except Exception:
            stats.total_failed += 1
            continue

    return stats
