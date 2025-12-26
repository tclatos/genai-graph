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

from dataclasses import dataclass
from typing import List, Type

from loguru import logger
from pydantic import BaseModel

from genai_graph.core.graph_backend import GraphBackend
from genai_graph.core.graph_schema import GraphSchema
from genai_graph.core.kg_manager import KgManager
from genai_graph.core.subgraph_factories import SubgraphFactory


@dataclass
class DocumentStats:
    total_processed: int = 0
    total_failed: int = 0
    nodes_created: int = 0
    relationships_created: int = 0


def _has_metadata_map(root_class: Type[BaseModel], schema: GraphSchema) -> bool:
    """Return True if root_class defines a `metadata` model field typed as dict or Optional[dict].

    Also return True if the schema config for the root class defines an
    ExtraFields class named `FileMetadata` (compatibility for new behavior).
    """
    try:
        from typing import get_args, get_origin

        if not hasattr(root_class, "model_fields"):
            # If model_fields missing, fall back to schema inspection
            root_node = None
            if hasattr(schema, "nodes"):
                for n in getattr(schema, "nodes", []):
                    if getattr(n, "node_class", None) is root_class:
                        root_node = n
                        break
            if root_node is not None:
                extras = getattr(root_node, "extra_classes", []) or []
                return any(getattr(ec, "__name__", "") == "FileMetadata" for ec in extras)
            return False
        if "metadata" not in root_class.model_fields:
            # Check schema node extra classes as a fallback
            root_node = None
            if hasattr(schema, "nodes"):
                for n in getattr(schema, "nodes", []):
                    if getattr(n, "node_class", None) is root_class:
                        root_node = n
                        break
            if root_node is not None:
                extras = getattr(root_node, "extra_classes", []) or []
                return any(getattr(ec, "__name__", "") == "FileMetadata" for ec in extras)
            return False
        ann = root_class.model_fields["metadata"].annotation
        # Direct dict
        if ann is dict:
            return True
        origin = get_origin(ann)
        if origin is dict:
            return True
        # Optional / Union[...] handling (Python 3.9 style or 3.12+ UnionType)
        if origin is None and hasattr(ann, "__args__"):
            origin = get_origin(ann)
        if origin is None:
            return False
        # Check for Union (typing.Union) or UnionType (Python 3.12+ dict | None)
        origin_name = getattr(origin, "__name__", "")
        if origin_name in ("Union", "UnionType") or origin is tuple:
            for a in get_args(ann):
                if a is dict:
                    return True
                if get_origin(a) is dict:
                    return True
        return False
    except Exception:
        return False


def _parse_pull_merge_on(merge_on: str) -> tuple[str, str] | None:
    """Parse a pull.merge_on spec of the form "NodeType.field_name"."""
    parts = merge_on.split(".", 1)
    if len(parts) != 2:
        return None
    node_type, field_name = parts[0].strip(), parts[1].strip()
    if not node_type or not field_name:
        return None
    return node_type, field_name


def _apply_pull_subgraphs(
    backend: GraphBackend,
    nodes_dict: dict[str, list[dict[str, object]]],
    pulled: set[tuple[str, str]],
    source_key: str,
    context: KgManager | None = None,
) -> None:
    """Pull DB-backed subgraphs on-demand based on configured triggers.

    Args:
        backend: GraphBackend instance
        nodes_dict: Dictionary of nodes by type
        pulled: Set of already pulled (subgraph_name, value) tuples
        source_key: Source key for provenance
        context: Optional KgContext for collecting warnings
    """
    from genai_graph.core.graph_core import create_graph
    from genai_graph.core.graph_registry import GraphRegistry
    from genai_graph.core.subgraph_factories import TableBackedSubgraphFactory

    registry = GraphRegistry.get_instance()

    # Map node types by case-insensitive key for tolerant matching.
    node_type_map = {k.lower(): k for k in nodes_dict.keys()}

    for subgraph in registry.subgraphs.values():
        if not isinstance(subgraph, TableBackedSubgraphFactory):
            continue
        if not subgraph.pull:
            continue

        parsed = _parse_pull_merge_on(subgraph.pull.merge_on)
        if not parsed:
            continue

        merge_node_type, merge_field = parsed
        actual_node_type = node_type_map.get(merge_node_type.lower())
        if not actual_node_type:
            continue

        for node_data in nodes_dict.get(actual_node_type, []):
            raw_value = node_data.get(merge_field) if isinstance(node_data, dict) else None
            value = str(raw_value).strip() if raw_value is not None else ""
            if not value:
                continue

            cache_key = (subgraph.name, value)
            if cache_key in pulled:
                continue

            try:
                db_model = subgraph.get_struct_data_by_field(subgraph.pull.db_field, value)
            except Exception:
                pulled.add(cache_key)
                continue

            if not db_model:
                pulled.add(cache_key)
                continue

            db_schema = subgraph.build_schema()
            create_graph(
                backend=backend,
                model=db_model,
                schema_config=db_schema,
                source_key=f"pull:{subgraph.name}:{source_key}",
                context=context,
            )
            pulled.add(cache_key)


def add_documents_to_graph(
    keys: List[str],
    subgraph_impl: SubgraphFactory,
    backend: GraphBackend,
    schema: GraphSchema,
    context: KgManager | None = None,
) -> DocumentStats:
    """Add one or more documents to the knowledge graph.

    Args:
        keys: list of keys to load via the subgraph implementation
        subgraph_impl: subgraph module providing `load_data`
        backend: GraphBackend instance
        schema: GraphSchema instance
        context: Optional KgContext for collecting warnings

    Returns:
        DocumentStats instance summarising processing results
    """
    from genai_graph.core.graph_core import create_graph

    stats = DocumentStats()

    root_class = getattr(schema, "root_model_class", None)
    if root_class is None:
        raise ValueError("schema does not expose root_model_class")

    # Validate presence of metadata map field (allow Optional[dict])
    if not _has_metadata_map(root_class, schema):
        msg = f"Subgraph root model '{root_class.__name__}' must expose a 'metadata' map field (dict or Optional[dict])"
        if context:
            context.add_warning(msg)
        raise ValueError(msg)

    pulled: set[tuple[str, str]] = set()

    for key in keys:
        try:
            logger.debug(f"Loading key {key} for subgraph {subgraph_impl.name}")
            data = subgraph_impl.get_struct_data_by_key(key)
            logger.debug(f"Loaded? {bool(data)}")
            if not data:
                stats.total_failed += 1
                continue

            # create_graph will attach source_key into the extracted root nodes
            nodes_dict, relationships = create_graph(backend, data, schema, source_key=key, context=context)

            # On-demand pull: DB-backed subgraphs can enrich the graph once a
            # matching node exists (avoids orphan nodes from bulk DB loads).
            try:
                _apply_pull_subgraphs(backend, nodes_dict, pulled, source_key=key, context=context)
            except Exception:
                pass

            nodes_created = sum(len(v) for v in nodes_dict.values()) if nodes_dict else 0
            rels_created = len(relationships) if relationships is not None else 0

            stats.nodes_created += nodes_created
            stats.relationships_created += rels_created
            stats.total_processed += 1

        except Exception as e:
            logger.error(f"Failed to process key {key}: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            stats.total_failed += 1
            continue

    return stats
