from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from genai_graph.kg.schema import GraphNode


def apply_extra_fields(
    item_data: dict[str, Any], node_info: GraphNode, model: BaseModel, item: Any, source_key: str | None
) -> None:
    """No-op placeholder kept for call-site compatibility.

    Provenance is now tracked via Document graph nodes and CONTAINS relationships
    created by ``create_document_nodes_task``.  The ``metadata`` dict field and
    ``FileMetadata`` class have been removed from all node models.
    """
