from typing import Self

from pydantic import Field

from genai_graph.core.graph_schema import (
    ExtraFields,
    GraphNode,
)
from genai_graph.ekg.baml_client.types import Customer, Opportunity, Person


class FileMetadata(ExtraFields):
    source: str = Field(..., description="Source of the file from which the data was extracted")

    @classmethod
    def get_data(cls, context: dict | None) -> Self | None:
        """Extract a `source` value from the provided context.

        Context keys:
        - `source_key`: optional source key passed from CLI or caller
        - `item_data`: the dict being prepared for the node
        """
        if not context:
            return None

        source = None
        # Prefer explicit source_key provided by create_graph
        source = context.get("source_key") if isinstance(context, dict) else None

        # Fallback: try to read existing metadata.source from item_data
        if not source:
            item_data = context.get("item_data") if isinstance(context, dict) else None
            if isinstance(item_data, dict):
                meta = item_data.get("metadata")
                if isinstance(meta, dict):
                    source = meta.get("source")

        # If still missing, return None (no extra struct)
        if not source:
            return None

        return cls(source=source)


class WinLoss(ExtraFields):
    result: str = Field(..., description="Win/Loss outcome (win|loss|unknown)")
    reason: str | None = Field(None, description="Short reason for the outcome")

    @classmethod
    def get_data(cls, context: dict | None) -> Self | None:
        """Return win/loss data when available in the source model."""
        if not context or not isinstance(context, dict):
            return None

        # Prefer win_loss from the root model (e.g. CrmExtract).
        root_model = context.get("root_model")
        if root_model is not None and hasattr(root_model, "win_loss"):
            win_loss = getattr(root_model, "win_loss", None)
            if win_loss is not None and hasattr(win_loss, "result"):
                result = getattr(win_loss, "result", None)
                reason = getattr(win_loss, "reason", None)
                if result:
                    return cls(result=str(result), reason=str(reason) if reason else None)

        # Fallback: accept already-materialised win_loss from item_data.
        item_data = context.get("item_data") or {}
        if isinstance(item_data, dict) and "win_loss" in item_data and isinstance(item_data["win_loss"], dict):
            win_loss_dict = item_data["win_loss"]
            result = win_loss_dict.get("result")
            reason = win_loss_dict.get("reason")
            if result:
                return cls(result=str(result), reason=str(reason) if reason else None)

        return None


def get_common_nodes() -> list[GraphNode]:
    return [
        GraphNode(
            node_class=Opportunity,
            extra_classes=[WinLoss],
            name_from="name",
            description="Core opportunity information with financial metrics embedded",
            deduplication_key="opportunity_id",
            index_fields=["name", "status"],
        ),
        GraphNode(
            node_class=Customer,
            name_from="name",
            description="Customer organization details",
            index_fields=["name"],
        ),
        GraphNode(
            node_class=Person,
            name_from="name",
            deduplication_key="name",
            description="Individual contacts and team members",
        ),
    ]
