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
        """Return win/loss struct for Opportunity nodes.

        Extracts win/loss data from the root model (e.g., CrmExtract) if available,
        or from item_data directly, or generates deterministic fake values for testing.
        """
        if not context or not isinstance(context, dict):
            return None

        # First try to get win_loss from root model (e.g., CrmExtract containing Opportunity)
        root_model = context.get("root_model")
        if root_model and hasattr(root_model, "win_loss"):
            win_loss = getattr(root_model, "win_loss", None)
            if win_loss and hasattr(win_loss, "result"):
                result = getattr(win_loss, "result", None)
                reason = getattr(win_loss, "reason", None)
                if result:
                    return cls(result=result, reason=reason)

        # Second, try to get win_loss data from item_data (table-backed source)
        item_data = context.get("item_data") or {}
        if "win_loss" in item_data and isinstance(item_data["win_loss"], dict):
            win_loss_dict = item_data["win_loss"]
            result = win_loss_dict.get("result")
            reason = win_loss_dict.get("reason")
            if result:
                return cls(result=result, reason=reason)

        # Fallback: generate deterministic fake data for testing
        candidate = item_data.get("opportunity_id") or item_data.get("id") or item_data.get("name")
        if candidate:
            h = abs(hash(str(candidate)))
            result = "win" if (h % 2 == 0) else "loss"
            reason = "Simulated outcome for testing"
        else:
            result = "unknown"
            reason = None

        return cls(result=result, reason=reason)


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
