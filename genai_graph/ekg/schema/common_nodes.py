from pydantic import Field

from genai_graph.core.graph_schema import (
    ExtraFields,
    GraphNode,
)
from genai_graph.ekg.baml_client.types import Customer, Opportunity, Person


class FileMetadata(ExtraFields):
    source: str = Field(..., description="Source of the file from which the data was extracted")

    def get_data(self, some_contex_data) -> "ExtraFields | None":
        # Implement logic to extract extra fields from context data
        pass  # TODO (replacement of current metadata.source field)


def get_common_nodes() -> list[GraphNode]:
    return [
        GraphNode(
            baml_class=Opportunity,
            name_from="name",
            description="Core opportunity information with financial metrics embedded",
            deduplication_key="opportunity_id",
            index_fields=["name", "status"],
        ),
        GraphNode(
            baml_class=Customer,
            name_from="name",
            description="Customer organization details",
            index_fields=["name"],
        ),
        GraphNode(
            baml_class=Person,
            name_from="name",
            deduplication_key="name",
            description="Individual contacts and team members",
        ),
    ]
