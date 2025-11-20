from genai_graph.core.graph_schema import (
    GraphNodeConfig,
)
from genai_graph.ekg.baml_client.types import (
    Customer,
    Opportunity,
    Person,
)


def get_common_nodes() -> list[GraphNodeConfig]:
    return [
        GraphNodeConfig(
            baml_class=Opportunity,
            name_from="name",
            description="Core opportunity information with financial metrics embedded",
            deduplication_key="opportunity_id",
            index_fields=["name", "status"],
        ),
        GraphNodeConfig(
            baml_class=Customer,
            name_from="name",
            description="Customer organization details",
            index_fields=["name"],
        ),
        GraphNodeConfig(
            baml_class=Person,
            name_from="name",
            deduplication_key="name",
            description="Individual contacts and team members",
        ),
    ]
