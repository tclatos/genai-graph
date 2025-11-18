"""Architecture document subgraph for EKG system.

Contains all architecture-specific data model logic and BAML client integration.
Builds a knowledge graph for Software Architecture documents with technical components
and solutions as nodes, and their relationships and purposes as edges.
"""

from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel

from genai_graph.core.graph_registry import register_subgraph
from genai_graph.core.graph_schema import GraphSchema
from genai_graph.core.subgraph import PydanticSubgraph
from genai_graph.ekg.baml_client.types import Customer, SWArchitectureDocument


class ArchitectureDocumentSubgraph(PydanticSubgraph, BaseModel):
    """Architecture document data subgraph implementation."""

    top_class: Type[BaseModel] = SWArchitectureDocument
    kv_store_id: str = "default"

    def build_schema(self) -> GraphSchema:
        """Build the graph schema configuration for architecture document data.

        Returns:
            GraphSchema with all node and relationship configurations
        """
        from genai_graph.core.graph_schema import (
            GraphNodeConfig,
            GraphRelationConfig,
            create_schema,
        )
        from genai_graph.ekg.baml_client.types import (
            Opportunity,
            Solution,
            SWArchitectureDocument,
            TechnicalComponent,
        )

        # BAML-generated types imported above

        # Define nodes with descriptions
        nodes = [
            # Root node - the architecture document itself
            GraphNodeConfig(
                baml_class=SWArchitectureDocument,
                name_from=lambda data, base: f"Architecture:{data.get('document_date', 'unknown')}",
                description="Root node containing the complete architecture document with technical stack and solutions",
            ),
            # Opportunity/Project node - the subject of the architecture
            GraphNodeConfig(
                baml_class=Opportunity,
                name_from="name",
                description="The opportunity/project that this architecture addresses",
                index_fields=["name"],
                deduplication_key="name",
            ),
            GraphNodeConfig(
                baml_class=Customer,
                name_from="name",
                description="Customer organization details",
                index_fields=["name"],
            ),
            # Technical Component nodes - individual technologies and tools
            GraphNodeConfig(
                baml_class=TechnicalComponent,
                name_from="name",
                description="Individual technology, framework, platform, tool, or infrastructure component",
                index_fields=["name", "type"],
                deduplication_key="name",
            ),
            # Solution nodes - managed services, products, and OSS solutions
            GraphNodeConfig(
                baml_class=Solution,
                name_from="name",
                description="Specific product, managed service, or OSS solution used in the architecture",
                index_fields=["name", "vendor", "type"],
                deduplication_key="name",
            ),
        ]

        # Define relationships with descriptions
        # BAML properties matching p_*_ pattern (e.g., p_purpose_) are automatically
        # converted to edge properties
        relations = [
            # Document to project
            GraphRelationConfig(
                from_node=SWArchitectureDocument,
                to_node=Opportunity,
                name="SOFWARE_ARCHITECURE",
                description="Architecture document for the opportunity/project",
            ),
            # Document to technical components in the stack
            GraphRelationConfig(
                from_node=SWArchitectureDocument,
                to_node=TechnicalComponent,
                name="USED_TECHNOLOGY",
                description="Architecture includes this technology component.",
            ),
            # Document to solutions
            GraphRelationConfig(
                from_node=SWArchitectureDocument,
                to_node=Solution,
                name="USED_SOLUTION",
                description="Architecture leverages this solution. ",
            ),
            GraphRelationConfig(
                from_node=Opportunity,
                to_node=Customer,
                name="HAS_CUSTOMER",
                description="Opportunity belongs to customer",
            ),
            # Component to component relationships (dependencies/integration)
        ]

        # Create and validate the schema - this will auto-deduce all field paths
        schema = create_schema(root_model_class=self.top_class, nodes=nodes, relations=relations)

        return schema

    def get_sample_queries(self) -> list[str]:
        """Get list of sample Cypher queries for architecture data."""
        return [
            # Node type summary
            "MATCH (n) RETURN labels(n)[0] as NodeType, count(n) as Count",
            # List all technologies in the stack
            "MATCH (doc:SWArchitectureDocument)-[r:USES_TECHNOLOGY]->(tech:TechnicalComponent) "
            "RETURN tech.name, tech.type, r.p_purpose_ LIMIT 10",
            # List all solutions used
            "MATCH (doc:SWArchitectureDocument)-[r:USES_SOLUTION]->(sol:Solution) "
            "RETURN sol.name, sol.vendor, sol.type, r.p_purpose_ LIMIT 10",
            # Find technology integrations
        ]

    def get_entity_name_from_data(self, data: Any) -> str:
        """Extract a human-readable entity name from loaded data."""
        if hasattr(data, "opportunity") and hasattr(data.opportunity, "name"):
            return f"Architecture: {data.opportunity.name}"
        if hasattr(data, "document_date"):
            return f"Architecture: {data.document_date}"
        return "Architecture Document"


# Register this subgraph implementation in the global registry


def register(registry: "GraphRegistry | None " = None) -> None:  # noqa: F821
    """Register the ArchitectureDocument subgraph implementation.

    The optional ``registry`` argument allows the caller (typically
    :class:`genai_graph.core.graph_registry.GraphRegistry`) to supply the
    registry instance explicitly.
    """
    register_subgraph("ArchitectureDocument", ArchitectureDocumentSubgraph(), registry=registry)
