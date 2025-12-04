"""Integration tests for full graph schema creation and database operations.

These tests validate the end-to-end workflow of creating graph schemas from
Pydantic models and ensuring they are correctly translated to database tables
with proper STRUCT types for embedded fields.
"""

from __future__ import annotations

from pydantic import BaseModel

from genai_graph.core.graph_backend import KuzuBackend
from genai_graph.core.graph_core import create_schema
from genai_graph.core.graph_schema import GraphNode, GraphSchema


class TestSchemaCreation:
    """Test end-to-end schema creation with embedded structs."""

    def test_struct_types_in_database(self, graph_backend: KuzuBackend) -> None:
        """Test that embedded structs are created as STRUCT types in the database.

        This is a regression test for the bug where financials and competition
        fields were created as STRING instead of STRUCT.
        """

        # Define test models matching the real-world case
        class FinancialMetrics(BaseModel):
            tcv: float | None = None
            annual_revenue: float | None = None
            project_margin: float | None = None

        class CompetitiveLandscape(BaseModel):
            competitive_position: str | None = None
            differentiators: list[str] = []

        class ReviewedOpportunity(BaseModel):
            id: str
            name: str
            start_date: str | None = None
            financials: FinancialMetrics | None = None
            competition: CompetitiveLandscape | None = None

        # Create schema
        node = GraphNode(
            node_class=ReviewedOpportunity,
            extra_classes=[FinancialMetrics, CompetitiveLandscape],
            name_from="name",
        )

        schema = GraphSchema(
            root_model_class=ReviewedOpportunity,
            nodes=[node],
            relations=[],
        )

        # Create database schema
        create_schema(graph_backend, schema.nodes, schema.relations)

        # Query database to verify STRUCT types
        result = graph_backend.execute("CALL table_info('ReviewedOpportunity') RETURN *")

        # Convert result to list of dicts for easier testing
        table_info = []
        for row in result:
            table_info.append(
                {
                    "name": row[1],
                    "type": row[2],
                }
            )

        # Verify financials field is STRUCT with correct nested types
        financials_field = next((f for f in table_info if f["name"] == "financials"), None)
        assert financials_field is not None, "financials field not found in table"
        assert "STRUCT" in financials_field["type"], f"financials should be STRUCT, got: {financials_field['type']}"
        assert "tcv DOUBLE" in financials_field["type"]
        assert "annual_revenue DOUBLE" in financials_field["type"]
        assert "project_margin DOUBLE" in financials_field["type"]

        # Verify competition field is STRUCT with correct nested types
        competition_field = next((f for f in table_info if f["name"] == "competition"), None)
        assert competition_field is not None, "competition field not found in table"
        assert "STRUCT" in competition_field["type"], f"competition should be STRUCT, got: {competition_field['type']}"
        assert "competitive_position STRING" in competition_field["type"]
        assert "differentiators STRING[]" in competition_field["type"]

    def test_optional_embedded_structs(self, graph_backend: KuzuBackend) -> None:
        """Test that Optional embedded structs are correctly handled."""

        class Address(BaseModel):
            street: str
            city: str
            country: str

        class Company(BaseModel):
            id: str
            name: str
            address: Address | None = None

        node = GraphNode(
            node_class=Company,
            extra_classes=[Address],
            name_from="name",
        )

        schema = GraphSchema(
            root_model_class=Company,
            nodes=[node],
            relations=[],
        )

        create_schema(graph_backend, schema.nodes, schema.relations)

        result = graph_backend.execute("CALL table_info('Company') RETURN *")

        table_info = []
        for row in result:
            table_info.append(
                {
                    "name": row[1],
                    "type": row[2],
                }
            )

        address_field = next((f for f in table_info if f["name"] == "address"), None)
        assert address_field is not None
        assert "STRUCT" in address_field["type"]
        assert "street STRING" in address_field["type"]
        assert "city STRING" in address_field["type"]
        assert "country STRING" in address_field["type"]

    def test_nested_struct_types(self, graph_backend: KuzuBackend) -> None:
        """Test various data types within embedded structs."""

        class Metrics(BaseModel):
            score: float
            count: int
            is_active: bool
            tags: list[str] = []

        class Project(BaseModel):
            id: str
            name: str
            metrics: Metrics | None = None

        node = GraphNode(
            node_class=Project,
            extra_classes=[Metrics],
            name_from="name",
        )

        schema = GraphSchema(
            root_model_class=Project,
            nodes=[node],
            relations=[],
        )

        create_schema(graph_backend, schema.nodes, schema.relations)

        result = graph_backend.execute("CALL table_info('Project') RETURN *")

        table_info = []
        for row in result:
            table_info.append(
                {
                    "name": row[1],
                    "type": row[2],
                }
            )

        metrics_field = next((f for f in table_info if f["name"] == "metrics"), None)
        assert metrics_field is not None
        assert "STRUCT" in metrics_field["type"]
        assert "score DOUBLE" in metrics_field["type"]
        assert "count INT64" in metrics_field["type"]
        assert "is_active" in metrics_field["type"]  # bool handling
        assert "tags STRING[]" in metrics_field["type"]
