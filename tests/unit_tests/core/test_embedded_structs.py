"""Unit tests for embedded struct handling in graph schema.

These tests ensure that embedded Pydantic models are correctly processed and
converted to STRUCT types in the database schema.

This test suite was created to prevent regressions like the bug where the code
after 'if not field_name: continue' was unreachable due to incorrect indentation,
causing embedded structs to be skipped entirely and fall back to STRING types.
"""

from __future__ import annotations

from pydantic import BaseModel

from genai_graph.core.graph_schema import GraphNode, GraphSchema


class TestEmbeddedStructs:
    """Test suite for embedded struct processing."""

    def test_find_embedded_field_simple(self) -> None:
        """Test finding embedded field in parent class."""
        from genai_graph.core.graph_schema import _find_embedded_field_for_class

        class Metrics(BaseModel):
            score: float
            count: int

        class Company(BaseModel):
            name: str
            metrics: Metrics

        field_name = _find_embedded_field_for_class(Company, Metrics)
        assert field_name == "metrics"

    def test_find_embedded_field_optional(self) -> None:
        """Test finding optional embedded field."""
        from genai_graph.core.graph_schema import _find_embedded_field_for_class

        class Address(BaseModel):
            city: str

        class Person(BaseModel):
            name: str
            address: Address | None = None

        field_name = _find_embedded_field_for_class(Person, Address)
        assert field_name == "address"

    def test_find_embedded_field_in_list(self) -> None:
        """Test finding embedded field when it's in a list."""
        from genai_graph.core.graph_schema import _find_embedded_field_for_class

        class Tag(BaseModel):
            name: str

        class Article(BaseModel):
            title: str
            tags: list[Tag] = []

        field_name = _find_embedded_field_for_class(Article, Tag)
        assert field_name == "tags"

    def test_find_embedded_field_not_found(self) -> None:
        """Test that None is returned when field is not found."""
        from genai_graph.core.graph_schema import _find_embedded_field_for_class

        class Unrelated(BaseModel):
            value: str

        class Company(BaseModel):
            name: str

        field_name = _find_embedded_field_for_class(Company, Unrelated)
        assert field_name is None

    def test_embedded_struct_classes_property(self) -> None:
        """Test that embedded_struct_classes correctly filters extra_classes."""

        class Metrics(BaseModel):
            score: float

        class Company(BaseModel):
            name: str
            metrics: Metrics

        node = GraphNode(
            node_class=Company,
            extra_classes=[Metrics],
            name_from="name",
        )

        # All items in extra_classes should be in embedded_struct_classes
        # since none are ExtraFields subclasses
        assert Metrics in node.embedded_struct_classes

    def test_struct_field_names(self) -> None:
        """Test that struct_field_names returns correct field names."""

        class Metrics(BaseModel):
            score: float
            count: int

        class Company(BaseModel):
            name: str
            metrics: Metrics | None = None

        node = GraphNode(
            node_class=Company,
            extra_classes=[Metrics],
            name_from="name",
        )

        field_names = node.struct_field_names()
        assert "metrics" in field_names


class TestSchemaValidation:
    """Test graph schema validation and struct generation."""

    def test_schema_with_embedded_structs(self) -> None:
        """Test that schema correctly identifies embedded struct fields.

        This is a regression test for the bug where embedded structs were
        not being processed due to unreachable code after 'continue'.
        """

        class FinancialMetrics(BaseModel):
            tcv: float | None = None
            annual_revenue: float | None = None
            project_margin: float | None = None

        class Opportunity(BaseModel):
            name: str
            financials: FinancialMetrics | None = None

        node = GraphNode(
            node_class=Opportunity,
            extra_classes=[FinancialMetrics],
            name_from="name",
        )

        schema = GraphSchema(
            root_model_class=Opportunity,
            nodes=[node],
            relations=[],
        )

        # Validate schema was built successfully
        assert len(schema.nodes) == 1
        assert schema.nodes[0].embedded_struct_classes == [FinancialMetrics]
        assert "financials" in schema.nodes[0].struct_field_names()

    def test_multiple_embedded_structs(self) -> None:
        """Test handling multiple embedded structs in one node."""

        class FinancialMetrics(BaseModel):
            revenue: float

        class CompetitiveLandscape(BaseModel):
            position: str
            differentiators: list[str] = []

        class Opportunity(BaseModel):
            name: str
            financials: FinancialMetrics | None = None
            competition: CompetitiveLandscape | None = None

        node = GraphNode(
            node_class=Opportunity,
            extra_classes=[FinancialMetrics, CompetitiveLandscape],
            name_from="name",
        )

        schema = GraphSchema(
            root_model_class=Opportunity,
            nodes=[node],
            relations=[],
        )

        field_names = schema.nodes[0].struct_field_names()
        assert "financials" in field_names
        assert "competition" in field_names
        assert len(schema.nodes[0].embedded_struct_classes) == 2
