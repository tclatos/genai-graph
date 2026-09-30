"""Unit tests for the SimilarityFactory and SimilaritySpec (pure logic, no database).

The end-to-end behaviour of ``compute_similarities`` against a real Ladybug
database is covered in ``tests/integration_tests/test_similarity_flow.py``.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from genai_graph.kg.factories.similarity import (
    SimilarityFactory,
    SimilaritySpec,
)
from genai_graph.kg.schema.core import GraphNode, GraphRelation, GraphSchema


class NodeA(BaseModel):
    code: str
    description: str


class NodeB(BaseModel):
    id: str
    architecture: str


class _TestMatcher(SimilarityFactory):
    """Minimal concrete SimilarityFactory for unit tests."""

    def build_schema(self) -> GraphSchema:
        node_a = GraphNode(node_class=NodeA, name_from="description", key_from="code")
        node_b = GraphNode(node_class=NodeB, name_from="architecture", key_from="AUTO_ID")
        relation = GraphRelation(
            from_node=node_b,
            to_node=node_a,
            name="SIMILAR_TO",
            properties={"similarity_score": float},
        )
        return GraphSchema(root_model_class=None, nodes=[node_b, node_a], relations=[relation])


def _make_matcher(threshold: float = 0.8, top_k: int = 5, iterate_over: str = "from") -> _TestMatcher:
    return _TestMatcher(
        similarities=[
            SimilaritySpec(
                relationship="SIMILAR_TO",
                from_node="NodeB.architecture",
                to_node="NodeA.description",
                iterate_over=iterate_over,  # type: ignore[arg-type]
                threshold=threshold,
                top_k=top_k,
            )
        ],
    )


# ---------------------------------------------------------------------------
# _resolve_field
# ---------------------------------------------------------------------------


class TestResolveField:
    def test_parses_table_and_field(self) -> None:
        table, field, index = SimilarityFactory._resolve_field("L3.description")
        assert table == "L3"
        assert field == "description"
        assert index == "description_index"

    def test_parses_camel_case_field(self) -> None:
        table, field, index = SimilarityFactory._resolve_field("TechnicalApproach.architecture")
        assert table == "TechnicalApproach"
        assert field == "architecture"
        assert index == "architecture_index"

    def test_strips_whitespace(self) -> None:
        table, field, _ = SimilarityFactory._resolve_field("  NodeA . my_field  ")
        assert table == "NodeA"
        assert field == "my_field"

    def test_raises_on_missing_dot(self) -> None:
        with pytest.raises(ValueError, match=r"NodeClass\.field_name"):
            SimilarityFactory._resolve_field("no_dot_here")


# ---------------------------------------------------------------------------
# _combine_scores
# ---------------------------------------------------------------------------


class TestCombineScores:
    def test_first_single(self) -> None:
        assert SimilarityFactory._combine_scores([0.9], "first") == 0.9

    def test_first_uses_first_element(self) -> None:
        assert SimilarityFactory._combine_scores([0.8, 0.95], "first") == 0.8

    def test_max(self) -> None:
        assert SimilarityFactory._combine_scores([0.8, 0.95, 0.7], "max") == 0.95

    def test_avg(self) -> None:
        result = SimilarityFactory._combine_scores([0.8, 0.9], "avg")
        assert abs(result - 0.85) < 1e-9

    def test_empty_scores_returns_zero(self) -> None:
        assert SimilarityFactory._combine_scores([], "max") == 0.0


# ---------------------------------------------------------------------------
# _pk_field_for
# ---------------------------------------------------------------------------


class TestPkFieldFor:
    def test_string_key_from(self) -> None:
        matcher = _make_matcher()
        assert matcher._pk_field_for("NodeA") == "code"

    def test_auto_id_key_from(self) -> None:
        matcher = _make_matcher()
        assert matcher._pk_field_for("NodeB") == "id"

    def test_unknown_table_returns_id(self) -> None:
        matcher = _make_matcher()
        assert matcher._pk_field_for("UnknownTable") == "id"


# ---------------------------------------------------------------------------
# compute_similarities — backend type guard
# ---------------------------------------------------------------------------


class TestComputeSimilaritiesGuard:
    def test_non_kuzu_backend_returns_empty_result(self) -> None:
        """Non-KuzuBackend backends are skipped with a warning (real Neo4jBackend, no mock)."""
        from genai_graph.kg.backend import Neo4jBackend

        matcher = _make_matcher()
        result = matcher.compute_similarities(Neo4jBackend())

        assert result.relationships_created == 0
        assert result.pairs_evaluated == 0
        assert result.factory_name == matcher.name


# ---------------------------------------------------------------------------
# SimilaritySpec
# ---------------------------------------------------------------------------


class TestSimilaritySpec:
    def test_from_alias(self) -> None:
        """SimilaritySpec accepts 'from' alias for from_node."""
        spec = SimilaritySpec.model_validate(
            {"relationship": "REL", "from": "A.f", "to": "B.g", "iterate_over": "from"}
        )
        assert spec.from_node == "A.f"
        assert spec.to_node == "B.g"

    def test_python_field_names(self) -> None:
        """SimilaritySpec can be constructed with Python names (populate_by_name=True)."""
        spec = SimilaritySpec(relationship="REL", from_node="A.f", to_node="B.g")
        assert spec.from_node == "A.f"
        assert spec.to_node == "B.g"

    def test_defaults(self) -> None:
        spec = SimilaritySpec(relationship="X", from_node="A.f", to_node="B.g")
        assert spec.threshold == 0.8
        assert spec.top_k == 5
        assert spec.iterate_over == "from"
        assert spec.combiner == "first"


# ---------------------------------------------------------------------------
# Schema construction
# ---------------------------------------------------------------------------


class TestSimilarityFactorySchema:
    """Test SimilarityFactory schema construction with generic domain models."""

    def _make_matcher(self) -> SimilarityFactory:
        """Build a minimal SimilarityFactory that links Concept → Topic by description."""

        class Concept(BaseModel):
            name: str
            description: str | None = None
            description_embedding: list[float] | None = None

        class Topic(BaseModel):
            name: str
            description: str | None = None
            description_embedding: list[float] | None = None

        concept_node = GraphNode(node_class=Concept, name_from="name", key_from="name", index_fields=["description"])
        topic_node = GraphNode(node_class=Topic, name_from="name", key_from="name", index_fields=["description"])

        class ConceptTopicMatcher(SimilarityFactory):
            def build_schema(self) -> GraphSchema:
                return GraphSchema(
                    nodes=[concept_node, topic_node],
                    relations=[
                        GraphRelation(
                            from_node=concept_node,
                            to_node=topic_node,
                            name="RELATED_TO",
                            properties={"similarity_score": float},
                        )
                    ],
                )

        return ConceptTopicMatcher(
            similarities=[
                SimilaritySpec(
                    relationship="RELATED_TO",
                    from_node="Concept.description",
                    to_node="Topic.description",
                    iterate_over="from",
                )
            ]
        )

    def test_build_schema_contains_expected_nodes(self) -> None:
        matcher = self._make_matcher()
        schema = matcher.build_schema()
        node_labels = {n.label for n in schema.nodes}
        assert "Concept" in node_labels
        assert "Topic" in node_labels

    def test_build_schema_has_relation(self) -> None:
        matcher = self._make_matcher()
        schema = matcher.build_schema()
        assert len(schema.relations) == 1
        rel = schema.relations[0]
        assert rel.name == "RELATED_TO"
        assert rel.from_node.label == "Concept"
        assert rel.to_node.label == "Topic"

    def test_build_schema_relation_has_similarity_score_property(self) -> None:
        matcher = self._make_matcher()
        schema = matcher.build_schema()
        rel = schema.relations[0]
        assert rel.properties is not None
        assert "similarity_score" in rel.properties
        assert rel.properties["similarity_score"] is float

    def test_get_struct_data_by_key_returns_none(self) -> None:
        matcher = self._make_matcher()
        assert matcher.get_struct_data_by_key("any-key") is None
