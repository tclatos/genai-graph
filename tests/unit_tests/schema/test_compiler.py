"""Tests for schema compilation functions.

Tests the standalone functions in genai_graph.kg.schema.compiler, covering
path deduction, field map construction, and relation wiring.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

pytestmark = pytest.mark.unit


class TestBuildModelFieldMap:
    def test_root_model_traversed(self, simple_schema) -> None:
        # _model_field_map is populated by the @model_validator — verify it
        field_map = simple_schema._model_field_map
        # Company and Person should appear in the map (reachable from Project)
        from .conftest import Company, Person

        assert Company in field_map
        assert Person in field_map

    def test_nested_class_traversed(self) -> None:
        from genai_graph.kg.schema import GraphNode, GraphSchema

        class Inner(BaseModel):
            value: str

        class Outer(BaseModel):
            name: str
            child: Inner

        node_outer = GraphNode(node_class=Outer, name_from="name", key_from="name")
        node_inner = GraphNode(node_class=Inner, name_from="value", key_from="value")
        schema = GraphSchema(root_model_class=Outer, nodes=[node_outer, node_inner], relations=[])
        assert Inner in schema._model_field_map

    def test_optional_field_traversed(self) -> None:
        from genai_graph.kg.schema import GraphNode, GraphSchema

        class Tag(BaseModel):
            label: str

        class Doc(BaseModel):
            title: str
            tag: Tag | None = None

        tag_node = GraphNode(node_class=Tag, name_from="label", key_from="label")
        doc_node = GraphNode(node_class=Doc, name_from="title", key_from="title")
        schema = GraphSchema(root_model_class=Doc, nodes=[doc_node, tag_node], relations=[])
        assert Tag in schema._model_field_map

    def test_list_field_traversed(self) -> None:
        from genai_graph.kg.schema import GraphNode, GraphSchema

        class Item(BaseModel):
            name: str

        class Collection(BaseModel):
            title: str
            items: list[Item] = Field(default_factory=list)

        item_node = GraphNode(node_class=Item, name_from="name", key_from="name")
        col_node = GraphNode(node_class=Collection, name_from="title", key_from="title")
        schema = GraphSchema(root_model_class=Collection, nodes=[col_node, item_node], relations=[])
        assert Item in schema._model_field_map


class TestDeduceNodeFieldPaths:
    def test_root_model_gets_empty_path(self, simple_schema) -> None:
        from .conftest import Project

        proj = next(n for n in simple_schema.nodes if n.node_class is Project)
        assert proj.field_paths == [""]

    def test_direct_field_path_deduced(self, simple_schema) -> None:
        from .conftest import Company

        comp = next(n for n in simple_schema.nodes if n.node_class is Company)
        assert "client" in comp.field_paths

    def test_optional_field_path_deduced(self, simple_schema) -> None:
        from .conftest import Person

        person = next(n for n in simple_schema.nodes if n.node_class is Person)
        # Person appears via Project.lead (optional) and Project.team (list)
        assert any("lead" in p or "team" in p for p in person.field_paths)

    def test_list_path_marked_as_list(self, simple_schema) -> None:
        from .conftest import Person

        person = next(n for n in simple_schema.nodes if n.node_class is Person)
        team_paths = [p for p in person.field_paths if "team" in p]
        for tp in team_paths:
            assert person.is_list_at_paths[tp] is True

    def test_no_root_model_skips_root_deduction(self) -> None:
        from genai_graph.kg.schema import GraphNode, GraphSchema

        class A(BaseModel):
            name: str

        class B(BaseModel):
            name: str

        a_node = GraphNode(node_class=A, name_from="name", key_from="name")
        b_node = GraphNode(node_class=B, name_from="name", key_from="name")
        # No root_model_class — no traversal, no field paths deduced
        GraphSchema(root_model_class=None, nodes=[a_node, b_node], relations=[])
        assert a_node.field_paths == []
        assert b_node.field_paths == []

    def test_explicit_field_paths_not_overridden(self) -> None:
        from genai_graph.kg.schema import GraphNode, GraphRelation, GraphSchema

        class Parent(BaseModel):
            name: str
            child: Child

        class Child(BaseModel):
            value: str

        parent_node = GraphNode(node_class=Parent, name_from="name", key_from="name")
        child_node = GraphNode(node_class=Child, name_from="value", key_from="value")
        rel = GraphRelation(
            from_node=parent_node,
            to_node=child_node,
            name="HAS_CHILD",
            field_paths=[("", "child")],
        )
        schema = GraphSchema(root_model_class=Parent, nodes=[parent_node, child_node], relations=[rel])
        # Explicit field_paths should be respected
        assert ("", "child") in schema.relations[0].field_paths


class TestDeduceRelationFieldPaths:
    def test_direct_parent_child_relation_wired(self, simple_schema) -> None:
        for_client = next(r for r in simple_schema.relations if r.name == "FOR_CLIENT")
        assert len(for_client.field_paths) > 0
        from_path, to_path = for_client.field_paths[0]
        # client is a direct field on Project (root)
        assert from_path == ""
        assert to_path == "client"

    def test_path_complexity_prefers_containment(self) -> None:
        from genai_graph.kg.schema import GraphNode, GraphRelation, GraphSchema

        class Inner(BaseModel):
            code: str

        class Middle(BaseModel):
            tag: Inner

        class Root(BaseModel):
            mid: Middle
            also_inner: Inner

        root_node = GraphNode(node_class=Root, name_from="mid", key_from="mid")
        mid_node = GraphNode(node_class=Middle, name_from="tag", key_from="tag")
        inner_node = GraphNode(node_class=Inner, name_from="code", key_from="code")
        rel = GraphRelation(from_node=mid_node, to_node=inner_node, name="HAS_INNER")
        schema = GraphSchema(
            root_model_class=Root,
            nodes=[root_node, mid_node, inner_node],
            relations=[rel],
        )
        # mid → inner should prefer mid.tag over also_inner (direct containment)
        from_p, to_p = schema.relations[0].field_paths[0]
        assert from_p == "mid"
        assert to_p == "mid.tag"


class TestComputeExcludedFields:
    def test_relation_target_excluded_from_source(self, simple_schema) -> None:
        from .conftest import Project

        project_node = next(n for n in simple_schema.nodes if n.node_class is Project)
        # 'client' is a FOR_CLIENT target → must be in excluded_fields of project_node
        assert "client" in project_node.excluded_fields

    def test_non_relation_field_not_excluded(self, simple_schema) -> None:
        from .conftest import Project

        project_node = next(n for n in simple_schema.nodes if n.node_class is Project)
        assert "title" not in project_node.excluded_fields
        assert "status" not in project_node.excluded_fields

    def test_p_prefix_fields_excluded(self) -> None:
        from genai_graph.kg.schema import GraphNode, GraphRelation, GraphSchema

        class Source(BaseModel):
            name: str

        class Target(BaseModel):
            name: str
            p_weight_: float | None = None  # edge property convention

        src = GraphNode(node_class=Source, name_from="name", key_from="name")
        tgt = GraphNode(node_class=Target, name_from="name", key_from="name")
        rel = GraphRelation(from_node=src, to_node=tgt, name="CONNECTS")

        class Root(BaseModel):
            name: str
            source: Source
            target: Target

        root = GraphNode(node_class=Root, name_from="name", key_from="name")
        GraphSchema(root_model_class=Root, nodes=[root, src, tgt], relations=[rel])
        # p_weight_ is an edge property on Target — should be excluded from node properties
        assert "p_weight_" in tgt.excluded_fields


class TestSchemaCoherenceValidation:
    def test_no_warnings_on_valid_schema(self, simple_schema) -> None:
        warnings = simple_schema.get_warnings()
        # May have path-ambiguity warnings for Person (appears at lead and team)
        # but no structural errors
        structural_errors = [w for w in warnings if "collision" in w.lower() or "referenced" in w.lower()]
        assert not structural_errors

    def test_label_collision_detected(self) -> None:
        from genai_graph.kg.schema import GraphNode, GraphSchema

        class Person(BaseModel):
            name: str

        class PersonV2(BaseModel):
            name: str

        # Give PersonV2 a table_name that matches Person's class name — collision
        node_a = GraphNode(node_class=Person, name_from="name", key_from="name")
        node_b = GraphNode(
            node_class=PersonV2,
            name_from="name",
            key_from="name",
            table_name="Person",  # explicit collision
        )
        schema = GraphSchema(root_model_class=None, nodes=[node_a, node_b], relations=[])
        warnings = schema.get_warnings()
        assert any("collision" in w.lower() or "share the label" in w.lower() for w in warnings)

    def test_orphan_warning_emitted(self) -> None:
        from genai_graph.kg.schema import GraphNode, GraphSchema

        class Root(BaseModel):
            name: str

        class Orphan(BaseModel):
            name: str

        root_node = GraphNode(node_class=Root, name_from="name", key_from="name")
        orphan_node = GraphNode(node_class=Orphan, name_from="name", key_from="name")
        schema = GraphSchema(root_model_class=Root, nodes=[root_node, orphan_node], relations=[])
        warnings = schema.get_warnings()
        assert any("Orphan" in w or "orphan" in w.lower() or "field paths" in w.lower() for w in warnings)

    def test_explicitly_defined_node_no_orphan_warning(self) -> None:
        from genai_graph.kg.schema import GraphNode, GraphSchema

        class Root(BaseModel):
            name: str

        class External(BaseModel):
            code: str

        root_node = GraphNode(node_class=Root, name_from="name", key_from="name")
        ext_node = GraphNode(node_class=External, name_from="code", key_from="code", explicitly_defined=True)
        schema = GraphSchema(root_model_class=Root, nodes=[root_node, ext_node], relations=[])
        warnings = schema.get_warnings()
        # explicitly_defined=True suppresses orphan warning
        assert not any("External" in w for w in warnings)
