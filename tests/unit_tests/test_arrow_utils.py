"""Unit tests for genai_graph.kg.ingest.arrow_utils."""

from __future__ import annotations

from enum import StrEnum
from typing import Optional

import pyarrow as pa
import pytest
from pydantic import BaseModel

from genai_graph.kg.ingest.arrow_utils import (
    arrow_type_contains_struct,
    ladybug_type_to_arrow,
    pydantic_annotation_to_arrow,
)
from genai_graph.kg.ingest.merge import _build_node_arrow_schema as build_node_arrow_schema

# ---------------------------------------------------------------------------
# pydantic_annotation_to_arrow
# ---------------------------------------------------------------------------


class TestPydanticAnnotationToArrow:
    def test_str(self):
        assert pydantic_annotation_to_arrow(str) == pa.string()

    def test_int(self):
        assert pydantic_annotation_to_arrow(int) == pa.int64()

    def test_float(self):
        assert pydantic_annotation_to_arrow(float) == pa.float64()

    def test_bool(self):
        assert pydantic_annotation_to_arrow(bool) == pa.bool_()

    def test_none_annotation(self):
        assert pydantic_annotation_to_arrow(None) == pa.string()

    def test_optional_str(self):
        assert pydantic_annotation_to_arrow(Optional[str]) == pa.string()  # noqa: UP045

    def test_optional_int(self):
        assert pydantic_annotation_to_arrow(Optional[int]) == pa.int64()  # noqa: UP045

    def test_union_syntax(self):
        # Python 3.10+ X | None syntax
        assert pydantic_annotation_to_arrow(str | None) == pa.string()

    def test_list_str(self):
        assert pydantic_annotation_to_arrow(list[str]) == pa.list_(pa.string())

    def test_list_float(self):
        assert pydantic_annotation_to_arrow(list[float]) == pa.list_(pa.float64())

    def test_optional_list_str(self):
        assert pydantic_annotation_to_arrow(list[str] | None) == pa.list_(pa.string())

    def test_enum(self):
        class Color(StrEnum):
            red = "red"

        assert pydantic_annotation_to_arrow(Color) == pa.string()

    def test_pydantic_submodel(self):
        class Inner(BaseModel):
            x: str
            y: int

        result = pydantic_annotation_to_arrow(Inner)
        assert pa.types.is_struct(result)
        assert result.field("x").type == pa.string()
        assert result.field("y").type == pa.int64()

    def test_optional_pydantic_submodel(self):
        class Inner(BaseModel):
            value: float

        result = pydantic_annotation_to_arrow(Inner | None)
        assert pa.types.is_struct(result)

    def test_unknown_type_falls_back_to_string(self):
        class WeirdType:
            pass

        assert pydantic_annotation_to_arrow(WeirdType) == pa.string()


# ---------------------------------------------------------------------------
# arrow_type_contains_struct
# ---------------------------------------------------------------------------


class TestArrowTypeContainsStruct:
    def test_plain_struct(self):
        t = pa.struct([pa.field("x", pa.string())])
        assert arrow_type_contains_struct(t) is True

    def test_list_of_struct(self):
        t = pa.list_(pa.struct([pa.field("x", pa.string())]))
        assert arrow_type_contains_struct(t) is True

    def test_string(self):
        assert arrow_type_contains_struct(pa.string()) is False

    def test_list_of_string(self):
        assert arrow_type_contains_struct(pa.list_(pa.string())) is False

    def test_int(self):
        assert arrow_type_contains_struct(pa.int64()) is False


# ---------------------------------------------------------------------------
# ladybug_type_to_arrow
# ---------------------------------------------------------------------------


class TestLadybugTypeToArrow:
    @pytest.mark.parametrize(
        "type_str, expected",
        [
            ("STRING", pa.string()),
            ("string", pa.string()),
            ("DOUBLE", pa.float64()),
            ("FLOAT", pa.float64()),
            ("INT64", pa.int64()),
            ("INT32", pa.int64()),
            ("INT16", pa.int64()),
            ("BOOL", pa.bool_()),
            ("UNKNOWN_TYPE", pa.string()),  # fallback
            ("STRING[]", pa.list_(pa.string())),
            ("INT64[]", pa.list_(pa.int64())),
            ("FLOAT[]", pa.list_(pa.float64())),
        ],
    )
    def test_type_mapping(self, type_str: str, expected: pa.DataType):
        assert ladybug_type_to_arrow(type_str) == expected

    def test_whitespace_stripped(self):
        assert ladybug_type_to_arrow("  STRING  ") == pa.string()


# ---------------------------------------------------------------------------
# build_node_arrow_schema
# ---------------------------------------------------------------------------


class TestBuildNodeArrowSchema:
    def test_basic_model(self):
        class Node(BaseModel):
            value: str
            count: int

        schema = build_node_arrow_schema(Node, primary_key_field="id")
        # Sentinels come first
        assert schema.names[:3] == ["id", "name", "_original_name"]
        assert "value" in schema.names
        assert "count" in schema.names
        assert "_created_at" in schema.names
        assert "_updated_at" in schema.names

    def test_primary_key_equals_name_no_duplicates(self):
        class Node(BaseModel):
            name: str
            org: str | None = None

        schema = build_node_arrow_schema(Node, primary_key_field="name")
        assert schema.names.count("name") == 1

    def test_excluded_fields_omitted(self):
        class Node(BaseModel):
            name: str
            p_role_: str | None = None  # edge-property sentinel

        schema = build_node_arrow_schema(Node, primary_key_field="name", excluded_fields={"p_role_"})
        assert "p_role_" not in schema.names

    def test_relationship_target_submodel_excluded(self):
        class Customer(BaseModel):
            name: str

        class Opportunity(BaseModel):
            name: str
            customer: Customer | None = None  # relationship target

        schema = build_node_arrow_schema(Opportunity, primary_key_field="name", excluded_fields={"customer"})
        assert "customer" not in schema.names
        assert "name" in schema.names

    def test_struct_safety_net_excludes_unnamed_submodels(self):
        """Sub-model fields NOT in excluded_fields are still skipped (safety net)."""

        class Inner(BaseModel):
            x: str

        class Node(BaseModel):
            id: str
            inner: Inner | None = None  # not excluded explicitly, not embedded

        schema = build_node_arrow_schema(Node, primary_key_field="id")
        assert "inner" not in schema.names

    def test_embedded_struct_class(self):
        class Meta(BaseModel):
            source: str
            confidence: float

        class Node(BaseModel):
            id: str
            meta: Meta | None = None

        schema = build_node_arrow_schema(
            Node,
            primary_key_field="id",
            embedded_struct_classes=[Meta],
        )
        assert "meta" in schema.names
        meta_idx = schema.get_field_index("meta")
        assert pa.types.is_struct(schema.field(meta_idx).type)

    def test_embedding_field(self):
        class Node(BaseModel):
            id: str
            name: str
            text_embedding: list[float] = []

        schema = build_node_arrow_schema(Node, primary_key_field="id")
        emb_idx = schema.get_field_index("text_embedding")
        assert pa.types.is_list(schema.field(emb_idx).type)
        assert schema.field(emb_idx).type.value_type == pa.float64()

    def test_timestamp_sentinels_added(self):
        class Node(BaseModel):
            id: str

        schema = build_node_arrow_schema(Node, primary_key_field="id")
        assert "_created_at" in schema.names
        assert "_updated_at" in schema.names

    def test_model_with_existing_timestamps_not_duplicated(self):
        class Node(BaseModel):
            id: str
            _created_at: str = ""

        schema = build_node_arrow_schema(Node, primary_key_field="id")
        assert schema.names.count("_created_at") == 1

    def test_no_model_fields(self):
        """Bare non-Pydantic class produces only sentinels."""

        class Plain:
            pass

        schema = build_node_arrow_schema(Plain, primary_key_field="id")
        assert set(schema.names) == {"id", "name", "_original_name", "_created_at", "_updated_at"}
