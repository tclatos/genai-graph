"""Unit tests for pure helpers in genai_graph.kg.ingest.extract."""

from __future__ import annotations

from pydantic import BaseModel

from genai_graph.kg.ingest.extract import (
    TypedNull,
    _get_kuzu_type,
    _normalize_embedded_dict,
    get_field_by_path,
)


class TestGetKuzuType:
    def test_none_annotation(self) -> None:
        assert _get_kuzu_type(None) == "STRING"

    def test_str(self) -> None:
        assert _get_kuzu_type(str) == "STRING"

    def test_int(self) -> None:
        assert _get_kuzu_type(int) == "INT64"

    def test_float(self) -> None:
        assert _get_kuzu_type(float) == "DOUBLE"

    def test_optional_unwrapping(self) -> None:
        assert _get_kuzu_type(int | None) == "INT64"
        assert _get_kuzu_type(None | str) == "STRING"

    def test_list_of_str(self) -> None:
        assert _get_kuzu_type(list[str]) == "STRING[]"

    def test_list_of_float(self) -> None:
        assert _get_kuzu_type(list[float]) == "FLOAT[]"

    def test_optional_list(self) -> None:
        assert _get_kuzu_type(list[str] | None) == "STRING[]"
        assert _get_kuzu_type(list[float] | None) == "FLOAT[]"

    def test_typing_union(self) -> None:
        from typing import Optional, Union

        assert _get_kuzu_type(Optional[int]) == "INT64"  # noqa: UP045
        assert _get_kuzu_type(Union[str, None]) == "STRING"  # noqa: UP007

    def test_unknown_type_falls_back_to_string(self) -> None:
        class Custom:
            pass

        assert _get_kuzu_type(Custom) == "STRING"


class TestTypedNull:
    def test_repr_is_kuzu_cast(self) -> None:
        assert repr(TypedNull("DOUBLE")) == "CAST(NULL AS DOUBLE)"

    def test_type_name_stored(self) -> None:
        assert TypedNull("STRING[]").type_name == "STRING[]"


class _Nested(BaseModel):
    score: float
    tags: list[str]


class _Root(BaseModel):
    name: str
    nested: _Nested | None = None
    items: list[_Nested] = []


class TestGetFieldByPath:
    def test_simple_field(self) -> None:
        root = _Root(name="r")
        assert get_field_by_path(root, "name") == "r"

    def test_nested_field(self) -> None:
        root = _Root(name="r", nested=_Nested(score=1.5, tags=["a"]))
        nested = get_field_by_path(root, "nested")
        assert isinstance(nested, _Nested)
        assert nested.score == 1.5

    def test_missing_field_returns_none(self) -> None:
        root = _Root(name="r")
        assert get_field_by_path(root, "nonexistent") is None

    def test_none_intermediate(self) -> None:
        root = _Root(name="r", nested=None)
        assert get_field_by_path(root, "nested.score") is None

    def test_list_field(self) -> None:
        root = _Root(name="r", items=[_Nested(score=1.0, tags=[]), _Nested(score=2.0, tags=[])])
        items = get_field_by_path(root, "items")
        assert isinstance(items, list)
        assert len(items) == 2


class _Embedded(BaseModel):
    count: int | None = None
    ratio: float | None = None
    label: str | None = None
    flag: bool | None = None
    tags: list[str] | None = None


class TestNormalizeEmbeddedDict:
    def test_all_fields_present_with_typed_nulls(self) -> None:
        result = _normalize_embedded_dict({}, _Embedded)
        # Missing optional numeric fields become TypedNull so Kuzu gets complete STRUCTs
        assert isinstance(result["count"], TypedNull)
        assert result["count"].type_name == "INT64"
        assert isinstance(result["ratio"], TypedNull)
        assert result["ratio"].type_name == "DOUBLE"
        assert isinstance(result["label"], TypedNull)
        assert result["label"].type_name == "STRING"
        assert isinstance(result["flag"], TypedNull)
        assert result["flag"].type_name == "BOOL"

    def test_empty_string_becomes_typed_null(self) -> None:
        result = _normalize_embedded_dict({"count": "", "ratio": "  "}, _Embedded)
        assert isinstance(result["count"], TypedNull)
        assert isinstance(result["ratio"], TypedNull)

    def test_single_string_promoted_to_list(self) -> None:
        result = _normalize_embedded_dict({"tags": "solo"}, _Embedded)
        assert result["tags"] == ["solo"]

    def test_empty_list_becomes_typed_null(self) -> None:
        result = _normalize_embedded_dict({"tags": []}, _Embedded)
        assert isinstance(result["tags"], TypedNull)
        assert result["tags"].type_name == "STRING[]"

    def test_values_pass_through(self) -> None:
        result = _normalize_embedded_dict(
            {"count": 3, "ratio": 0.5, "label": "x", "flag": True, "tags": ["a", "b"]}, _Embedded
        )
        assert result["count"] == 3
        assert result["ratio"] == 0.5
        assert result["label"] == "x"
        assert result["flag"] is True
        assert result["tags"] == ["a", "b"]

    def test_none_values_become_typed_null(self) -> None:
        result = _normalize_embedded_dict({"count": None}, _Embedded)
        assert isinstance(result["count"], TypedNull)
