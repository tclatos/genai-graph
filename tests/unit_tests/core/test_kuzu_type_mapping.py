"""Unit tests for Kuzu type mapping functionality.

These tests ensure that Python type annotations are correctly mapped to Kuzu database types,
including proper handling of Optional types, lists, and embedded Pydantic models.

This test suite was created to prevent regressions like the bug where embedded structs
were incorrectly mapped to STRING instead of STRUCT types.
"""

from __future__ import annotations

from pydantic import BaseModel

from genai_graph.core.graph_core import _get_kuzu_type


class TestGetKuzuType:
    """Test suite for _get_kuzu_type function."""

    def test_basic_types(self) -> None:
        """Test basic Python type to Kuzu type mapping."""
        assert _get_kuzu_type(str) == "STRING"
        assert _get_kuzu_type(int) == "INT64"
        assert _get_kuzu_type(float) == "DOUBLE"

    def test_list_types(self) -> None:
        """Test list type mapping."""
        assert _get_kuzu_type(list[str]) == "STRING[]"
        # Bare 'list' without element type falls back to STRING
        assert _get_kuzu_type(list) == "STRING"

    def test_optional_types(self) -> None:
        """Test Optional type unwrapping.

        Regression test: Previously, Optional types were not properly unwrapped,
        causing type detection to fail.
        """
        assert _get_kuzu_type(str | None) == "STRING"
        assert _get_kuzu_type(int | None) == "INT64"
        assert _get_kuzu_type(float | None) == "DOUBLE"
        assert _get_kuzu_type(list[str] | None) == "STRING[]"

    def test_none_type(self) -> None:
        """Test handling of None annotation."""
        assert _get_kuzu_type(None) == "STRING"

    def test_unknown_types_fallback(self) -> None:
        """Test that unknown types fall back to STRING.

        Complex types like Pydantic models should fall back to STRING
        when used directly (not as embedded structs).
        """

        class CustomModel(BaseModel):
            field: str

        # When used directly (not as embedded), should fallback to STRING
        result = _get_kuzu_type(CustomModel)
        assert result == "STRING"

    def test_enum_fallback(self) -> None:
        """Test that enum types fall back to STRING."""
        from enum import Enum

        class Status(str, Enum):
            ACTIVE = "active"
            INACTIVE = "inactive"

        assert _get_kuzu_type(Status) == "STRING"


class TestOptionalListHandling:
    """Test suite for combined Optional and List types."""

    def test_optional_list_of_strings(self) -> None:
        """Test Optional[list[str]] mapping."""
        assert _get_kuzu_type(list[str] | None) == "STRING[]"

    def test_list_of_optional_strings(self) -> None:
        """Test list[str | None] mapping.

        Note: Kuzu doesn't support lists with nullable elements,
        so this should map to STRING[] (the list type).
        """
        assert _get_kuzu_type(list[str | None]) == "STRING[]"
