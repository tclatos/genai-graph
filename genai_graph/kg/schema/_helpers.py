"""KG-schema helper functions.

Generic Pydantic and type-annotation utilities live in
``genai_tk.utils.pydantic_utils.common``; generic BAML text-parsing utilities
live in ``genai_tk.extra.structured.baml_util``.

This module contains only knowledge-graph-domain helpers:

- Optional BAML description loading (injectable via ``baml_file_map`` parameter)
- Node / field / relation description extraction (uses GraphNode API)
- Enum collection from a GraphSchema
- Kuzu/Ladybug type mapping
"""

from __future__ import annotations

from enum import Enum
from typing import Any, get_args, get_origin

from genai_tk.extra.structured.baml_util import parse_baml_content
from genai_tk.utils.pydantic_utils.common import get_class_description, get_field_description, humanize_type

# Re-export so existing callers in this package don't need updating.
_get_class_description = get_class_description
_humanize_type_compact = humanize_type


# ---------------------------------------------------------------------------
# BAML description loading (injectable; project passes its own baml_file_map)
# ---------------------------------------------------------------------------


def _parse_baml_descriptions(baml_file_map: dict[str, str] | None = None) -> dict[str, Any]:
    """Parse ``@description`` annotations from BAML files.

    Args:
        baml_file_map: Optional mapping of filename → BAML content.  When
            ``None`` or empty, an empty description dict is returned so that
            downstream enrichment gracefully falls back to Pydantic
            ``Field(description=...)`` and docstrings.

    Returns:
        Dict with keys ``classes``, ``fields``, ``enums``.
    """
    classes: dict[str, str] = {}
    fields: dict[str, dict[str, str]] = {}
    enums: dict[str, dict[str, str]] = {}

    if not baml_file_map:
        return {"classes": classes, "fields": fields, "enums": enums}

    excluded = {"clients.baml", "generators.baml"}
    for filename, content in baml_file_map.items():
        if filename not in excluded:
            parse_baml_content(content, classes, fields, enums)

    return {"classes": classes, "fields": fields, "enums": enums}


# ---------------------------------------------------------------------------
# Node / field / relation description extraction
# ---------------------------------------------------------------------------


def _get_field_description(node_class: type, field_name: str, field_info: Any, baml_docs: dict[str, Any]) -> str:
    """Return field description: Pydantic ``Field(description=...)`` then BAML fallback."""
    pydantic_desc = get_field_description(field_info)
    if pydantic_desc:
        return pydantic_desc
    return baml_docs["fields"].get(node_class.__name__, {}).get(field_name, "")


def _get_node_description(node: Any, baml_docs: dict[str, Any]) -> str:
    """Return node description: GraphNode.description → docstring → BAML."""
    if node.description:
        return node.description
    desc = get_class_description(node.node_class)
    if desc:
        return desc
    return baml_docs["classes"].get(node.node_class.__name__, "")


def _get_relation_properties(node_class: Any, baml_docs: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Extract ``p_*_`` relationship properties from *node_class*.

    Returns list of ``(display_name, humanized_type, description)`` tuples.
    """
    properties = []
    if not hasattr(node_class, "model_fields"):
        return properties
    for field_name, field_info in node_class.model_fields.items():
        if field_name.startswith("p_") and field_name.endswith("_"):
            display_name = field_name[2:-1]
            properties.append(
                (
                    display_name,
                    humanize_type(field_info.annotation),
                    _get_field_description(node_class, field_name, field_info, baml_docs),
                )
            )
    return properties


# ---------------------------------------------------------------------------
# Enum collection
# ---------------------------------------------------------------------------


def _collect_used_enums(schema: Any) -> set[type]:
    """Collect all Enum types referenced by node classes in *schema*."""
    import types
    import typing

    used_enums: set[type] = set()

    def _from_annotation(annotation: Any) -> None:
        if annotation is None or annotation is type(None):
            return
        if isinstance(annotation, type) and issubclass(annotation, Enum):
            used_enums.add(annotation)
            return
        origin = get_origin(annotation)
        if origin is typing.Union or (hasattr(types, "UnionType") and origin is types.UnionType):
            for arg in get_args(annotation):
                _from_annotation(arg)
            return
        if origin in (list, set, tuple, dict):
            for arg in get_args(annotation):
                _from_annotation(arg)

    for node in schema.nodes:
        for fi in getattr(node.node_class, "model_fields", {}).values():
            _from_annotation(fi.annotation)
        for emb_cls in getattr(node, "embedded_struct_classes", []) or []:
            for fi in getattr(emb_cls, "model_fields", {}).values():
                _from_annotation(fi.annotation)

    return used_enums


# ---------------------------------------------------------------------------
# Kuzu / Ladybug type mapping
# ---------------------------------------------------------------------------


def _get_kuzu_type_for_field(annotation: Any) -> str:
    """Map a Python type annotation to a Kuzu/Ladybug type string."""
    import types
    import typing

    if annotation is None or annotation is type(None):
        return "STRING"

    origin = get_origin(annotation)
    actual = annotation

    if origin is typing.Union or (hasattr(types, "UnionType") and origin is types.UnionType):
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if non_none:
            actual = non_none[0]
            origin = get_origin(actual)

    if origin is list:
        inner_args = get_args(actual)
        if inner_args:
            if inner_args[0] is float:
                return "FLOAT[]"
            return f"{_get_kuzu_type_for_field(inner_args[0]).rstrip('[]')}[]"
        return "STRING[]"

    if origin is dict:
        return "MAP(STRING, STRING)"

    if actual is int:
        return "INT64"
    if actual is float:
        return "DOUBLE"
    if actual is bool:
        return "BOOL"
    if actual is str:
        return "STRING"
    if isinstance(actual, type) and issubclass(actual, Enum):
        return "STRING"
    if hasattr(actual, "__name__"):
        return f"STRUCT:{actual.__name__}"
    return "STRING"


# ---------------------------------------------------------------------------
# BAML description parsing
# ---------------------------------------------------------------------------
