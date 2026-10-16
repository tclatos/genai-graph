"""PyArrow utilities for graph node schema derivation.

These helpers convert Pydantic type annotations to PyArrow types and build
``pa.Schema`` objects that match the Ladybug node-table definitions produced
by :func:`genai_graph.kg.ingest.extract.create_schema`.

Extracted from ``merge.py`` so they can be reused as more of the ingestion
pipeline moves to Arrow-native operations.
"""

from __future__ import annotations

import types as _types
from enum import Enum
from typing import Any, get_args, get_origin

import pyarrow as pa

# ---------------------------------------------------------------------------
# Primitive map — defined once at module level to avoid per-call recreation
# ---------------------------------------------------------------------------
_PRIMITIVE_TO_ARROW: dict[type, pa.DataType] = {
    str: pa.string(),
    float: pa.float64(),
    int: pa.int64(),
    bool: pa.bool_(),
}


def pydantic_annotation_to_arrow(annotation: Any) -> pa.DataType:
    """Convert a Python type annotation to a PyArrow ``DataType``.

    Handles the full set of annotations that appear in Pydantic node models:

    - Primitives (``str``, ``int``, ``float``, ``bool``)
    - ``Optional[T]`` / ``T | None`` — unwrapped and recursed
    - ``list[T]`` — maps to ``pa.list_(T)``
    - ``Enum`` subclasses — ``pa.string()``
    - Pydantic sub-models — ``pa.struct([...])`` recursively
    - Unknown / unrecognised annotations — ``pa.string()`` fallback

    Args:
        annotation: Python type annotation to convert.

    Returns:
        Corresponding ``pa.DataType``.
    """
    import typing

    if annotation is None or annotation is type(None):
        return pa.string()

    origin = get_origin(annotation)

    # Unwrap Optional[X] / Union[X, None] / X | None
    if origin is typing.Union or isinstance(annotation, _types.UnionType):
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        return pydantic_annotation_to_arrow(non_none[0]) if non_none else pa.string()

    # list[T]
    if origin is list:
        args = get_args(annotation)
        inner = pydantic_annotation_to_arrow(args[0]) if args else pa.string()
        return pa.list_(inner)

    # dict[str, str] -> MAP(STRING, STRING)
    if origin is dict:
        return pa.map_(pa.string(), pa.string())

    # Primitives
    if annotation in _PRIMITIVE_TO_ARROW:
        return _PRIMITIVE_TO_ARROW[annotation]

    # Enum → stored as string
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return pa.string()

    # Pydantic sub-model → pa.struct (recursive)
    if isinstance(annotation, type) and hasattr(annotation, "model_fields"):
        return pa.struct(
            [
                pa.field(name, pydantic_annotation_to_arrow(fi.annotation))
                for name, fi in annotation.model_fields.items()
            ]
        )

    return pa.string()  # fallback


def arrow_type_contains_struct(arrow_type: pa.DataType) -> bool:
    """Return ``True`` if ``arrow_type`` is or contains a ``pa.struct()``.

    Used as a safety-net to detect Pydantic sub-model fields that resolve to a
    struct but were not listed in ``embedded_struct_classes`` (and therefore
    should be excluded from the node table).

    Args:
        arrow_type: Arrow data type to inspect.

    Returns:
        ``True`` when the type is a struct or a list whose element type is a struct.
    """
    if pa.types.is_struct(arrow_type):
        return True
    if pa.types.is_list(arrow_type):
        return arrow_type_contains_struct(arrow_type.value_type)
    return False


def ladybug_type_to_arrow(type_str: str) -> pa.DataType:
    """Map a Ladybug/Kuzu DDL type string to a PyArrow ``DataType``.

    Handles the scalar types used in ``CREATE NODE TABLE`` statements as well
    as list variants.

    Args:
        type_str: Ladybug type string such as ``"STRING"``, ``"INT64"``,
            ``"FLOAT[]"``, ``"DOUBLE"``.

    Returns:
        Corresponding ``pa.DataType``.
    """
    type_str = type_str.strip()
    if type_str.endswith("[]"):
        return pa.list_(ladybug_type_to_arrow(type_str[:-2]))
    upper = type_str.upper()
    return {
        "STRING": pa.string(),
        "DOUBLE": pa.float64(),
        "FLOAT": pa.float64(),
        "INT64": pa.int64(),
        "INT32": pa.int64(),
        "INT16": pa.int64(),
        "BOOL": pa.bool_(),
    }.get(upper, pa.string())
