"""Graph schema configuration API.

This module provides the core graph schema definitions:
- GraphNode: Configuration for a single node type
- GraphRelation: Configuration for relationships between nodes
- GraphSchema: Complete schema with validation and auto-deduction

The API automatically introspects Pydantic models to derive field paths
and relationships, reducing boilerplate and errors.
"""

from __future__ import annotations

import re
import types
import unicodedata
import uuid
import warnings
from collections.abc import Callable
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Union,
    get_args,
    get_origin,
    no_type_check,
)

from genai_tk.utils.hashing import buffer_digest
from loguru import logger
from pydantic import BaseModel, PrivateAttr, model_validator

if TYPE_CHECKING:
    from genai_graph.kg.manager import KgManager


# Unicode characters that should be normalised to ASCII for dedup keys.
# Maps various dash/hyphen codepoints to a plain ASCII hyphen-minus.
_HYPHEN_LIKE = re.compile(
    "["
    "\u2010"  # HYPHEN
    "\u2011"  # NON-BREAKING HYPHEN
    "\u2012"  # FIGURE DASH
    "\u2013"  # EN DASH
    "\u2014"  # EM DASH
    "\u2015"  # HORIZONTAL BAR
    "\u00ad"  # SOFT HYPHEN
    "\ufe63"  # SMALL HYPHEN-MINUS
    "\uff0d"  # FULLWIDTH HYPHEN-MINUS
    "]"
)


def _normalize_key(value: str) -> str:
    """Normalise a string for use as a deduplication key.

    Applies NFKC unicode normalisation and replaces common Unicode
    dash/hyphen variants with a plain ASCII hyphen-minus so that
    e.g. "Gérard Lassalle‑Valier" (U+2011) and
    "Gérard Lassalle-Valier" (U+002D) map to the same key.
    """
    value = unicodedata.normalize("NFKC", value)
    value = _HYPHEN_LIKE.sub("-", value)
    return value


def _find_embedded_field_for_class(parent_cls: type[BaseModel], embedded_cls: type[BaseModel]) -> str | None:
    """Return the field name on *parent_cls* that holds *embedded_cls*.

    The field may be typed directly as the embedded class, or wrapped inside
    Optional/Union or list containers, for example::

        financials: FinancialMetrics
        financials: FinancialMetrics | None
        financials: list[FinancialMetrics] | None
    """
    import types

    if not hasattr(parent_cls, "model_fields"):
        return None

    for field_name, field_info in parent_cls.model_fields.items():
        annotation = field_info.annotation
        origin = get_origin(annotation)
        args = get_args(annotation)

        candidate_types: list[type[BaseModel]] = []
        if origin is None:
            if isinstance(annotation, type):
                candidate_types = [annotation]
        elif origin is list:
            inner = args[0] if args else None
            if isinstance(inner, type):
                candidate_types = [inner]
        elif origin is Union or origin is types.UnionType:
            # Handle both typing.Union and types.UnionType (Python 3.10+)
            non_none_args = [t for t in args if t is not type(None)]
            for t in non_none_args:
                t_origin = get_origin(t)
                t_args = get_args(t)
                if t_origin is list and t_args:
                    inner = t_args[0]
                    if isinstance(inner, type):
                        candidate_types.append(inner)
                elif isinstance(t, type):
                    candidate_types.append(t)

        if any(ct is embedded_cls for ct in candidate_types):
            return field_name

    return None


def find_embedded_field_for_class(parent_cls: type[BaseModel], embedded_cls: type[BaseModel]) -> str | None:
    """Public API wrapper over :func:`_find_embedded_field_for_class`."""

    return _find_embedded_field_for_class(parent_cls, embedded_cls)


class GraphNode(BaseModel):
    """Simplified node configuration for graph creation.

    Only requires the essential information that cannot be auto-deduced:

    - Which Pydantic class to create nodes for (`node_class`)
    - Which field to use as primary key for display (`name_from`)
    - Optional customizations like additional structured `extra_classes`
    - Optional `index_fields` to enable embedding computation

    All field paths, excluded fields, list detection, and embedding dimensions
    are automatically determined — either by introspecting the Pydantic model
    structure or by the schema/factory layer — and are not part of the public
    constructor API.

    The ``extra_classes`` attribute is the unified configuration entry for
    additional structured properties attached to a node. It should contain
    regular Pydantic models referenced from the main ``node_class``, which
    are treated as embedded structs and stored as MAP/STRUCT properties on
    the node.
    """

    node_class: type[BaseModel]
    extra_classes: list[type[BaseModel]] = []

    model_config = {
        "populate_by_name": True,
    }
    name_from: str | Callable[[dict[str, Any], str], str]
    key_from: str | Callable[[dict[str, Any], str], str] = "AUTO_ID"
    description: str = ""
    index_fields: list[str | tuple[str, str]] = []
    table_name: str | None = None
    """Explicit database table name for this node.

    When set, overrides ``node_class.__name__`` as the Ladybug table name and
    merge/dedup key.  Use this when two different Pydantic classes should map
    to the same node table (e.g. a shared canonical type across projects) or
    to prevent accidental collisions between classes that happen to have the
    same ``__name__``.

    Leave as ``None`` (default) to use the class name.
    """

    # TODO: consider removing once orphan detection can infer reachability automatically.
    # Set to True for nodes from explicit mappings (Neo4j, etc.) to skip orphan warnings.
    explicitly_defined: bool = False

    # Internal state — set by GraphSchema during schema validation.
    # Not part of the public constructor API; use the read-only properties below.
    _field_paths: list[str] = PrivateAttr(default_factory=list)
    _is_list_at_paths: dict[str, bool] = PrivateAttr(default_factory=dict)
    _excluded_fields: set[str] = PrivateAttr(default_factory=set)
    _embedding_field_dimensions: dict[str, int] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Hook for future post-init logic (currently unused)."""
        # Kept for forwards-compatibility; no-op for now.
        return None

    @property
    def compute_embeddings(self) -> bool:
        """Return True when embedding computation is required (index_fields is non-empty)."""
        return bool(self.index_fields)

    @property
    def index_field_specs(self) -> list[tuple[str, str | None]]:
        """Normalised index field specs as (field_name, model_override_or_None) pairs.

        Plain strings yield (name, None) meaning use the default embedding model.
        Tuples yield (name, model_id) meaning use the specified model.
        """
        result: list[tuple[str, str | None]] = []
        for entry in self.index_fields:
            if isinstance(entry, tuple):
                result.append(entry)
            else:
                result.append((entry, None))
        return result

    @property
    def field_paths(self) -> list[str]:
        """All field paths where this node class appears in the root model (auto-deduced)."""
        return self._field_paths

    @property
    def is_list_at_paths(self) -> dict[str, bool]:
        """Whether the node is a list at each field path (auto-deduced)."""
        return self._is_list_at_paths

    @property
    def excluded_fields(self) -> set[str]:
        """Fields excluded from node properties because they are handled by relationships (auto-computed)."""
        return self._excluded_fields

    @property
    def embedding_field_dimensions(self) -> dict[str, int]:
        """Dimensions for pre-computed list[float] fields (set by factory, not by users)."""
        return self._embedding_field_dimensions

    @property
    def embedded_struct_classes(self) -> list[type[BaseModel]]:
        """Return Pydantic classes configured as embedded structs.

        These classes must be regular :class:`pydantic.BaseModel` subclasses
        referenced from :attr:`node_class` fields. They will be materialised
        as MAP/STRUCT properties on the parent node.
        """

        embedded: list[type[BaseModel]] = []
        for struct_cls in self.extra_classes:
            if isinstance(struct_cls, type) and issubclass(struct_cls, BaseModel):
                embedded.append(struct_cls)
        return embedded

    def get_name_value(self, data: dict[str, Any], node_type: str) -> str:
        """Get the node name value for a node instance.

        This computes the node name based on the ``name_from`` configuration.
        This becomes the primary 'name' field in the graph. Any original Pydantic
        'name' field is preserved as '_original_name'.

        The result is normalised via :func:`_normalize_key` so that
        equivalent Unicode representations produce the same display name.

        Args:
            data: Node data dictionary
            node_type: Name of the node type

        Returns:
            Node name value as string
        """

        value = data.get(self.name_from) if isinstance(self.name_from, str) else self.name_from(data, node_type)
        if not value:
            return f"{node_type}_unnamed"
        if isinstance(value, Enum):
            return value.name
        else:
            return _normalize_key(str(value))

    def get_key_value(self, data: dict[str, Any], node_type: str) -> str | None:
        """Get the primary key value for a node instance.

        This computes the primary key based on the ``key_from`` configuration.
        String keys are normalised via :func:`_normalize_key` so that
        equivalent Unicode representations produce the same dedup key.

        Args:
            data: Node data dictionary
            node_type: Name of the node type

        Returns:
            Primary key value as string, or None when a callable ``key_from``
            signals that this item should be skipped (no node created).
        """
        if self.key_from == "AUTO_ID":
            # AUTO_ID generates a unique UUID for each node
            # This ensures uniqueness without relying on any data field
            return str(uuid.uuid4())
        elif isinstance(self.key_from, str):
            # Use the specified field value
            value = data.get(self.key_from)
            if not value:
                available_fields = list(data.keys())[:10]  # Show first 10 fields
                fields_preview = ", ".join(available_fields)
                if len(data.keys()) > 10:
                    fields_preview += f" (and {len(data.keys()) - 10} more)"
                raise ValueError(
                    f"Key field '{self.key_from}' not found or empty in data for {node_type}. "
                    f"Available fields: {fields_preview}. "
                    f"Consider using key_from='AUTO_ID' if this field may be missing."
                )
            return _normalize_key(str(value))
        else:
            # key_from is a callable - compute the key value
            # Return None to signal "skip this item" (no node should be created)
            value = self.key_from(data, node_type)
            if value is None:
                return None
            if not value:
                raise ValueError(f"Computed key is empty for {node_type}")
            if isinstance(value, Enum):
                return value.name
            return _normalize_key(str(value))

    @property
    def label(self) -> str:
        """Return the canonical label for this node (table name or class name).

        Returns ``table_name`` when explicitly set, otherwise falls back to
        ``node_class.__name__``.
        """
        return self.table_name if self.table_name is not None else self.node_class.__name__

    def struct_field_names(self) -> list[str]:
        """Return the field names under which embedded structs are stored.

        For embedded structs configured via :attr:`extra_classes`, this
        returns the actual field names detected from the parent Pydantic
        model.
        """
        names: list[str] = []

        # Embedded structs: resolve field names on the parent model
        for embedded_cls in self.embedded_struct_classes:
            field_name = _find_embedded_field_for_class(self.node_class, embedded_cls)
            if field_name:
                names.append(field_name)

        return names


class GraphRelation(BaseModel):
    """Simplified relationship configuration.

    Only requires the essential relationship information:
    - Source and target node classes
    - Relationship name

    All field paths are automatically deduced from the Pydantic model structure.
    """

    from_node: GraphNode
    to_node: GraphNode
    name: str
    description: str = ""
    properties: dict[str, Any] | None = None  # Property name -> annotation/type

    # Auto-deduced attributes (populated during schema validation)
    field_paths: list[tuple[str, str]] = []  # (from_path, to_path) pairs

    @property
    def label(self) -> str:
        """Return the canonical label for this relationship (its name)."""
        return self.name

    @property
    def endpoints_label(self) -> str:
        """Return a human-readable description of the endpoints.

        Example: ``ReviewedOpportunity → HAS_RISK → RiskAnalysis``.
        """
        return f"{self.from_node.label} → {self.name} → {self.to_node.label}"

    def iter_field_paths(self) -> list[tuple[str, str]]:
        """Return a copy of the (from_path, to_path) pairs for this relation."""
        return list(self.field_paths)


class GraphSchema(BaseModel):
    """Complete graph schema with validation and auto-deduction capabilities.

    The root_model_class is optional for schemas that don't have a single root
    (e.g., Neo4j imports with multiple independent node types). When not set,
    schema auto-deduction features that rely on it will be skipped.
    """

    root_model_class: type[BaseModel] | None = None
    nodes: list[GraphNode]
    relations: list[GraphRelation]
    # Track all root model classes from merged schemas (for combined schemas)
    merged_root_classes: list[type[BaseModel]] = []

    model_config = {
        "populate_by_name": True,
    }

    # Validation results - must be instance variables, not class variables
    _model_field_map: dict[type[BaseModel], dict[str, Any]] = PrivateAttr(default_factory=dict)
    _warnings: list[str] = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def validate_and_deduce_schema(self) -> GraphSchema:
        """Validate schema coherence and auto-deduce missing information."""
        self._build_model_field_map()
        self._deduce_node_field_paths()
        self._deduce_relation_field_paths()
        self._compute_excluded_fields()
        self._validate_coherence(context=None)  # Context not available during model validation
        return self

    def _build_model_field_map(self) -> None:
        """Build a map of all reachable Pydantic model classes and their fields."""
        visited = set()

        def explore_model(model_class: type[BaseModel], path: str = ""):
            if model_class in visited:
                return
            visited.add(model_class)

            if not hasattr(model_class, "model_fields"):
                return

            self._model_field_map[model_class] = {}

            # Use get_type_hints to resolve ForwardRefs automatically
            try:
                from typing import get_type_hints

                type_hints = get_type_hints(model_class)
            except Exception:
                type_hints = {}

            for field_name, field_info in model_class.model_fields.items():
                field_path = f"{path}.{field_name}" if path else field_name
                # Use resolved type hint if available, otherwise use annotation
                annotation = type_hints.get(field_name, field_info.annotation)

                # Handle List[Model] annotations
                if get_origin(annotation) is list:
                    args = get_args(annotation)
                    inner_type = args[0] if args else None

                    # Handle ForwardRef by trying to resolve it to a real class
                    if inner_type is not None and hasattr(inner_type, "__forward_arg__"):
                        # Try to find the class by name in the model's module
                        try:
                            forward_name = inner_type.__forward_arg__
                            import sys

                            module = sys.modules.get(model_class.__module__)
                            if module and hasattr(module, forward_name):
                                resolved = getattr(module, forward_name)
                                if hasattr(resolved, "model_fields"):
                                    inner_type = resolved
                        except (AttributeError, KeyError):
                            pass

                    if inner_type is not None and hasattr(inner_type, "model_fields"):
                        self._model_field_map[model_class][field_name] = {
                            "path": field_path,
                            "type": inner_type,
                            "is_list": True,
                            "annotation": annotation,
                        }
                        explore_model(inner_type, field_path)
                    else:
                        self._model_field_map[model_class][field_name] = {
                            "path": field_path,
                            "type": annotation,
                            "is_list": True,
                            "annotation": annotation,
                        }
                # Handle Optional[Model] and Union[Model, None] - including types.UnionType from Python 3.10+
                elif get_origin(annotation) is Union or get_origin(annotation) is types.UnionType:
                    args = get_args(annotation)
                    non_none_args = [arg for arg in args if arg is not type(None)]
                    # Unwrap Optional[List[T]] or Union[List[T], None]
                    if len(non_none_args) == 1 and get_origin(non_none_args[0]) is list:
                        inner_args = get_args(non_none_args[0])
                        inner = inner_args[0] if inner_args else None

                        # Handle ForwardRef in Optional[List[ForwardRef]]
                        if inner is not None and hasattr(inner, "__forward_arg__"):
                            try:
                                forward_name = inner.__forward_arg__
                                import sys

                                module = sys.modules.get(model_class.__module__)
                                if module and hasattr(module, forward_name):
                                    inner = getattr(module, forward_name)
                            except (AttributeError, KeyError):
                                pass

                        if inner is not None and hasattr(inner, "model_fields"):
                            self._model_field_map[model_class][field_name] = {
                                "path": field_path,
                                "type": inner,
                                "is_list": True,
                                "annotation": annotation,
                            }
                            explore_model(inner, field_path)
                            continue

                    # Unwrap Optional[T]
                    if len(non_none_args) == 1 and hasattr(non_none_args[0], "model_fields"):
                        self._model_field_map[model_class][field_name] = {
                            "path": field_path,
                            "type": non_none_args[0],
                            "is_list": False,
                            "annotation": annotation,
                        }
                        explore_model(non_none_args[0], field_path)
                    else:
                        self._model_field_map[model_class][field_name] = {
                            "path": field_path,
                            "type": annotation,
                            "is_list": False,
                            "annotation": annotation,
                        }
                # Handle ForwardRef annotations
                elif hasattr(annotation, "__forward_arg__"):
                    # Try to resolve ForwardRef to actual class
                    try:
                        forward_name = annotation.__forward_arg__  # type: ignore
                        import sys

                        module = sys.modules.get(model_class.__module__)
                        if module and hasattr(module, forward_name):
                            resolved_type = getattr(module, forward_name)
                            if hasattr(resolved_type, "model_fields"):
                                self._model_field_map[model_class][field_name] = {
                                    "path": field_path,
                                    "type": resolved_type,
                                    "is_list": False,
                                    "annotation": annotation,
                                }
                                explore_model(resolved_type, field_path)
                            else:
                                # Resolved but not a model
                                self._model_field_map[model_class][field_name] = {
                                    "path": field_path,
                                    "type": annotation,
                                    "is_list": False,
                                    "annotation": annotation,
                                }
                        else:
                            # Could not resolve ForwardRef
                            self._model_field_map[model_class][field_name] = {
                                "path": field_path,
                                "type": annotation,
                                "is_list": False,
                                "annotation": annotation,
                            }
                    except (AttributeError, KeyError):
                        self._model_field_map[model_class][field_name] = {
                            "path": field_path,
                            "type": annotation,
                            "is_list": False,
                            "annotation": annotation,
                        }
                # Handle direct Model references
                elif hasattr(annotation, "model_fields"):
                    self._model_field_map[model_class][field_name] = {
                        "path": field_path,
                        "type": annotation,
                        "is_list": False,
                        "annotation": annotation,
                    }
                    explore_model(annotation, field_path)  # type: ignore
                else:
                    # Primitive field
                    self._model_field_map[model_class][field_name] = {
                        "path": field_path,
                        "type": annotation,
                        "is_list": False,
                        "annotation": annotation,
                    }

        # For combined schemas, explore ALL root model classes
        root_classes_to_explore = ([self.root_model_class] if self.root_model_class else []) + self.merged_root_classes
        for root_class in root_classes_to_explore:
            explore_model(root_class)

    def _deduce_node_field_paths(self) -> None:
        """Auto-deduce field paths for all node configurations."""
        for node_config in self.nodes:
            node_config._field_paths = []
            node_config._is_list_at_paths = {}

            # Special case: root model (only if root_model_class is set)
            if self.root_model_class is not None and node_config.node_class == self.root_model_class:
                node_config._field_paths = [""]  # Empty path = root
                node_config._is_list_at_paths[""] = False
                continue

            # Find all paths where this class appears
            for _model_class, fields in self._model_field_map.items():
                for _field_name, field_info in fields.items():
                    field_type = field_info["type"]
                    # Match by __name__ to support extended types (e.g.,
                    # common_nodes.Customer extends BamlCustomer but both
                    # share __name__ == "Customer" for Kuzu table dedup).
                    if field_type == node_config.node_class or (
                        hasattr(field_type, "__name__") and field_type.__name__ == node_config.node_class.__name__
                    ):
                        path = field_info["path"]
                        is_list = field_info["is_list"]
                        node_config.field_paths.append(path)
                        node_config.is_list_at_paths[path] = is_list

    def _deduce_relation_field_paths(self) -> None:
        """Auto-deduce field paths for all relationship configurations."""

        for relation_config in self.relations:
            # Skip deduction if field paths are already explicitly provided
            if relation_config.field_paths:
                continue

            relation_config.field_paths = []

            # Find all possible paths between from_node and to_node.
            # relation_config.from_node IS the GraphNode (same object as in self.nodes),
            # so its field_paths are already populated by _deduce_node_field_paths.
            from_node_paths = relation_config.from_node.field_paths
            to_node_paths = relation_config.to_node.field_paths

            # Find all valid connections
            candidate_paths = []
            for from_path in from_node_paths:
                for to_path in to_node_paths:
                    if self._is_valid_relationship_path(from_path, to_path, relation_config):
                        candidate_paths.append((from_path, to_path))

            # Sort by path simplicity/directness - prefer direct parent-child relationships
            candidate_paths.sort(key=lambda p: self._path_complexity_score(p[0], p[1]))

            # Use the simplest path(s)
            if candidate_paths:
                relation_config.field_paths = [candidate_paths[0]]

                # Warn if multiple valid paths exist
                if len(candidate_paths) > 1:
                    from_label = relation_config.from_node.label
                    to_label = relation_config.to_node.label
                    chosen = f"{candidate_paths[0][0] or '(root)'} → {candidate_paths[0][1] or '(root)'}"
                    alternatives = "; ".join([f"{p[0] or '(root)'} → {p[1] or '(root)'}" for p in candidate_paths[1:]])
                    warning_msg = (
                        f"Multiple valid paths found for {relation_config.name} ({from_label} → {to_label}). "
                        f"Using: {chosen}. Alternatives: {alternatives}. "
                        f"Specify field_paths=[...] explicitly if this is incorrect."
                    )
                    # Store in _warnings list so it gets picked up by validate_with_context
                    self._warnings.append(warning_msg)

    def _path_complexity_score(self, from_path: str, to_path: str) -> tuple[int, int, int, int]:
        """Calculate a complexity score for a relationship path.

        Returns a tuple ``(containment, path_depth, nesting_depth,
        combined_length)`` where:

        - containment: **0** when ``to_path`` is nested inside ``from_path``
          (i.e. a true parent→child containment such as
          ``customer → customer.employees``), **1** otherwise.  This is the
          **most important** criterion — a containment relationship is always
          preferred over a lateral/sibling one because it means the target
          objects are directly *owned* by the source node.
        - path_depth: sum of from/to path depths (fewer dots = simpler).
        - nesting_depth: how deeply nested the relationship is.
        - combined_length: total character length as final tiebreaker.

        Lower scores are preferred (simpler, more direct paths).
        """
        # -- PRIMARY CRITERION: containment ---------------------------------
        # ``to_path`` starts with ``from_path.`` ⇒ true ownership.
        # When ``from_path`` is root (""), every ``to_path`` is "contained",
        # so we don't penalise that case.
        is_contained = from_path == "" or to_path.startswith(from_path + ".")
        containment = 0 if is_contained else 1

        # -- SECONDARY: path depth ------------------------------------------
        from_depth = from_path.count(".") if from_path else 0
        to_depth = to_path.count(".") if to_path else 0
        path_depth = from_depth + to_depth

        # -- TERTIARY: nesting depth ----------------------------------------
        if is_contained and from_path:
            nesting_depth = to_depth
        else:
            from_parts = from_path.split(".") if from_path else []
            to_parts = to_path.split(".") if to_path else []
            common_len = 0
            for i in range(min(len(from_parts), len(to_parts))):
                if from_parts[i] == to_parts[i]:
                    common_len = i + 1
                else:
                    break
            nesting_depth = common_len

        combined_length = len(from_path) + len(to_path)

        return (containment, path_depth, nesting_depth, combined_length)

    def _is_valid_relationship_path(self, from_path: str, to_path: str, relation_config: GraphRelation) -> bool:
        """Check if a relationship path makes logical sense.

        Valid relationships include:
        1. From root to anything
        2. Parent-child relationships (one path contains the other)
        3. Sibling relationships (both are direct children of the same parent, including root)
        """
        # Root to anything is valid
        if from_path == "":
            return True

        # Check if to_path is a sub-path of from_path or vice versa
        if to_path.startswith(from_path + ".") or from_path.startswith(to_path + "."):
            return True

        # Check if they share a common parent path (including root as parent)
        from_parts = from_path.split(".")
        to_parts = to_path.split(".")

        # Find common prefix
        common_len = 0
        for i in range(min(len(from_parts), len(to_parts))):
            if from_parts[i] == to_parts[i]:
                common_len = i + 1
            else:
                break

        # They're siblings if:
        # - They share the same parent (common_len > 0), OR
        # - They're both direct children of root (both have depth 1, common_len = 0)
        if common_len > 0:
            return True

        # Both are direct children of root (siblings at root level)
        return bool(len(from_parts) == 1 and len(to_parts) == 1)

    def _compute_excluded_fields(self) -> None:
        """Compute which fields should be excluded from each node based on relationships.

        Notes:
            Relationship targets (other nodes) are excluded so they are not
            materialised twice.
        """
        for node_config in self.nodes:
            excluded_fields = set()

            # Exclude fields with p_*_ pattern (these become edge properties)
            if hasattr(node_config.node_class, "model_fields"):
                for field_name in node_config.node_class.model_fields:
                    if field_name.startswith("p_") and field_name.endswith("_"):
                        excluded_fields.add(field_name)

            # Find all fields that are handled by relationships
            for relation_config in self.relations:
                if relation_config.from_node.label == node_config.label:
                    # Fields that point to other nodes should be excluded
                    for from_path, to_path in relation_config.field_paths:
                        # Extract the field name from the path
                        # Only exclude if this relationship applies to this node's field_path
                        for node_field_path in node_config.field_paths:
                            if to_path and "." in to_path:
                                if from_path == "" and node_field_path == "":
                                    # Root node excluding direct field
                                    field_name = to_path.split(".")[0]
                                    excluded_fields.add(field_name)
                                elif from_path == node_field_path and to_path.startswith(from_path + "."):
                                    # from_path matches this node's field path
                                    relative_path = to_path[len(from_path) + 1 :]
                                    field_name = relative_path.split(".")[0]
                                    excluded_fields.add(field_name)
                            elif (
                                to_path
                                and "." not in to_path
                                and ((from_path == "" and node_field_path == "") or from_path == node_field_path)
                            ):
                                # Direct field reference
                                excluded_fields.add(to_path)

            # Note: legacy `embed_in_parent` behaviour has been removed. All
            # additional structured data should now be modelled via
            # ``extra_classes`` on the parent node and is never flattened into
            # scalar columns here.

            node_config._excluded_fields = excluded_fields

    @no_type_check  # Avoid type-checking *ANY* methods or attributes of this class.
    def _validate_coherence(self, context: KgManager | None = None) -> None:
        """Validate that the schema configuration is coherent with the Pydantic model.

        Args:
            context: Optional KgManager for collecting warnings
        """
        warnings_list = []

        # Check that all referenced node labels in relationships have node configurations
        referenced_labels = {rel.from_node.label for rel in self.relations} | {
            rel.to_node.label for rel in self.relations
        }
        configured_labels = {node.label for node in self.nodes}
        missing_labels = referenced_labels - configured_labels

        for label in missing_labels:
            warnings_list.append(f"Class {label} is referenced in relationships but has no GraphNode")

        # Check for duplicate node labels — two nodes sharing the same table name is an error.
        seen_labels: dict[str, str] = {}
        for node in self.nodes:
            lbl = node.label
            cls_name = node.node_class.__name__
            if lbl in seen_labels:
                existing_cls = seen_labels[lbl]
                if existing_cls != cls_name:
                    warnings_list.append(
                        f"Two different node classes share the label '{lbl}': "
                        f"'{existing_cls}' and '{cls_name}'. "
                        f"Set `table_name` on one of them to resolve the collision."
                    )
            else:
                seen_labels[lbl] = cls_name

        # Check for duplicate relationships between the same node pair
        relation_pairs: dict[tuple[str, str], list[str]] = {}
        for relation in self.relations:
            key = (relation.from_node.label, relation.to_node.label)
            if key in relation_pairs:
                relation_pairs[key].append(relation.name)
            else:
                relation_pairs[key] = [relation.name]

        for (from_label, to_label), names in relation_pairs.items():
            if len(names) > 1:
                warnings_list.append(
                    f"Multiple relationships defined between {from_label} and {to_label}: {', '.join(names)}"
                )

        # Warn when we have node classes that never appear in the reachable
        # model structure (likely orphan configurations).
        # For combined schemas, also check if node is a root in any merged schema
        all_root_classes = ({self.root_model_class} if self.root_model_class else set()) | set(self.merged_root_classes)

        for node in self.nodes:
            # Robustly skip the root node (by class or by field_paths)
            is_root_node = node.node_class in all_root_classes or node.field_paths == [""]
            # Never warn for the root node, even if field_paths is empty or [""]
            if is_root_node:
                continue
            # Skip nodes that are explicitly defined (e.g., from Neo4j mappings)
            # These nodes don't need field paths since they come from explicit definitions
            if node.explicitly_defined:
                continue
            # When there is no root model at all the concept of "orphaned" does not apply
            # (e.g. SimilarityFactory schemas have root_model_class=None by design)
            if not all_root_classes:
                continue
            # Only warn if not root node and field_paths is empty or None
            if not node.field_paths:
                warnings_list.append(
                    f"No field paths found for {node.node_class.__name__} in the root model structure; "
                    "this node may be orphaned."
                )

        # # Check that field paths were found for relationships
        # for relation in self.relations:
        #     if not relation.field_paths:
        #         warnings_list.append(
        #             f"No valid field paths found for relationship {relation.name} "
        #             f"between {relation.from_node.label} and {relation.to_node.label}"
        #         )

        # Validate embedded field configurations (MAP/STRUCT support)
        import types
        from typing import get_args, get_origin

        for node in self.nodes:
            if not node.embedded_struct_classes:
                continue

            model_fields = getattr(node.node_class, "model_fields", {})
            for embedded_class in node.embedded_struct_classes:
                field_name = _find_embedded_field_for_class(node.node_class, embedded_class)
                if not field_name:
                    warnings_list.append(
                        f"Embedded class {embedded_class.__name__} is not referenced on "
                        f"{node.node_class.__name__}; it will not be materialised."
                    )
                    continue
                # Check that the field exists on the parent class
                if field_name not in model_fields:
                    warnings_list.append(
                        f"Embedded field '{field_name}' is not defined on class {node.node_class.__name__}"
                    )
                    continue

                annotation = model_fields[field_name].annotation
                origin = get_origin(annotation)
                args = get_args(annotation)

                # Unwrap Optional/Union
                candidate_types = []
                if origin is None:
                    candidate_types = [annotation]
                elif origin is list:
                    # Embedded should be a single object, not a list
                    inner = args[0] if args else None
                    if inner is not None:
                        candidate_types = [inner]
                elif origin is Union or origin is types.UnionType:
                    candidate_types = [t for t in args if t is not type(None)]

                if embedded_class not in candidate_types:
                    warnings_list.append(
                        "Embedded field '"
                        f"{field_name}' on class {node.node_class.__name__} has incompatible type "
                        f"{annotation!r}; expected {embedded_class.__name__} or Optional[{embedded_class.__name__}]"
                    )

        # Check that index_fields are stored as direct node columns, not as relationship
        # properties (p_<field>_ pattern).  A field named `foo` that exists on the model
        # only as `p_foo_` is an edge property and will never appear in item_data, so the
        # embedding column will always be NULL.
        for node in self.nodes:
            if not node.index_fields:
                continue
            model_fields = getattr(node.node_class, "model_fields", {})
            for field_name in node.index_fields:
                if field_name not in model_fields:
                    rel_prop_name = f"p_{field_name}_"
                    if rel_prop_name in model_fields:
                        warnings_list.append(
                            f"{node.node_class.__name__}.index_fields contains '{field_name}', "
                            f"but that field is defined as a relationship property ('{rel_prop_name}'). "
                            "Relationship properties are stored on edges, not on the node itself, "
                            "so the embedding column will always be NULL. "
                            f"Remove '{field_name}' from index_fields or promote it to a regular node field."
                        )

        # Extend warnings (don't replace, to preserve warnings from path deduction)
        self._warnings.extend(warnings_list)

        # Add warnings to context if provided
        if context:
            for warning_msg in warnings_list:
                context.add_warning(f"Schema validation: {warning_msg}")

        # Emit warnings
        for warning_msg in warnings_list:
            warnings.warn(f"Graph schema validation: {warning_msg}", UserWarning, stacklevel=2)

    def get_warnings(self) -> list[str]:
        """Get all validation warnings."""
        return self._warnings.copy()

    def fingerprint(self) -> str:
        """Compute a stable hex digest of this schema's structure.

        Captures node types, key fields, extra classes, relationship
        endpoints and properties — i.e. everything that would change the
        shape of the Kuzu tables.  The digest is deterministic for the
        same logical schema, independent of Python object identity.

        Returns:
            Hex string (xxh3_64) of the schema structure.
        """
        parts: list[str] = []

        # Root model class
        if self.root_model_class is not None:
            parts.append(f"root:{self.root_model_class.__name__}")

        # Nodes — sorted by class name for determinism
        for node in sorted(self.nodes, key=lambda n: n.node_class.__name__):
            name_from = node.name_from if isinstance(node.name_from, str) else "<callable>"
            key_from = node.key_from if isinstance(node.key_from, str) else "<callable>"
            extras = ",".join(sorted(c.__name__ for c in node.extra_classes))
            idx = ",".join(sorted(node.index_fields))
            parts.append(
                f"node:{node.node_class.__name__}|name={name_from}|key={key_from}"
                f"|extras={extras}|idx={idx}|desc={node.description}"
            )

        # Relations — sorted by (from, name, to)
        for rel in sorted(self.relations, key=lambda r: (r.from_node.label, r.name, r.to_node.label)):
            props = ",".join(sorted(rel.properties.keys())) if rel.properties else ""
            parts.append(
                f"rel:{rel.from_node.label}->{rel.name}->{rel.to_node.label}|props={props}|desc={rel.description}"
            )

        combined = "\n".join(parts)
        return buffer_digest(combined.encode())

    def validate_with_context(self, context: KgManager) -> None:
        """Re-run coherence validation and collect warnings into a KgManager.

        This method allows re-validating the schema after initial construction
        to collect warnings into the central KgManager singleton.

        Args:
            context: KgManager for collecting warnings
        """
        # First add any warnings accumulated during schema construction (e.g., from path deduction)
        for warning_msg in self._warnings:
            context.add_warning(f"Schema: {warning_msg}")

        # Then run coherence validation which may add more warnings
        self._validate_coherence(context=context)

    def index_fields_in_vector_store(self, model_instance: BaseModel, embeddings_store_config: str) -> None:
        """Index specified fields from model instance in a vector store.

        Args:
            model_instance: Instance of the root model
            embeddings_store_config: Config name for the EmbeddingsStore
        """
        from genai_tk.core.embeddings_store import EmbeddingsStore
        from langchain_core.documents import Document

        # Create embeddings store
        embeddings_store = EmbeddingsStore.create_from_config(embeddings_store_config)
        vector_store = embeddings_store.get()

        documents: list[Document] = []

        # Iterate through nodes with index_fields
        for node_config in self.nodes:
            if not node_config.index_fields:
                continue

            # Get the model instance data
            for field_path in node_config.field_paths:
                # Extract data at the field path
                data = self._get_field_by_path(model_instance, field_path) if field_path else model_instance

                if data is None:
                    continue

                # Handle list of instances
                items = data if isinstance(data, list) else [data]

                for item in items:
                    if item is None:
                        continue

                    # Extract indexed fields
                    for field_name in node_config.index_fields:
                        if not hasattr(item, field_name):
                            continue

                        field_value = getattr(item, field_name)
                        if field_value is None:
                            continue

                        # Convert to string for indexing
                        content = str(field_value)

                        # Get primary key for metadata
                        primary_key = getattr(item, node_config.key, "unknown")

                        # Create document
                        doc = Document(
                            page_content=content,
                            metadata={
                                "node_type": node_config.node_class.__name__,
                                "field_name": field_name,
                                "primary_key": str(primary_key),
                                "field_path": field_path or "root",
                            },
                        )
                        documents.append(doc)

        # Add documents to vector store
        if documents:
            vector_store.add_documents(documents)

    def _get_field_by_path(self, obj: Any, path: str) -> Any:
        """Get a field by dot-separated path."""
        if not path:
            return obj

        try:
            current = obj
            for part in path.split("."):
                if hasattr(current, part):
                    current = getattr(current, part)
                elif isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
            return current
        except (AttributeError, KeyError, TypeError):
            return None

    def print_schema_summary(self) -> None:
        """Print a summary of the deduced schema configuration."""
        root_name = self.root_model_class.__name__ if self.root_model_class else "(no root)"
        logger.debug(f"Graph Schema Summary for {root_name}")

        # Nodes
        logger.debug("Node Configurations.")
        for node in self.nodes:
            paths_str = ", ".join(node.field_paths) if node.field_paths else "ROOT"
            excluded_str = ", ".join(sorted(node.excluded_fields)) if node.excluded_fields else "None"
            logger.debug(f"  {node.node_class.__name__}: key={node.key}, paths={paths_str}, excluded={excluded_str}")

        # Relations
        logger.debug("Relationship Configurations.")
        for relation in self.relations:
            from_to = f"{relation.from_node.label} → {relation.to_node.label}"
            paths_str = (
                "; ".join([f"{fp} → {tp}" for fp, tp in relation.field_paths]) if relation.field_paths else "None"
            )
            logger.debug(f"  {relation.name}: {from_to}, paths={paths_str}")

        # Warnings
        if self._warnings:
            logger.warning("Schema warnings.")
            for warning in self._warnings:
                logger.warning(f"  {warning}")
