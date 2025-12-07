"""Simplified graph schema configuration API.

This module provides a refactored approach to defining graph schemas with minimal
configuration required from users. It automatically introspects Pydantic models
to derive field paths and relationships, reducing boilerplate and errors.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    Any,
    Callable,
    Self,
    Union,
    get_args,
    get_origin,
    no_type_check,
)

from pydantic import BaseModel, PrivateAttr, model_validator


class ExtraFields(BaseModel, ABC):
    @classmethod
    @abstractmethod
    def get_data(cls, context: dict | None) -> Self | None:
        """Return an instance (or None) containing extra structured data.

        The `contex` parameter is a  dictionary provided by
        the extractor and can contain keys like `root_model`, `item`,
        `item_data` and `source_key` to help populate the extra fields.
        """
        ...


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
            non_none_args = [t for t in args if t is not type(None)]  # noqa: E721
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


class GraphNode(BaseModel):
    """Simplified node configuration for graph creation.

    Only requires the essential information that cannot be auto-deduced:

    - Which Pydantic class to create nodes for (`node_class`)
    - Which field to use as primary key for display (`name_from`)
    - Optional customizations like additional structured `extra_classes`

    All field paths, excluded fields, and list detection are automatically
    determined by introspecting the Pydantic model structure.

    The ``extra_classes`` attribute is the unified configuration entry for
    additional structured properties attached to a node. It can contain
    either:

    * ``ExtraFields`` subclasses (for synthetic/derived data such as
      ``FileMetadata`` or ``WinLoss``) – values are computed via
      :meth:`ExtraFields.get_data` and inserted as nested maps.
    * Regular Pydantic models referenced from the main ``node_class``,
      which are treated as embedded structs and stored as MAP/STRUCT
      properties on the node.
    """

    node_class: type[BaseModel]
    extra_classes: list[type[BaseModel]] = []

    model_config = {
        "populate_by_name": True,
    }
    name_from: str | Callable[[dict[str, Any], str], str]
    description: str = ""
    # Can be a field name or a callable similar to ``name_from``
    deduplication_key: str | Callable[[dict[str, Any], str], Any] | None = None
    index_fields: list[str] = []

    # Auto-deduced attributes (populated during schema validation)
    field_paths: list[str] = []  # All paths where this class appears in the root model
    is_list_at_paths: dict[str, bool] = {}  # Whether it's a list at each path
    excluded_fields: set[str] = set()  # Auto-computed based on relationships

    def model_post_init(self, __context: Any) -> None:  # noqa: D401
        """Hook for future post-init logic (currently unused)."""
        # Kept for forwards-compatibility; no-op for now.
        return None

    @property
    def extra_field_classes(self) -> list[type[ExtraFields]]:
        """Return configured ``ExtraFields`` subclasses for this node.

        These classes are used to compute synthetic/derived structured
        properties that are stored directly on the node (e.g. ``file_metadata``).
        """

        extras: list[type[ExtraFields]] = []
        for struct_cls in self.extra_classes:
            try:
                if issubclass(struct_cls, ExtraFields):
                    extras.append(struct_cls)  # type: ignore[arg-type]
            except TypeError:
                # Not a class or not suitable for issubclass; ignore
                continue
        return extras

    @property
    def embedded_struct_classes(self) -> list[type[BaseModel]]:
        """Return non-``ExtraFields`` Pydantic classes used as embedded structs."""

        embedded: list[type[BaseModel]] = []
        for struct_cls in self.extra_classes:
            try:
                if issubclass(struct_cls, ExtraFields):
                    continue
            except TypeError:
                continue
            if isinstance(struct_cls, type) and issubclass(struct_cls, BaseModel):
                embedded.append(struct_cls)
        return embedded

    @property
    def key(self) -> str:
        """Get the primary key field name.

        All nodes use 'id' as the primary key (UUID).
        """
        return "id"

    def get_name_value(self, data: dict[str, Any], node_type: str) -> str:
        """Get the _name value for a node instance.

        Args:
            data: Node data dictionary
            node_type: Name of the node type

        Returns:
            Name value as string
        """

        if isinstance(self.name_from, str):
            value = data.get(self.name_from)
        else:
            # name_from is a callable
            value = self.name_from(data, node_type)
        if not value:
            return f"{node_type}_unnamed"
        if isinstance(value, Enum):
            return value.name
        else:
            return str(value)

    def get_dedup_value(self, data: dict[str, Any], node_type: str) -> str | None:
        """Get the value used for deduplication for a node instance.

        When ``deduplication_key`` is not set, this falls back to the
        computed ``_name`` so that all downstream components can always
        rely on a single canonical dedup value.
        """

        # Default: use the name as dedup key
        if not self.deduplication_key:
            return self.get_name_value(data, node_type)

        if isinstance(self.deduplication_key, str):
            value = data.get(self.deduplication_key)
        else:
            # deduplication_key is a callable
            value = self.deduplication_key(data, node_type)

        if value is None or value == "":
            return None
        if isinstance(value, Enum):
            return value.name
        return str(value)

    @property
    def label(self) -> str:
        """Return the canonical label for this node (its class name)."""
        return self.node_class.__name__

    def struct_field_names(self) -> list[str]:
        """Return the field names under which extra structs are stored.

        ExtraFields subclasses are exposed using their snake_case class name,
        while embedded structs use the actual field names detected from the
        parent Pydantic model.
        """
        names: list[str] = []

        # ExtraFields-based structs
        for extra_cls in self.extra_field_classes:
            struct_name = "".join(["_" + c.lower() if c.isupper() else c for c in extra_cls.__name__]).lstrip("_")
            names.append(struct_name)

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

    from_node: type[BaseModel]
    to_node: type[BaseModel]
    name: str
    description: str = ""

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
        return f"{self.from_node.__name__} → {self.name} → {self.to_node.__name__}"

    def iter_field_paths(self) -> list[tuple[str, str]]:
        """Return a copy of the (from_path, to_path) pairs for this relation."""
        return list(self.field_paths)


class GraphSchema(BaseModel):
    """Complete graph schema with validation and auto-deduction capabilities."""

    root_model_class: type[BaseModel]
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
    def validate_and_deduce_schema(self) -> "GraphSchema":
        """Validate schema coherence and auto-deduce missing information."""
        self._build_model_field_map()
        self._deduce_node_field_paths()
        self._deduce_relation_field_paths()
        self._compute_excluded_fields()
        self._validate_coherence()
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
                # Handle Optional[Model] and Union[Model, None]
                elif get_origin(annotation) is Union:
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
        root_classes_to_explore = [self.root_model_class] + self.merged_root_classes
        for root_class in root_classes_to_explore:
            explore_model(root_class)

    def _deduce_node_field_paths(self) -> None:
        """Auto-deduce field paths for all node configurations."""
        for node_config in self.nodes:
            node_config.field_paths = []
            node_config.is_list_at_paths = {}

            # Special case: root model
            if node_config.node_class == self.root_model_class:
                node_config.field_paths = [""]  # Empty path = root
                node_config.is_list_at_paths[""] = False
                continue

            # Find all paths where this class appears
            for _model_class, fields in self._model_field_map.items():
                for _field_name, field_info in fields.items():
                    if field_info["type"] == node_config.node_class:
                        path = field_info["path"]
                        is_list = field_info["is_list"]
                        node_config.field_paths.append(path)
                        node_config.is_list_at_paths[path] = is_list

    def _deduce_relation_field_paths(self) -> None:
        """Auto-deduce field paths for all relationship configurations."""
        from loguru import logger

        for relation_config in self.relations:
            # Skip deduction if field paths are already explicitly provided
            if relation_config.field_paths:
                continue

            relation_config.field_paths = []

            # Find all possible paths between from_node and to_node
            from_node_paths = self._get_node_paths(relation_config.from_node)
            to_node_paths = self._get_node_paths(relation_config.to_node)

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
                    from_label = relation_config.from_node.__name__
                    to_label = relation_config.to_node.__name__
                    chosen = f"{candidate_paths[0][0] or '(root)'} → {candidate_paths[0][1] or '(root)'}"
                    alternatives = "; ".join([f"{p[0] or '(root)'} → {p[1] or '(root)'}" for p in candidate_paths[1:]])
                    logger.warning(
                        f"Multiple valid paths found for {relation_config.name} ({from_label} → {to_label}). "
                        f"Using: {chosen}. Alternatives: {alternatives}. "
                        f"Specify field_paths=[...] explicitly if this is incorrect."
                    )

    def _get_node_paths(self, node_class: type[BaseModel]) -> list[str]:
        """Get all field paths for a given node class."""
        node_config = next((n for n in self.nodes if n.node_class == node_class), None)
        return node_config.field_paths if node_config else []

    def _path_complexity_score(self, from_path: str, to_path: str) -> tuple[int, int, int, int]:
        """Calculate a complexity score for a relationship path.

        Returns a tuple (path_depth, is_nested, nesting_depth, combined_length) where:
        - path_depth: sum of path depths (fewer dots = simpler) - MOST IMPORTANT
        - is_nested: 0 if sibling relationship, 1 if one is nested in the other
        - nesting_depth: how deeply nested the relationship is
        - combined_length: total character length

        Lower scores are preferred (simpler, more direct paths).
        The primary criterion is minimizing total path depth (preferring simple, direct fields).
        """
        # Calculate path depth (number of dots = nesting level) - PRIMARY CRITERION
        from_depth = from_path.count(".") if from_path else 0
        to_depth = to_path.count(".") if to_path else 0
        path_depth = from_depth + to_depth

        # Check if paths are siblings (share same parent) vs nested (one contains the other)
        # Sibling relationships (e.g., "opportunity" → "lead") are simpler than nested ones
        is_direct_child = to_path.startswith(from_path + ".") if from_path else False
        is_direct_parent = from_path.startswith(to_path + ".") if to_path else False
        is_nested = 1 if (is_direct_child or is_direct_parent) else 0

        # Calculate nesting depth (how many levels deep the relationship goes)
        # For siblings, this is the depth of their common parent
        # For nested relationships, this is the depth of the deeper path
        if is_nested:
            nesting_depth = max(from_depth, to_depth)
        else:
            # For siblings, find common parent depth
            from_parts = from_path.split(".") if from_path else []
            to_parts = to_path.split(".") if to_path else []
            common_len = 0
            for i in range(min(len(from_parts), len(to_parts))):
                if from_parts[i] == to_parts[i]:
                    common_len = i + 1
                else:
                    break
            nesting_depth = common_len

        # Total length as final tiebreaker
        combined_length = len(from_path) + len(to_path)

        return (path_depth, is_nested, nesting_depth, combined_length)

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
        if len(from_parts) == 1 and len(to_parts) == 1:
            return True

        return False

    def _compute_excluded_fields(self) -> None:
        """Compute which fields should be excluded from each node based on relationships.

        Notes:
            Relationship targets (other nodes) are excluded so they are not
            materialised twice. Additional structured data modelled via
            ``extra_classes`` is never excluded here; it is always represented
            as MAP/STRUCT properties on the parent node.
        """
        for node_config in self.nodes:
            excluded_fields = set()

            # Exclude fields with p_*_ pattern (these become edge properties)
            if hasattr(node_config.node_class, "model_fields"):
                for field_name in node_config.node_class.model_fields.keys():
                    if field_name.startswith("p_") and field_name.endswith("_"):
                        excluded_fields.add(field_name)

            # Find all fields that are handled by relationships
            for relation_config in self.relations:
                if relation_config.from_node == node_config.node_class:
                    # Fields that point to other nodes should be excluded
                    for from_path, to_path in relation_config.field_paths:
                        # Extract the field name from the path
                        if to_path and "." in to_path:
                            if from_path == "":
                                # Root node excluding direct field
                                field_name = to_path.split(".")[0]
                                excluded_fields.add(field_name)
                            else:
                                # Get relative field name
                                if to_path.startswith(from_path + "."):
                                    relative_path = to_path[len(from_path) + 1 :]
                                    field_name = relative_path.split(".")[0]
                                    excluded_fields.add(field_name)
                        elif to_path and "." not in to_path:
                            # Direct field reference
                            if from_path == "":
                                excluded_fields.add(to_path)

            # Note: legacy `embed_in_parent` behaviour has been removed. All
            # additional structured data should now be modelled via
            # ``extra_classes`` on the parent node and is never flattened into
            # scalar columns here.

            node_config.excluded_fields = excluded_fields

    @no_type_check  # Avoid type-checking *ANY* methods or attributes of this class.
    def _validate_coherence(self) -> None:
        """Validate that the schema configuration is coherent with the Pydantic model."""
        warnings_list = []

        # Check that all referenced classes in relationships have node configurations
        referenced_classes = set()
        for relation in self.relations:
            referenced_classes.add(relation.from_node)
            referenced_classes.add(relation.to_node)

        configured_classes = {node.node_class for node in self.nodes}
        missing_classes = referenced_classes - configured_classes

        if missing_classes:
            for cls in missing_classes:
                warnings_list.append(f"Class {cls.__name__} is referenced in relationships but has no GraphNode")

        # Check for duplicate relationships between the same classes
        relation_pairs = {}
        for relation in self.relations:
            key = (relation.from_node, relation.to_node)
            if key in relation_pairs:
                relation_pairs[key].append(relation.name)
            else:
                relation_pairs[key] = [relation.name]

        for (from_cls, to_cls), names in relation_pairs.items():
            if len(names) > 1:
                warnings_list.append(
                    f"Multiple relationships defined between {from_cls.__name__} and {to_cls.__name__}: {', '.join(names)}"
                )

        # Warn when we have node classes that never appear in the reachable
        # model structure (likely orphan configurations).
        # For combined schemas, also check if node is a root in any merged schema
        all_root_classes = {self.root_model_class} | set(self.merged_root_classes)

        for node in self.nodes:
            # Robustly skip the root node (by class or by field_paths)
            is_root_node = node.node_class in all_root_classes or node.field_paths == [""]
            # Never warn for the root node, even if field_paths is empty or [""]
            if is_root_node:
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
        #             f"between {relation.from_node.__name__} and {relation.to_node.__name__}"
        #         )

        # Validate embedded field configurations (MAP/STRUCT support)
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
                elif origin is Union:
                    candidate_types = [t for t in args if t is not type(None)]  # noqa: E721

                if embedded_class not in candidate_types:
                    warnings_list.append(
                        "Embedded field '"
                        f"{field_name}' on class {node.node_class.__name__} has incompatible type "
                        f"{annotation!r}; expected {embedded_class.__name__} or Optional[{embedded_class.__name__}]"
                    )

        # Store warnings
        self._warnings = warnings_list

        # Emit warnings
        for warning_msg in warnings_list:
            warnings.warn(f"Graph schema validation: {warning_msg}", UserWarning, stacklevel=2)

    def get_warnings(self) -> list[str]:
        """Get all validation warnings."""
        return self._warnings.copy()

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
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()

        console.print(Panel(f"[bold cyan]Graph Schema Summary for {self.root_model_class.__name__}[/bold cyan]"))

        # Nodes table
        nodes_table = Table(title="Node Configurations")
        nodes_table.add_column("Class", style="cyan")
        nodes_table.add_column("Key Field", style="magenta")
        nodes_table.add_column("Field Paths", style="green")
        nodes_table.add_column("Excluded Fields", style="yellow")

        for node in self.nodes:
            paths_str = ", ".join(node.field_paths) if node.field_paths else "ROOT"
            excluded_str = ", ".join(sorted(node.excluded_fields)) if node.excluded_fields else "None"
            nodes_table.add_row(node.node_class.__name__, node.key, paths_str, excluded_str)

        console.print(nodes_table)

        # Relations table
        relations_table = Table(title="Relationship Configurations")
        relations_table.add_column("Name", style="cyan")
        relations_table.add_column("From → To", style="magenta")
        relations_table.add_column("Field Path Pairs", style="green")

        for relation in self.relations:
            from_to = f"{relation.from_node.__name__} → {relation.to_node.__name__}"
            paths_str = (
                "; ".join([f"{fp} → {tp}" for fp, tp in relation.field_paths]) if relation.field_paths else "None"
            )
            relations_table.add_row(relation.name, from_to, paths_str)

        console.print(relations_table)

        # Warnings
        if self._warnings:
            console.print("\n[bold red]Warnings:[/bold red]")
            for warning in self._warnings:
                console.print(f"⚠️  {warning}")
