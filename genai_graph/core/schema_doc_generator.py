"""Generate Markdown schema documentation for knowledge graphs.

This module provides functionality to generate comprehensive, LLM-friendly
Markdown documentation of graph schemas, including node types, relationships,
properties, descriptions from BAML files, and indexed fields.
"""

from __future__ import annotations

import re
from enum import Enum
from functools import lru_cache
from typing import Any, get_args, get_origin

from genai_graph.core.graph_registry import GraphRegistry, get_subgraph
from genai_graph.core.graph_schema import GraphSchema


def generate_schema_description(subgraphs: str | list[str]) -> str:
    """Generate a compact, token-efficient LLM description of the graph schema.

    This unified function accepts either a single subgraph name (string)
    or a list of subgraph names. Passing an empty list means "all registered"
    subgraphs (delegated to `GraphRegistry.build_combined_schema`).

    Examples:
        ```python
        # Single subgraph
        description = generate_schema_description("ReviewedOpportunity")

        # Combined (multiple or empty list = all)
        description = generate_schema_description(["ReviewedOpportunity", "ArchitectureDocument"])
        ```
    """
    baml_docs = _parse_baml_descriptions()

    # Single subgraph name provided
    if isinstance(subgraphs, str):
        subgraph_impl = get_subgraph(subgraphs)
        subgraph_impl.build_schema()
        schema = _load_schema(subgraphs)
        return _format_schema_description(schema=schema, baml_docs=baml_docs)

    # Otherwise, treat as list of subgraph names (possibly empty => all)
    registry = GraphRegistry.get_instance()
    schema = registry.build_combined_schema(subgraphs)
    return _format_schema_description(schema=schema, baml_docs=baml_docs)


# NOTE: Combined-generator removed — use `generate_schema_description(list_or_name)`


def _load_schema(subgraph_name: str) -> GraphSchema:
    """Load and validate the subgraph schema.

    Ensures default subgraphs are registered before lookup.
    """

    try:
        subgraph_impl = get_subgraph(subgraph_name)
        return subgraph_impl.build_schema()
    except ValueError as e:
        raise ValueError(f"Unknown subgraph '{subgraph_name}': {e}") from e


@lru_cache(maxsize=1)
def _parse_baml_descriptions() -> dict[str, Any]:
    """Parse descriptions from BAML files.

    Returns dictionary with:
        - classes: dict[str, str] - Class name to description
        - fields: dict[str, dict[str, str]] - Class to field descriptions
        - enums: dict[str, dict[str, str]] - Enum name to value descriptions
    """
    from genai_graph.ekg.baml_client.inlinedbaml import _file_map

    classes: dict[str, str] = {}
    fields: dict[str, dict[str, str]] = {}
    enums: dict[str, dict[str, str]] = {}

    # Exclude client and generator files
    excluded_files = {"clients.baml", "generators.baml"}
    for filename, content in _file_map.items():
        if filename in excluded_files:
            continue

        _parse_baml_content(content, classes, fields, enums)

    return {"classes": classes, "fields": fields, "enums": enums}


def _parse_baml_content(
    content: str,
    classes: dict[str, str],
    fields: dict[str, dict[str, str]],
    enums: dict[str, dict[str, str]],
) -> None:
    """Parse a single BAML file content for descriptions."""
    lines = content.split("\n")
    current_block: str | None = None
    current_block_type: str | None = None  # 'class' or 'enum'
    pending_description: str | None = None

    for _, line in enumerate(lines):
        stripped = line.strip()

        # Check for standalone @@description before class/enum
        desc_match = re.match(r'@@description\s*\(\s*["\'](.+?)["\']\s*\)', stripped)
        if desc_match:
            pending_description = desc_match.group(1)
            continue

        # Check for class/enum declaration
        block_match = re.match(r"(class|enum)\s+([A-Za-z_]\w*)\s*\{", stripped)
        if block_match:
            current_block_type = block_match.group(1)
            current_block = block_match.group(2)

            # Check if @@description is on the same line or was pending
            inline_desc = re.search(r'@@description\s*\(\s*["\'](.+?)["\']\s*\)', line)
            if inline_desc:
                if current_block_type == "class":
                    classes[current_block] = inline_desc.group(1)
                elif current_block_type == "enum":
                    enums[current_block] = {}
            elif pending_description:
                if current_block_type == "class":
                    classes[current_block] = pending_description
                elif current_block_type == "enum":
                    enums[current_block] = {}
                pending_description = None
            else:
                # Initialize without description
                if current_block_type == "enum":
                    enums[current_block] = {}

            continue

        # Check for closing brace
        if stripped == "}" and current_block:
            current_block = None
            current_block_type = None
            continue

        # Inside a block - parse fields or enum values
        if current_block:
            if current_block_type == "class":
                # Parse field with optional @description
                # Match field_name type_annotation and capture @description if present
                field_match = re.match(r"([A-Za-z_]\w*)\s+([^@\n]+)", stripped)
                if field_match:
                    field_name = field_match.group(1)
                    # Look for @description in the line
                    desc_match = re.search(r'@description\s*\(\s*["\'](.+?)["\']\s*\)', line)
                    if desc_match:
                        field_desc = desc_match.group(1)
                        if current_block not in fields:
                            fields[current_block] = {}
                        fields[current_block][field_name] = field_desc

            elif current_block_type == "enum":
                # Parse enum value with optional @description
                enum_val_match = re.match(
                    r'([A-Za-z_]\w*)(?:\s+@alias\([^)]+\))?(?:\s+@description\s*\(\s*["\'](.+?)["\']\s*\))?',
                    stripped,
                )
                if enum_val_match:
                    enum_val = enum_val_match.group(1)
                    enum_desc = enum_val_match.group(2)
                    if enum_desc and current_block in enums:
                        enums[current_block][enum_val] = enum_desc


def _get_relation_properties(node_class: Any, baml_docs: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Extract relationship properties from a node class.

    Relationship properties are fields that match the p_*_ pattern (prefix p_, suffix _).
    Returns list of (name, type, description) tuples.
    """
    properties = []
    node_name = node_class.__name__

    if not hasattr(node_class, "model_fields"):
        return properties

    for field_name, field_info in node_class.model_fields.items():
        # Check if this is a relationship property (p_*_ pattern)
        if field_name.startswith("p_") and field_name.endswith("_"):
            # Remove the p_ prefix and _ suffix to get the display name
            display_name = field_name[2:-1]
            field_type = _humanize_type_compact(field_info.annotation)
            field_desc = baml_docs["fields"].get(node_name, {}).get(field_name, "")
            properties.append((display_name, field_type, field_desc))

    return properties


def _format_schema_description(schema: GraphSchema, baml_docs: dict[str, Any]) -> str:
    """Format schema as a compact, token-efficient description.

    Output format:
    - Nodes grouped by type with fields in format: name: type? // description
    - Relationships as: Source → [RELATION] → Dest with properties and description
    - Enumeration types with their values
    - Excludes embeddings and subgraph names
    """
    lines = ["## Graph Schema Description", ""]

    # Track embedded classes to exclude from main type listing
    embedded_classes = set()
    for node in schema.nodes:
        for _, embedded_class in node.embedded:
            embedded_classes.add(embedded_class.__name__)

    # Group nodes by type
    lines.append("### Node Types")
    lines.append("")

    for node in schema.nodes:
        node_name = node.baml_class.__name__

        # Skip embedded classes from main listing
        if node_name in embedded_classes:
            continue

        # Prefer schema description, fallback to BAML
        description = node.description or baml_docs["classes"].get(node_name, "")

        # Start with node type header
        if description:
            lines.append(f"{node_name}  // {description}")
        else:
            lines.append(f"{node_name}")

        # Build field list with compact format
        for field_name, field_info in node.baml_class.model_fields.items():
            if field_name not in node.excluded_fields:
                field_type = _humanize_type_compact(field_info.annotation)
                field_desc = baml_docs["fields"].get(node_name, {}).get(field_name, "")

                # Format: name: type  // description
                line = f"  {field_name}: {field_type}"
                if field_desc:
                    line += f"  // {field_desc}"
                lines.append(line)

        lines.append("")

    # Group relationships
    lines.extend(["### Relationships", ""])

    # Group by source node for clarity
    rels_by_source = {}
    for relation in schema.relations:
        source = relation.from_node.__name__
        if source not in rels_by_source:
            rels_by_source[source] = []
        rels_by_source[source].append(relation)

    for source in sorted(rels_by_source.keys()):
        for relation in rels_by_source[source]:
            dest = relation.to_node.__name__
            rel_name = relation.name
            description = relation.description

            # Format: Source → [RELATION] → Dest  # description
            line = f"{source} → [{rel_name}] → {dest}"
            if description:
                line += f"  // {description}"
            lines.append(line)

            # Add relationship properties from destination node (fields with p_*_ pattern)
            rel_properties = _get_relation_properties(relation.to_node, baml_docs)
            for prop_name, prop_type, prop_desc in rel_properties:
                prop_line = f"  {prop_name}: {prop_type}"
                if prop_desc:
                    prop_line += f"  // {prop_desc}"
                lines.append(prop_line)

        lines.append("")

    # Add enumerations section
    if baml_docs["enums"]:
        lines.extend(["### Enumerations", ""])

        for enum_name in sorted(baml_docs["enums"].keys()):
            enum_values = baml_docs["enums"][enum_name]
            enum_desc = baml_docs["classes"].get(enum_name, "")

            if enum_desc:
                lines.append(f"{enum_name}  // {enum_desc}")
            else:
                lines.append(f"{enum_name}")

            # List enum values
            for value_name in sorted(enum_values.keys()):
                value_desc = enum_values[value_name]
                if value_desc:
                    lines.append(f"  {value_name}  // {value_desc}")
                else:
                    lines.append(f"  {value_name}")

            lines.append("")

    return "\n".join(lines)


def _humanize_type_compact(annotation: Any, is_optional: bool = False) -> str:
    """Convert Python type annotation to compact LLM-friendly format.

    Examples:
        - string, int, float, boolean
        - string[], int[] (for lists)
        - string? (for optional)
        - string[]? (for optional list)
    """
    # Handle None/NoneType
    if annotation is type(None):
        return "null"

    # Unwrap Optional
    base_type, is_opt = _unwrap_optional(annotation)
    is_optional = is_optional or is_opt

    # Get the actual type to process
    origin = get_origin(base_type)
    args = get_args(base_type)

    # Handle generic types
    if origin is list:
        inner = _humanize_type_compact(args[0]) if args else "any"
        # Remove optional marker from inner type for list display
        inner_clean = inner.rstrip("?")
        result = f"{inner_clean}[]"
    elif origin is set:
        inner = _humanize_type_compact(args[0]) if args else "any"
        inner_clean = inner.rstrip("?")
        result = f"{inner_clean}[]"
    elif origin is tuple:
        inner = _humanize_type_compact(args[0]) if args else "any"
        inner_clean = inner.rstrip("?")
        result = f"{inner_clean}[]"
    elif origin is dict:
        result = "object"
    # Handle basic types
    elif base_type is str:
        result = "string"
    elif base_type is int:
        result = "int"
    elif base_type is float:
        result = "float"
    elif base_type is bool:
        result = "boolean"
    # Handle Enums
    elif isinstance(base_type, type) and issubclass(base_type, Enum):
        result = f"enum({base_type.__name__})"
    # Default to class name
    elif hasattr(base_type, "__name__"):
        result = base_type.__name__
    else:
        result = str(base_type)

    # Add optional marker with ? suffix
    if is_optional:
        result = f"{result}?"

    return result


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Unwrap Optional/Union types to get base type and optionality."""
    import types
    from typing import Union

    origin = get_origin(annotation)

    # Check for Union (including Optional which is Union[T, None])
    # Handle both Union (typing.Union) and UnionType (| syntax)
    if origin is Union or (hasattr(types, "UnionType") and origin is types.UnionType):
        args = get_args(annotation)
        # Filter out NoneType
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return non_none_args[0], True
        # Multiple non-None types - return first
        return non_none_args[0] if non_none_args else annotation, True

    return annotation, False


def _format_markdown(
    subgraph_name: str,
    node_sections: list[dict[str, Any]],
    rel_sections: list[dict[str, Any]],
    indexed_section: list[str],
) -> str:
    """Format all sections into final Markdown document."""
    lines = [
        f"# Knowledge Graph Schema — Subgraph: {subgraph_name}",
        "",
        "## Node Types",
        "",
    ]

    # Format nodes
    for node in node_sections:
        lines.append(f"### {node['name']}")
        if node["description"]:
            lines.append(f"**Description:** {node['description']}")
            lines.append("")

        if node["fields"]:
            lines.append("**Fields:**")
            for field in node["fields"]:
                field_line = f"- `{field['name']}`: {field['type']}"
                if field["description"]:
                    field_line += f" — {field['description']}"
                lines.append(field_line)
            lines.append("")

        if node["embedded"]:
            lines.append("**Embedded Fields:**")
            for emb in node["embedded"]:
                field_line = f"- `{emb['name']}`: {emb['type']}"
                if emb["description"]:
                    field_line += f" — {emb['description']}"
                lines.append(field_line)
            lines.append("")

    # Format relationships
    lines.extend(["", "## Relationships (Edges)", ""])

    for rel in rel_sections:
        lines.append(f"### {rel['name']}")
        lines.append(f"**Direction:** {rel['source']} → {rel['dest']}")
        if rel["description"]:
            lines.append(f"**Description:** {rel['description']}")
        lines.append("")

        if rel["properties"]:
            lines.append("**Properties:**")
            for prop in rel["properties"]:
                prop_line = f"- `{prop['name']}`: {prop['type']}"
                if prop["description"]:
                    prop_line += f" — {prop['description']}"
                lines.append(prop_line)
            lines.append("")

    # Format indexed fields
    lines.extend(["", "## Indexed Fields", ""])
    if indexed_section:
        lines.append("The following fields are indexed for vector search:")
        lines.append("")
        for field in indexed_section:
            lines.append(f"- `{field}`")
        lines.append("")
    else:
        lines.append("No fields are indexed for vector search.")
        lines.append("")

    return "\n".join(lines)
