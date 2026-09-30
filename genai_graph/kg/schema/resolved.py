"""Canonical in-memory schema representation.

``ResolvedSchema`` is the single source of truth for a fully-enriched graph
schema.  It is built once from ``GraphSchema`` + BAML descriptions and then
renders to all output formats:

- ``.to_markdown()``   → LLM-facing prompt text (replaces ``format_schema_description``)
- ``.to_d3_json()``    → D3-ready dict (replaces ``build_schema_d3_data``)
- ``.to_html()``       → interactive HTML visualization
- ``.vector_indexes``  → structured list for programmatic DB index creation

This eliminates duplicated schema traversals across ``doc_generator.py``,
``schema_d3.py``, ``artifacts.py``, and the query layer.
"""

from __future__ import annotations

import json
import warnings
from datetime import UTC, datetime
from typing import Any

from genai_tk.utils.pydantic_utils.common import get_class_description as _get_class_description
from pydantic import BaseModel, Field

from genai_graph.kg.schema._helpers import (
    _collect_used_enums,
    _get_field_description,
    _get_kuzu_type_for_field,
    _get_node_description,
    _get_relation_properties,
    _humanize_type_compact,
    _parse_baml_descriptions,
)
from genai_graph.kg.schema.core import GraphSchema, find_embedded_field_for_class

# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class ResolvedField(BaseModel):
    """A single field on a resolved node."""

    name: str
    type_human: str
    type_kuzu: str
    description: str = ""
    indexed: bool = False
    embedded: bool = False
    parent_field: str | None = None
    embedded_class: str | None = None


class ResolvedNode(BaseModel):
    """A fully-described graph node."""

    name: str
    description: str = ""
    primary_key: str
    name_from: str
    index_fields: list[str] = Field(default_factory=list)
    fields: list[ResolvedField] = Field(default_factory=list)


class ResolvedRelationProperty(BaseModel):
    """A property on a relationship."""

    name: str
    type_human: str
    type_kuzu: str
    description: str = ""


class ResolvedRelation(BaseModel):
    """A fully-described graph relationship."""

    id: str
    source: str
    target: str
    name: str
    description: str = ""
    field_paths: list[dict[str, str]] = Field(default_factory=list)
    properties: list[ResolvedRelationProperty] = Field(default_factory=list)


class ResolvedEnumValue(BaseModel):
    """A single enum value with optional description."""

    name: str
    description: str = ""


class ResolvedEnum(BaseModel):
    """A fully-described enumeration type."""

    name: str
    description: str = ""
    values: list[ResolvedEnumValue] = Field(default_factory=list)


class VectorIndexInfo(BaseModel):
    """Metadata for a vector index on a table field."""

    table: str
    index_name: str
    embedding_column: str
    source_field: str


class SchemaMeta(BaseModel):
    """Metadata about when and how the schema was generated."""

    format: str = "genai_graph.resolved_schema"
    format_version: int = 1
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    graphs: list[str] = Field(default_factory=list)
    root_model: str | None = None


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class ResolvedSchema(BaseModel):
    """Canonical fully-enriched schema representation.

    Build via ``ResolvedSchema.from_graph_schema()`` then render to any
    target format with ``.to_markdown()``, ``.to_d3_json()``, ``.to_html()``.
    """

    meta: SchemaMeta
    nodes: list[ResolvedNode] = Field(default_factory=list)
    relations: list[ResolvedRelation] = Field(default_factory=list)
    enums: list[ResolvedEnum] = Field(default_factory=list)
    vector_indexes: list[VectorIndexInfo] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_graph_schema(
        cls,
        schema: GraphSchema,
        graph_names: list[str] | None = None,
        print_enums: bool = True,
        descriptions: dict[str, Any] | None = None,
    ) -> ResolvedSchema:
        """Build a ``ResolvedSchema`` from a ``GraphSchema`` and optional descriptions.

        Args:
            schema: The raw graph schema (from registry or factory).
            graph_names: Names of the graph factories included (for metadata).
            print_enums: Whether to resolve and include enumeration types.
            descriptions: Optional pre-parsed BAML/custom description dict with
                keys ``classes``, ``fields``, ``enums``.  When ``None``, field
                and class descriptions fall back to Pydantic ``Field(description
                =...)`` and docstrings only.  Pass the result of
                ``_parse_baml_descriptions(your_baml_file_map)`` to enrich with
                BAML ``@description`` annotations.
        """
        baml_docs = _parse_baml_descriptions(None) if descriptions is None else descriptions

        # Track embedded classes so they don't appear as top-level nodes.
        embedded_class_names: set[str] = set()
        for node in schema.nodes:
            for embedded_cls in getattr(node, "embedded_struct_classes", []) or []:
                embedded_class_names.add(embedded_cls.__name__)

        # ----------------------------------------------------------------
        # Nodes
        # ----------------------------------------------------------------
        nodes_out: list[ResolvedNode] = []
        for node in schema.nodes:
            node_name = node.node_class.__name__
            if node_name in embedded_class_names:
                continue

            description = _get_node_description(node, baml_docs)

            primary_key = "id" if node.key_from == "AUTO_ID" or callable(node.key_from) else str(node.key_from)
            name_from = node.name_from if isinstance(node.name_from, str) else "<callable>"

            fields_out: list[ResolvedField] = []
            model_fields = getattr(node.node_class, "model_fields", {})
            indexed_field_names = {fn for fn, _ in node.index_field_specs}

            for field_name, field_info in model_fields.items():
                if field_name == "metadata":
                    continue
                if field_name in node.excluded_fields:
                    continue

                type_human = _humanize_type_compact(field_info.annotation)
                if "ForwardRef" in type_human:
                    continue

                type_kuzu = _get_kuzu_type_for_field(field_info.annotation)
                field_desc = _get_field_description(node.node_class, field_name, field_info, baml_docs)

                # Check for embedded struct
                embedded_cls = None
                for emb_cls in getattr(node, "embedded_struct_classes", []) or []:
                    if find_embedded_field_for_class(node.node_class, emb_cls) == field_name:
                        embedded_cls = emb_cls
                        break

                if embedded_cls:
                    for sub_name, sub_info in getattr(embedded_cls, "model_fields", {}).items():
                        fields_out.append(
                            ResolvedField(
                                name=f"{field_name}.{sub_name}",
                                type_human=_humanize_type_compact(sub_info.annotation),
                                type_kuzu=_get_kuzu_type_for_field(sub_info.annotation),
                                description=_get_field_description(embedded_cls, sub_name, sub_info, baml_docs),
                                indexed=False,
                                embedded=True,
                                parent_field=field_name,
                                embedded_class=embedded_cls.__name__,
                            )
                        )
                else:
                    fields_out.append(
                        ResolvedField(
                            name=field_name,
                            type_human=type_human,
                            type_kuzu=type_kuzu,
                            description=field_desc,
                            indexed=field_name in indexed_field_names,
                            embedded=False,
                        )
                    )

            # Provenance metadata.source
            if "metadata" in model_fields:
                fields_out.append(
                    ResolvedField(
                        name="metadata.source",
                        type_human="string",
                        type_kuzu="STRING",
                        description="source of the document",
                        indexed=False,
                        embedded=True,
                        parent_field="metadata",
                        embedded_class="metadata",
                    )
                )

            nodes_out.append(
                ResolvedNode(
                    name=node_name,
                    description=description,
                    primary_key=primary_key,
                    name_from=name_from,
                    index_fields=[fn for fn, _ in node.index_field_specs],
                    fields=fields_out,
                )
            )

        # ----------------------------------------------------------------
        # Relations
        # ----------------------------------------------------------------
        relations_out: list[ResolvedRelation] = []
        for rel in schema.relations:
            source = rel.from_node.label
            target = rel.to_node.label
            props_out: list[ResolvedRelationProperty] = []
            for prop_name, prop_type_human, prop_desc in _get_relation_properties(rel.to_node.node_class, baml_docs):
                raw_field = f"p_{prop_name}_"
                fi = getattr(rel.to_node.node_class, "model_fields", {}).get(raw_field)
                prop_type_kuzu = _get_kuzu_type_for_field(fi.annotation) if fi else "STRING"
                props_out.append(
                    ResolvedRelationProperty(
                        name=prop_name,
                        type_human=prop_type_human,
                        type_kuzu=prop_type_kuzu,
                        description=prop_desc,
                    )
                )
            relations_out.append(
                ResolvedRelation(
                    id=f"{source}::{rel.name}::{target}",
                    source=source,
                    target=target,
                    name=rel.name,
                    description=rel.description or "",
                    field_paths=[{"from": fp or "", "to": tp or ""} for fp, tp in (rel.field_paths or [])],
                    properties=props_out,
                )
            )

        # ----------------------------------------------------------------
        # Enums
        # ----------------------------------------------------------------
        enums_out: list[ResolvedEnum] = []
        if print_enums:
            used_enums = _collect_used_enums(schema)
            used_enum_names = {e.__name__ for e in used_enums}
            relevant: dict[str, dict[str, str]] = {
                name: vals for name, vals in baml_docs["enums"].items() if name in used_enum_names
            }
            for enum_cls in used_enums:
                enum_name = enum_cls.__name__
                if enum_name not in relevant:
                    relevant[enum_name] = {m.name: (m.value if isinstance(m.value, str) else "") for m in enum_cls}
            for enum_name in sorted(relevant):
                enum_desc = baml_docs["classes"].get(enum_name, "")
                if not enum_desc:
                    for enum_cls in used_enums:
                        if enum_cls.__name__ == enum_name:
                            enum_desc = _get_class_description(enum_cls)
                            break
                values = [ResolvedEnumValue(name=vn, description=vd) for vn, vd in sorted(relevant[enum_name].items())]
                enums_out.append(ResolvedEnum(name=enum_name, description=enum_desc, values=values))

        # ----------------------------------------------------------------
        # Vector indexes (only fields whose _embedding column exists)
        # ----------------------------------------------------------------
        vector_indexes_out: list[VectorIndexInfo] = []
        for node in schema.nodes:
            if not node.compute_embeddings:
                continue
            table_name = node.node_class.__name__
            model_fields = node.node_class.model_fields
            for field_name, _model_override in node.index_field_specs:
                embedding_col = f"{field_name}_embedding"
                if embedding_col not in model_fields:
                    continue
                vector_indexes_out.append(
                    VectorIndexInfo(
                        table=table_name,
                        index_name=f"{field_name}_index",
                        embedding_column=embedding_col,
                        source_field=field_name,
                    )
                )

        root_name = schema.root_model_class.__name__ if schema.root_model_class else None
        return cls(
            meta=SchemaMeta(graphs=list(graph_names or []), root_model=root_name),
            nodes=nodes_out,
            relations=relations_out,
            enums=enums_out,
            vector_indexes=vector_indexes_out,
        )

    # ------------------------------------------------------------------
    # Convenience: load from registry for a given KG config
    # ------------------------------------------------------------------

    @classmethod
    def from_registry(
        cls,
        graph_names: list[str] | None = None,
        print_enums: bool = True,
    ) -> ResolvedSchema:
        """Build from the global GraphRegistry.

        Args:
            graph_names: Subset of registered graph names to include.
                ``None`` / empty → all registered graphs.
            print_enums: Whether to resolve and include enumeration types.
        """
        from genai_graph.kg.schema.registry import GraphRegistry

        registry = GraphRegistry.get_instance()
        names = graph_names or registry.list_graphs()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="Graph schema validation.")
            schema = registry.build_combined_schema(names)
        return cls.from_graph_schema(schema, graph_names=names, print_enums=print_enums)

    # ------------------------------------------------------------------
    # Render: Markdown
    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        """Render as compact, token-efficient Markdown for LLM prompts."""
        lines = ["## Graph Schema Description", ""]

        lines.append("### Node Types and their fields (labels)")
        lines.append("")

        # Names of embedded nodes (fields with parent_field set) — already
        # flattened into parent node fields; skip them as top-level entries.
        embedded_node_names: set[str] = set()
        for node in self.nodes:
            for f in node.fields:
                if f.embedded and f.embedded_class and f.embedded_class != "metadata":
                    embedded_node_names.add(f.embedded_class)

        for node in self.nodes:
            header = node.name
            if node.description:
                header += f" // {node.description}"
            lines.append(header)

            for field in node.fields:
                line = f"  {field.name}: {field.type_human}"
                if field.description:
                    line += f" // {field.description}"
                lines.append(line)

            lines.append("")

        lines.extend(["### Relationships and their properties", ""])

        rels_by_source: dict[str, list[ResolvedRelation]] = {}
        for rel in self.relations:
            rels_by_source.setdefault(rel.source, []).append(rel)

        for source in sorted(rels_by_source):
            for rel in rels_by_source[source]:
                line = f"{rel.source} → {rel.name} → {rel.target}"
                if rel.description:
                    line += f" // {rel.description}"
                lines.append(line)
                for prop in rel.properties:
                    prop_line = f"  {prop.name}: {prop.type_human}"
                    if prop.description:
                        prop_line += f" // {prop.description}"
                    lines.append(prop_line)
            lines.append("")

        root = self.meta.root_model or "(no root)"
        lines.append(f"{root} → [relation] → [Target] // Relationships originating from the root entity")

        if self.enums:
            lines.extend(["### Enumerations", ""])
            for enum in self.enums:
                header = enum.name
                if enum.description:
                    header += f" // {enum.description}"
                lines.append(header)
                for val in enum.values:
                    vline = f"  {val.name}"
                    if val.description:
                        vline += f" // {val.description}"
                    lines.append(vline)
                lines.append("")

        vector_section = self.to_vector_section_markdown()
        if vector_section:
            lines.append("")
            lines.append(vector_section)

        return "\n".join(lines)

    def to_vector_section_markdown(self) -> str:
        """Return the ``### Vector-Indexed Fields`` Markdown section, or empty string."""
        if not self.vector_indexes:
            return ""
        lines = ["### Vector-Indexed Fields (for semantic similarity search)", ""]
        for vi in self.vector_indexes:
            lines.append(
                f"- QUERY_VECTOR_INDEX('{vi.table}', '{vi.index_name}', ...)  → searches {vi.table}.{vi.source_field}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Render: D3 JSON
    # ------------------------------------------------------------------

    def to_d3_json(self) -> dict[str, Any]:
        """Render as D3-ready JSON (nodes + links + meta)."""
        nodes_out = [
            {
                "id": n.name,
                "label": n.name,
                "description": n.description,
                "primary_key": n.primary_key,
                "name_from": n.name_from,
                "index_fields": n.index_fields,
                "fields": [
                    {
                        "name": f.name,
                        "type_human": f.type_human,
                        "type_kuzu": f.type_kuzu,
                        "description": f.description,
                        "indexed": f.indexed,
                        "embedded": f.embedded,
                        **({"parent_field": f.parent_field} if f.parent_field else {}),
                        **({"embedded_class": f.embedded_class} if f.embedded_class else {}),
                    }
                    for f in n.fields
                ],
            }
            for n in self.nodes
        ]

        links_out = [
            {
                "id": r.id,
                "source": r.source,
                "target": r.target,
                "label": r.name,
                "description": r.description,
                "field_paths": r.field_paths,
                "properties": [
                    {
                        "name": p.name,
                        "type_human": p.type_human,
                        "type_kuzu": p.type_kuzu,
                        "description": p.description,
                    }
                    for p in r.properties
                ],
            }
            for r in self.relations
        ]

        return {
            "meta": self.meta.model_dump(),
            "nodes": nodes_out,
            "links": links_out,
            "vector_indexes": [vi.model_dump() for vi in self.vector_indexes],
        }

    # ------------------------------------------------------------------
    # Render: HTML
    # ------------------------------------------------------------------

    def to_html(self, destination_file_path: str | None = None) -> str:
        """Render as an interactive D3.js HTML visualization.

        Args:
            destination_file_path: If provided, write the HTML to this path.

        Returns:
            HTML string.
        """
        from genai_graph.kg.schema.schema_html import generate_schema_html

        html = generate_schema_html(self.to_d3_json(), destination_file_path=destination_file_path)
        return html or ""

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_json_str(self, indent: int = 2) -> str:
        """Serialize the canonical schema to a JSON string."""
        return json.dumps(self.to_d3_json(), indent=indent)

    @classmethod
    def from_json_file(cls, path: str) -> ResolvedSchema:
        """Load a ``ResolvedSchema`` from a saved canonical JSON file.

        The file must have been written by ``to_json_str()`` / ``export_schema_json()``.
        """
        import json as _json
        from pathlib import Path

        data = _json.loads(Path(path).read_text(encoding="utf-8"))
        meta_data = data.get("meta", {})
        meta = SchemaMeta(
            format=meta_data.get("format", "genai_graph.resolved_schema"),
            format_version=meta_data.get("format_version", 1),
            generated_at=meta_data.get("generated_at", datetime.now(UTC).isoformat()),
            graphs=meta_data.get("graphs", []),
            root_model=meta_data.get("root_model"),
        )

        def _load_fields(raw_fields: list[dict]) -> list[ResolvedField]:
            return [ResolvedField(**f) for f in raw_fields]

        nodes = [
            ResolvedNode(
                name=n["id"],
                description=n.get("description", ""),
                primary_key=n.get("primary_key", "id"),
                name_from=n.get("name_from", "name"),
                index_fields=n.get("index_fields", []),
                fields=_load_fields(n.get("fields", [])),
            )
            for n in data.get("nodes", [])
        ]

        relations = [
            ResolvedRelation(
                id=r["id"],
                source=r["source"],
                target=r["target"],
                name=r["label"],
                description=r.get("description", ""),
                field_paths=r.get("field_paths", []),
                properties=[ResolvedRelationProperty(**p) for p in r.get("properties", [])],
            )
            for r in data.get("links", [])
        ]

        vector_indexes = [VectorIndexInfo(**vi) for vi in data.get("vector_indexes", [])]

        return cls(meta=meta, nodes=nodes, relations=relations, enums=[], vector_indexes=vector_indexes)
