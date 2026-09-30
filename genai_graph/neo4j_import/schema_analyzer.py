"""Schema analyzer for Neo4j JSONL exports.

Analyzes the JSONL export to extract node labels, relationship types, and their properties
to generate Ladybug (Kuzu-compatible) CREATE NODE TABLE and CREATE REL TABLE statements.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


class PropertyInfo(BaseModel):
    """Information about a property across all occurrences."""

    name: str
    types: set[str] = Field(default_factory=set)
    nullable: bool = False
    sample_values: list[Any] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    def infer_kuzu_type(self) -> str:
        """Infer the best Kuzu type for this property."""
        # Check if this is an embedding vector field
        if "embedding" in self.name.lower() and "list" in self.types:
            # Determine the vector size from sample values
            vector_size = self._infer_vector_size()
            if vector_size is not None:
                return f"FLOAT[{vector_size}]"

        # If multiple types or nullable, prefer STRING for flexibility
        if len(self.types) > 1 or self.nullable:
            return "STRING"

        if not self.types:
            return "STRING"

        python_type = next(iter(self.types))

        type_mapping = {
            "int": "INT64",
            "float": "DOUBLE",
            "bool": "BOOL",
            "str": "STRING",
            "list": "STRING[]",  # Simplified - could be more specific
            "dict": "STRING",  # Store as JSON string
            "NoneType": "STRING",
        }

        return type_mapping.get(python_type, "STRING")

    def _infer_vector_size(self) -> int | None:
        """Infer the size of an embedding vector from sample values.

        Returns:
            The consistent vector size if found, None otherwise.
        """
        sizes = set()
        for value in self.sample_values:
            # Check if all elements are numeric (int or float)
            if isinstance(value, list) and value and all(isinstance(x, (int, float)) for x in value):
                sizes.add(len(value))

        # Return size only if all samples have the same size
        if len(sizes) == 1:
            return sizes.pop()
        return None


class NodeTableInfo(BaseModel):
    """Information about a node table (label)."""

    label: str
    properties: dict[str, PropertyInfo] = Field(default_factory=dict)
    count: int = 0

    model_config = {"arbitrary_types_allowed": True}


class RelTableInfo(BaseModel):
    """Information about a relationship table."""

    rel_type: str
    from_labels: set[str] = Field(default_factory=set)
    to_labels: set[str] = Field(default_factory=set)
    properties: dict[str, PropertyInfo] = Field(default_factory=dict)
    count: int = 0

    model_config = {"arbitrary_types_allowed": True}


class SchemaInfo(BaseModel):
    """Complete schema information extracted from JSONL."""

    node_tables: dict[str, NodeTableInfo] = Field(default_factory=dict)
    rel_tables: dict[str, RelTableInfo] = Field(default_factory=dict)
    total_nodes: int = 0
    total_relationships: int = 0

    model_config = {"arbitrary_types_allowed": True}


class SchemaAnalyzer:
    """Analyzes Neo4j JSONL exports to extract schema information."""

    def __init__(self, jsonl_path: str | Path) -> None:
        """Initialize the analyzer with a JSONL file path."""
        self.jsonl_path = Path(jsonl_path)
        self.schema = SchemaInfo()
        self._node_id_to_labels: dict[str, list[str]] = {}

    def analyze(self, max_samples: int = 5) -> SchemaInfo:
        """Analyze the JSONL file and extract schema information.

        Args:
            max_samples: Maximum number of sample values to keep per property.

        Returns:
            SchemaInfo containing all extracted schema information.
        """
        logger.info("Analyzing JSONL file: {}", self.jsonl_path)

        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 10000 == 0:
                    logger.debug("Processed {} lines...", line_num)

                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("Line {}: Invalid JSON - {}", line_num, e)
                    continue

                record_type = record.get("type")

                if record_type == "node":
                    self._process_node(record, max_samples)
                elif record_type == "relationship":
                    self._process_relationship(record, max_samples)

        logger.info(
            "Analysis complete: {} nodes, {} relationships", self.schema.total_nodes, self.schema.total_relationships
        )
        logger.info("Node tables: {}", list(self.schema.node_tables.keys()))
        logger.info("Rel tables: {}", list(self.schema.rel_tables.keys()))

        return self.schema

    def _process_node(self, record: dict, max_samples: int) -> None:
        """Process a node record and update schema."""
        self.schema.total_nodes += 1

        node_id = record.get("id", "")
        labels = record.get("labels", [])
        properties = record.get("properties", {})

        # Store ID -> labels mapping for relationship processing
        self._node_id_to_labels[str(node_id)] = labels

        for label in labels:
            if label not in self.schema.node_tables:
                self.schema.node_tables[label] = NodeTableInfo(label=label)

            table_info = self.schema.node_tables[label]
            table_info.count += 1

            self._update_properties(table_info.properties, properties, max_samples)

    def _process_relationship(self, record: dict, max_samples: int) -> None:
        """Process a relationship record and update schema."""
        self.schema.total_relationships += 1

        rel_type = record.get("label", "UNKNOWN")
        start_node = record.get("start", {})
        end_node = record.get("end", {})
        properties = record.get("properties", {})

        # Extract labels from start/end nodes
        start_labels = start_node.get("labels", [])
        end_labels = end_node.get("labels", [])

        if rel_type not in self.schema.rel_tables:
            self.schema.rel_tables[rel_type] = RelTableInfo(rel_type=rel_type)

        rel_info = self.schema.rel_tables[rel_type]
        rel_info.count += 1
        rel_info.from_labels.update(start_labels)
        rel_info.to_labels.update(end_labels)

        self._update_properties(rel_info.properties, properties, max_samples)

    def _update_properties(self, props_dict: dict[str, PropertyInfo], properties: dict, max_samples: int) -> None:
        """Update property information dictionary with new property values."""
        for prop_name, prop_value in properties.items():
            if prop_name not in props_dict:
                props_dict[prop_name] = PropertyInfo(name=prop_name)

            prop_info = props_dict[prop_name]

            if prop_value is None:
                prop_info.nullable = True
            else:
                prop_info.types.add(type(prop_value).__name__)

            if len(prop_info.sample_values) < max_samples and prop_value is not None:
                prop_info.sample_values.append(prop_value)

    def generate_kuzu_schema(self, primary_key_property: str = "id") -> list[str]:
        """Generate Kuzu CREATE TABLE statements.

        Args:
            primary_key_property: The property to use as primary key for nodes.
                If not present in node properties, a synthetic '_neo4j_id' will be used.

        Returns:
            List of Kuzu Cypher CREATE statements.
        """
        statements = []

        # Generate node table statements
        for label, table_info in sorted(self.schema.node_tables.items()):
            stmt = self._generate_node_table_statement(label, table_info, primary_key_property)
            statements.append(stmt)

        # Generate relationship table statements
        for rel_type, rel_info in sorted(self.schema.rel_tables.items()):
            stmts = self._generate_rel_table_statements(rel_type, rel_info)
            statements.extend(stmts)

        return statements

    def _generate_node_table_statement(self, label: str, table_info: NodeTableInfo, primary_key_property: str) -> str:
        """Generate CREATE NODE TABLE statement."""
        props = []

        # Always add _neo4j_id as primary key for reliable linking
        props.append("_neo4j_id STRING")

        for prop_name, prop_info in sorted(table_info.properties.items()):
            kuzu_type = prop_info.infer_kuzu_type()
            # Escape property names that might be reserved
            safe_name = self._escape_property_name(prop_name)
            props.append(f"{safe_name} {kuzu_type}")

        props_str = ",\n    ".join(props)
        return f"""CREATE NODE TABLE IF NOT EXISTS {label} (
    {props_str},
    PRIMARY KEY (_neo4j_id)
);"""

    def _generate_rel_table_statements(self, rel_type: str, rel_info: RelTableInfo) -> list[str]:
        """Generate CREATE REL TABLE statements.

        For relationships that span multiple node types, we create one rel table
        per (from_label, to_label) combination.
        """
        statements = []

        # Build property string
        props = []
        for prop_name, prop_info in sorted(rel_info.properties.items()):
            kuzu_type = prop_info.infer_kuzu_type()
            safe_name = self._escape_property_name(prop_name)
            props.append(f"{safe_name} {kuzu_type}")

        props_str = ""
        if props:
            props_str = ",\n    " + ",\n    ".join(props)

        # Create one rel table per (from_label, to_label) combination
        from_labels = sorted(rel_info.from_labels) or ["UnknownNode"]
        to_labels = sorted(rel_info.to_labels) or ["UnknownNode"]

        for from_label in from_labels:
            for to_label in to_labels:
                # Always use the composite name format for consistency with JSON files
                table_name = f"{rel_type}_{from_label}_{to_label}"

                stmt = f"""CREATE REL TABLE IF NOT EXISTS {table_name} (
    FROM {from_label} TO {to_label}{props_str}
);"""
                statements.append(stmt)

        return statements

    def _escape_property_name(self, name: str) -> str:
        """Escape property names that might be reserved words."""
        reserved_words = {
            "asc",
            "desc",
            "from",
            "to",
            "match",
            "where",
            "return",
            "create",
            "delete",
            "set",
            "order",
            "by",
            "limit",
            "skip",
            "with",
            "as",
            "and",
            "or",
            "not",
            "in",
            "is",
            "null",
            "true",
            "false",
        }
        if name.lower() in reserved_words:
            return f"`{name}`"
        return name

    def print_summary(self) -> None:
        """Print a summary of the schema analysis."""
        print(f"\n{'=' * 60}")
        print("SCHEMA ANALYSIS SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total nodes: {self.schema.total_nodes:,}")
        print(f"Total relationships: {self.schema.total_relationships:,}")

        print(f"\n{'─' * 60}")
        print("NODE TABLES.")
        print(f"{'─' * 60}")
        for label, info in sorted(self.schema.node_tables.items()):
            print(f"\n  {label} ({info.count:,} nodes)")
            for prop_name, prop_info in sorted(info.properties.items()):
                types_str = ", ".join(prop_info.types)
                nullable_str = " (nullable)" if prop_info.nullable else ""
                print(f"    - {prop_name}: {types_str}{nullable_str}")

        print(f"\n{'─' * 60}")
        print("RELATIONSHIP TABLES.")
        print(f"{'─' * 60}")
        for rel_type, info in sorted(self.schema.rel_tables.items()):
            from_str = ", ".join(sorted(info.from_labels))
            to_str = ", ".join(sorted(info.to_labels))
            print(f"\n  {rel_type} ({info.count:,} relationships)")
            print(f"    From: [{from_str}]")
            print(f"    To: [{to_str}]")
            if info.properties:
                for prop_name, prop_info in sorted(info.properties.items()):
                    types_str = ", ".join(prop_info.types)
                    nullable_str = " (nullable)" if prop_info.nullable else ""
                    print(f"    - {prop_name}: {types_str}{nullable_str}")

        print(f"\n{'=' * 60}\n")
