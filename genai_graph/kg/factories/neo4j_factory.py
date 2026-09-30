"""Neo4j JSONL-backed factory for Knowledge Graph construction.

This factory reads structured data from Neo4j JSONL exports and transforms
them according to a mapping specification. It uses GraphNode and GraphRelation
classes for schema definition, enabling rich documentation generation for LLMs.

Features:
- Dynamic Pydantic model generation from Neo4j data
- Property mapping and renaming
- Full GraphSchema support with descriptions
- Index fields for vector search
"""

import json
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from loguru import logger
from pydantic import BaseModel, Field

from genai_graph.kg.factories.base import KgFactory
from genai_graph.kg.schema.core import GraphNode, GraphRelation, GraphSchema

if TYPE_CHECKING:
    from genai_graph.kg.ingest.extract import RelationshipRecord
    from genai_graph.kg.ingest.merge import NodeDataCollection


class Neo4jNodeMapping(BaseModel):
    """Configuration for mapping a Neo4j node type to the target schema.

    This class defines how a Neo4j node type should be transformed:
    - Target node class (actual Pydantic model for type safety)
    - Property mappings (rename, filter)
    - Index fields for vector search

    Example:
        Neo4jNodeMapping(
            neo4j_label="Account",
            node_class=Customer,
            property_mappings={"irisCode": "iris_code"},
        )
    """

    neo4j_label: str = Field(description="Original Neo4j label")
    node_class: type[BaseModel] = Field(description="Target Pydantic model class")
    property_mappings: dict[str, str] = Field(
        default_factory=dict,
        description="Map of neo4j_prop -> target_prop. Empty means copy all properties.",
    )
    name_field: str = Field(default="name", description="Field to use as node display name")
    key_field: str = Field(default="id", description="Field to use as primary key")
    index_fields: list[str | tuple[str, str]] = Field(
        default_factory=list,
        description="Fields to index for vector search. Each entry is either a plain field name "
        "(uses the default embedding model) or a (field_name, model_id) tuple that pins a specific "
        'embedding model for that field. Example: ["name", ("description", "ada_002@openai")].',
    )

    model_config = {"arbitrary_types_allowed": True}

    @property
    def target_label(self) -> str:
        """Get target label from node class name."""
        return self.node_class.__name__

    @property
    def description(self) -> str:
        """Get description from node class docstring."""
        return self.node_class.__doc__ or ""


class Neo4jRelationMapping(BaseModel):
    """Configuration for mapping a Neo4j relationship type to the target schema.

    This class defines how a Neo4j relationship should be transformed:
    - Source and target node types (as model classes for type safety)
    - Target relationship name (defaults to neo4j_type if not specified)
    - Description for LLM documentation

    Example:
        Neo4jRelationMapping(
            neo4j_type="LOCATED_IN",
            from_node=Customer,
            to_node=GEO,
            description="Geographic location of customer",
        )
    """

    neo4j_type: str = Field(description="Original Neo4j relationship type")
    target_rel: str | None = Field(default=None, description="Target relationship name (defaults to neo4j_type)")
    from_node: type[BaseModel] = Field(description="Source node model class")
    to_node: type[BaseModel] = Field(description="Target node model class")
    description: str = Field(default="", description="Human-readable description for LLM documentation")
    property_mappings: dict[str, str] = Field(
        default_factory=dict, description="Map of neo4j_prop -> target_prop for relationship properties"
    )

    model_config = {"arbitrary_types_allowed": True}

    @property
    def rel_name(self) -> str:
        """Get the target relationship name, defaulting to neo4j_type."""
        return self.target_rel if self.target_rel is not None else self.neo4j_type


class Neo4jFactory(KgFactory):
    """KG factory that reads structured data from Neo4j JSONL exports.

    This factory analyzes and processes Neo4j JSONL export files, transforming
    the nodes and relationships according to a mapping specification. It handles
    large JSONL files efficiently with streaming processing.

    Configuration attributes:
        neo4j_export_file: Path to the Neo4j JSONL export file.
    """

    neo4j_export_file: str

    # Caches for processed data
    _schema_info: Any = None
    _node_data: dict[str, list[dict[str, Any]]] = {}
    _rel_data: dict[str, list[dict[str, Any]]] = {}
    _initialized: bool = False

    # Neo4j ID to node type mapping for relationship resolution
    _neo4j_id_to_label: dict[str, str] = {}

    # Class-level cache to track which export files have been initialized
    _initialized_files: ClassVar[set[str]] = set()

    model_config = {
        "arbitrary_types_allowed": True,
    }

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the class-level initialization cache.

        Call this at the start of a new KG creation workflow to ensure
        fresh file discovery.
        """
        cls._initialized_files.clear()
        logger.debug(f"Cleared Neo4jFactory cache ({cls.__name__})")

    def model_post_init(self, _context: object) -> None:
        """Initialize and analyze the Neo4j JSONL file.

        Uses class-level cache to avoid redundant processing when the same
        factory is instantiated multiple times.
        """
        from genai_tk.config_mgmt.file_patterns import resolve_config_path

        resolved_path = resolve_config_path(self.neo4j_export_file)
        export_path = Path(resolved_path)

        if not export_path.exists():
            logger.warning(f"Neo4j export file not found: {export_path}")
            self._initialized = False
            return

        # Check if this file has already been processed
        file_key = str(export_path)
        if file_key in Neo4jFactory._initialized_files:
            logger.debug(f"Skipping duplicate Neo4j JSONL analysis for {export_path}")
            self._initialized = True
            return

        logger.info(f"Analyzing Neo4j JSONL export: {export_path}")
        self._analyze_and_load(export_path)
        Neo4jFactory._initialized_files.add(file_key)
        self._initialized = True

    def _analyze_and_load(self, export_path: Path) -> None:
        """Analyze and load data from Neo4j JSONL file.

        This method performs streaming processing of the JSONL file,
        building node and relationship data structures.

        Args:
            export_path: Path to the JSONL file
        """
        from genai_graph.neo4j_import.schema_analyzer import SchemaAnalyzer

        # First pass: analyze schema
        analyzer = SchemaAnalyzer(str(export_path))
        self._schema_info = analyzer.analyze()

        # Second pass: collect transformed data
        self._node_data = {}
        self._rel_data = {}

        with export_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 10000 == 0:
                    logger.debug(f"Processing line {line_num}...")

                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: Invalid JSON - {e}")
                    continue

                record_type = record.get("type")

                if record_type == "node":
                    self._process_node_record(record)
                elif record_type == "relationship":
                    self._process_rel_record(record)

        logger.info(
            f"Loaded {sum(len(v) for v in self._node_data.values())} nodes "
            f"and {sum(len(v) for v in self._rel_data.values())} relationships"
        )

    def _process_node_record(self, record: dict[str, Any]) -> None:
        """Process a single node record from JSONL.

        Args:
            record: The JSON record for a node
        """
        node_id = str(record.get("id", ""))
        labels = record.get("labels", [])
        properties = record.get("properties", {})

        # Track the primary label for this neo4j ID (used for relationship resolution)
        if labels:
            self._neo4j_id_to_label[node_id] = labels[0]

        for label in labels:
            if label not in self._node_data:
                self._node_data[label] = []

            node_record = {
                "_neo4j_id": node_id,
                **properties,
            }
            self._node_data[label].append(node_record)

    def _process_rel_record(self, record: dict[str, Any]) -> None:
        """Process a single relationship record from JSONL.

        Args:
            record: The JSON record for a relationship
        """
        rel_type = record.get("label", "UNKNOWN")
        start_node = record.get("start", {})
        end_node = record.get("end", {})
        properties = record.get("properties", {})

        start_id = str(start_node.get("id", ""))
        end_id = str(end_node.get("id", ""))
        start_labels = start_node.get("labels", [])
        end_labels = end_node.get("labels", [])

        # Create a key for this relationship type
        from_label = start_labels[0] if start_labels else "Unknown"
        to_label = end_labels[0] if end_labels else "Unknown"
        rel_key = f"{rel_type}__{from_label}__{to_label}"

        if rel_key not in self._rel_data:
            self._rel_data[rel_key] = []

        rel_record = {
            "_from_id": start_id,
            "_to_id": end_id,
            "_from_label": from_label,
            "_to_label": to_label,
            **properties,
        }
        self._rel_data[rel_key].append(rel_record)

    def get_schema_info(self) -> Any:
        """Return the analyzed schema information."""
        return self._schema_info

    def get_node_labels(self) -> dict[str, str]:  # type: ignore[override]
        """Return all discovered node labels.

        Note: This implementation returns a simplified dict[str, str] mapping
        instead of the base class's dict[str, str] with descriptions.
        """
        return {label: label for label in self._node_data}

    def get_relationship_types(self) -> list[str]:
        """Return all discovered relationship types."""
        return list(self._rel_data.keys())

    def get_nodes_by_label(self, label: str) -> list[dict[str, Any]]:
        """Get all nodes with a specific label.

        Args:
            label: The node label to filter by

        Returns:
            List of node data dictionaries
        """
        return self._node_data.get(label, [])

    def get_relationships_by_type(self, rel_type: str) -> list[dict[str, Any]]:
        """Get all relationships of a specific type.

        Args:
            rel_type: The relationship type key (format: TYPE__FromLabel__ToLabel)

        Returns:
            List of relationship data dictionaries
        """
        return self._rel_data.get(rel_type, [])

    def get_all_node_ids(self) -> list[str]:
        """Get all unique node IDs from the export.

        Returns:
            List of neo4j node IDs
        """
        ids = set()
        for nodes in self._node_data.values():
            for node in nodes:
                ids.add(node.get("_neo4j_id", ""))
        return sorted(ids)

    def get_struct_data_by_key(self, key: str) -> BaseModel | None:
        """Load structured data by key (neo4j node ID).

        This method is called during document ingestion. The key is expected
        to be a neo4j node ID, and this returns transformed data according
        to the schema mapping defined in build_schema.

        Args:
            key: The neo4j node ID

        Returns:
            Pydantic model instance or None if not found
        """
        # Subclasses must implement the mapping logic
        # Default implementation returns None - override in subclass
        return self._map_node_to_model(key)

    def _map_node_to_model(self, node_id: str) -> BaseModel | None:
        """Map a Neo4j node to a Pydantic model instance.

        Override this in subclasses to implement custom mapping logic.

        Args:
            node_id: The neo4j node ID

        Returns:
            Pydantic model instance or None
        """
        # Default: return None - subclasses must override
        return None

    def build_schema(self) -> GraphSchema:
        """Build and return the graph schema configuration.

        Subclasses must implement this to provide their specific schema.
        """
        raise NotImplementedError("Subclasses must implement build_schema()")


class Neo4jImportFactory(Neo4jFactory):
    """Extended Neo4j factory for direct graph import with schema-based mappings.

    This factory provides a complete solution for importing Neo4j data with:
    - Node type mapping via Pydantic classes (type-safe)
    - Property renaming (e.g., "irisCode" -> "iris_code")
    - Property filtering (include only specified properties)
    - Relationship mapping via Pydantic classes (type-safe)
    - Full GraphSchema support with descriptions for LLM documentation

    Subclasses should:
    1. Define Pydantic model classes for each node type (e.g., Customer, L3)
    2. Override get_node_mappings() to return list of Neo4jNodeMapping with node_class
    3. Override get_relation_mappings() to return list of Neo4jRelationMapping with from_node/to_node
    """

    @property
    def name(self) -> str:
        """Factory name for registration."""
        return self.__class__.__name__

    def get_node_mappings(self) -> list[Neo4jNodeMapping]:
        """Get node type mappings using Neo4jNodeMapping.

        Override this to define your node mappings with Pydantic classes.

        Returns:
            List of Neo4jNodeMapping configurations

        Example:
            return [
                Neo4jNodeMapping(
                    neo4j_label="Account",
                    node_class=Customer,
                    property_mappings={"irisCode": "iris_code", "name": "name"},
                    name_field="name",
                    key_field="iris_code",
                ),
            ]
        """
        return []

    def get_relation_mappings(self) -> list[Neo4jRelationMapping]:
        """Get relationship type mappings using Neo4jRelationMapping.

        Override this to define your relationship mappings with descriptions.
        This method is called after node models are created, so you can access
        them via self._dynamic_models.

        Returns:
            List of Neo4jRelationMapping configurations

        Example:
            Customer = self._dynamic_models.get("Customer")
            Ambition = self._dynamic_models.get("Ambition")
            return [
                Neo4jRelationMapping(
                    neo4j_type="HAS_AMBITION",
                    from_node=Customer,
                    to_node=Ambition,
                    description="Customer's strategic ambitions",
                ),
            ]
        """
        return []

    def get_included_node_types(self) -> set[str] | None:
        """Get the set of Neo4j labels to include.

        By default, returns labels from get_node_mappings().
        Override to customize filtering.

        Returns:
            Set of Neo4j labels to include, or None for all
        """
        mappings = self.get_node_mappings()
        if mappings:
            return {m.neo4j_label for m in mappings}
        return None

    def get_included_rel_types(self) -> set[str] | None:
        """Get the set of Neo4j relationship types to include.

        By default, returns types from get_relation_mappings().
        Override to customize filtering.

        Returns:
            Set of Neo4j rel types to include, or None for all
        """
        mappings = self.get_relation_mappings()
        if mappings:
            return {m.neo4j_type for m in mappings}
        return None

    def build_schema(self) -> GraphSchema:
        """Build a GraphSchema from the node and relation mappings.

        This generates proper GraphNode and GraphRelation configurations
        that can be used for documentation generation.

        Note: For Neo4j imports, we always use 'id' as the database primary key
        column to maintain compatibility with the import_neo4j_data function.
        The `key_field` in Neo4jNodeMapping determines which field to use as
        the logical key value stored in the `id` column.

        Returns:
            GraphSchema with full node and relationship definitions
        """
        node_mappings = self.get_node_mappings()
        relation_mappings = self.get_relation_mappings()

        # Build GraphNode list from node mappings
        graph_nodes: list[GraphNode] = []
        for mapping in node_mappings:
            node = GraphNode(
                node_class=mapping.node_class,
                name_from=mapping.name_field,
                key_from=mapping.key_field,
                description=mapping.description,
                index_fields=mapping.index_fields,
                explicitly_defined=True,  # Neo4j nodes don't need field path validation
            )
            graph_nodes.append(node)

        # Build lookup: Pydantic class → GraphNode (for resolving relation endpoints)
        node_by_class: dict[type[BaseModel], GraphNode] = {n.node_class: n for n in graph_nodes}

        # Build GraphRelation list from relation mappings
        graph_relations: list[GraphRelation] = []
        for mapping in relation_mappings:
            from_graph_node = node_by_class.get(mapping.from_node)
            to_graph_node = node_by_class.get(mapping.to_node)
            if from_graph_node is None or to_graph_node is None:
                logger.warning(
                    f"Skipping relation {mapping.rel_name!r}: "
                    f"from_node {mapping.from_node.__name__!r} or "
                    f"to_node {mapping.to_node.__name__!r} not found in node mappings"
                )
                continue

            # Transfer property_mappings to GraphRelation
            rel_properties = None
            if mapping.property_mappings:
                rel_properties = {target_prop: str for _, target_prop in mapping.property_mappings.items()}

            graph_relations.append(
                GraphRelation(
                    from_node=from_graph_node,
                    to_node=to_graph_node,
                    name=mapping.rel_name,
                    description=mapping.description,
                    properties=rel_properties,
                )
            )

        return GraphSchema(root_model_class=None, nodes=graph_nodes, relations=graph_relations)

    def build_nodes_and_relationships(
        self,
    ) -> tuple["NodeDataCollection", list["RelationshipRecord"]]:
        """Build NodeDataCollection and relationships from Neo4j data.

        This method applies the configured mappings to transform Neo4j data
        into the target format for direct import.

        Returns:
            Tuple of (NodeDataCollection, list[RelationshipRecord])
        """
        from datetime import datetime

        from genai_graph.kg.ingest.extract import RelationshipRecord
        from genai_graph.kg.ingest.merge import NodeDataCollection

        nodes_data = NodeDataCollection()
        relationships: list[RelationshipRecord] = []

        node_mappings = self.get_node_mappings()
        rel_mappings = self.get_relation_mappings()
        included_nodes = self.get_included_node_types()
        included_rels = self.get_included_rel_types()

        # Build lookup dicts from mapping lists
        node_mapping_by_label: dict[str, Neo4jNodeMapping] = {m.neo4j_label: m for m in node_mappings}
        rel_mapping_by_type: dict[str, Neo4jRelationMapping] = {m.neo4j_type: m for m in rel_mappings}

        # Track neo4j_id -> (target_type, target_id) for relationship resolution
        id_mapping: dict[str, tuple[str, str]] = {}
        now = datetime.now(UTC).isoformat()

        # Process nodes
        for neo4j_label, node_list in self._node_data.items():
            # Filter by included types
            if included_nodes is not None and neo4j_label not in included_nodes:
                logger.debug(f"Skipping node type {neo4j_label} (not in included types)")
                continue

            # Get mapping config (or use defaults)
            mapping = node_mapping_by_label.get(neo4j_label)
            if mapping:
                target_type = mapping.target_label
                prop_mapping = mapping.property_mappings
                name_field = mapping.name_field
                key_field = mapping.key_field
            else:
                # No mapping defined - use original label and all properties
                target_type = neo4j_label
                prop_mapping = {}
                name_field = "name"
                key_field = "id"

            for node in node_list:
                neo4j_id = node.get("_neo4j_id", "")

                # Apply property mapping
                if prop_mapping:
                    # Only include mapped properties
                    mapped_props = {}
                    model_fields = getattr(mapping.node_class, "model_fields", {}) if mapping else {}
                    for neo4j_prop, target_prop in prop_mapping.items():
                        if neo4j_prop not in node:
                            continue
                        value = node[neo4j_prop]
                        # Deserialize JSON-string embeddings (list[float] fields stored as strings in JSONL)
                        if isinstance(value, str) and target_prop in model_fields:
                            ann = model_fields[target_prop].annotation
                            # Unwrap Optional
                            if hasattr(ann, "__args__"):
                                inner_args = [a for a in ann.__args__ if a is not type(None)]
                                if inner_args:
                                    ann = inner_args[0]
                            if (
                                hasattr(ann, "__origin__")
                                and ann.__origin__ is list
                                and hasattr(ann, "__args__")
                                and ann.__args__
                                and ann.__args__[0] is float
                            ):
                                try:
                                    parsed = json.loads(value)
                                    if isinstance(parsed, list):
                                        value = [float(v) for v in parsed]
                                except (json.JSONDecodeError, ValueError, TypeError):
                                    logger.debug(f"Could not parse embedding string for {target_prop}")
                        mapped_props[target_prop] = value
                else:
                    # No mapping - copy all properties except internal ones
                    mapped_props = {k: v for k, v in node.items() if not k.startswith("_")}

                # Ensure key_field has a value
                if key_field not in mapped_props:
                    # Fallback: use neo4j_id as key value
                    mapped_props[key_field] = neo4j_id
                else:
                    # Stringify for consistency
                    mapped_props[key_field] = str(mapped_props[key_field])

                if "name" not in mapped_props:
                    # Use name_field value or try common name fields
                    if name_field in mapped_props:
                        mapped_props["name"] = str(mapped_props[name_field])
                    else:
                        for fallback_field in ["name", "title", "label", key_field]:
                            if fallback_field in mapped_props:
                                mapped_props["name"] = str(mapped_props[fallback_field])
                                break
                        else:
                            mapped_props["name"] = neo4j_id

                # Validate that primary key is not empty
                key_value = mapped_props.get(key_field)
                if not key_value or key_value == "":
                    logger.warning(
                        f"Skipping {target_type} node with empty {key_field}: neo4j_id={neo4j_id}, properties={mapped_props}"
                    )
                    continue

                # Add metadata timestamps
                mapped_props["_created_at"] = now
                mapped_props["_updated_at"] = now

                # Track for relationship resolution
                id_mapping[neo4j_id] = (target_type, mapped_props[key_field])

                nodes_data.add(target_type, mapped_props)

        # Process relationships
        for rel_key, rel_list in self._rel_data.items():
            # rel_key format: TYPE__FromLabel__ToLabel
            # Filter by included types using the full rel_key
            if included_rels is not None and rel_key not in included_rels:
                logger.debug(f"Skipping relationship type {rel_key} (not in included types)")
                continue

            # Get mapping config using the full rel_key
            rel_mapping = rel_mapping_by_type.get(rel_key)
            if rel_mapping:
                target_rel_type = rel_mapping.rel_name
                rel_prop_mapping = rel_mapping.property_mappings
            else:
                # No mapping - extract just the relationship type part
                parts = rel_key.split("__")
                target_rel_type = parts[0] if parts else rel_key
                rel_prop_mapping = {}

            for rel in rel_list:
                from_neo4j_id = rel.get("_from_id", "")
                to_neo4j_id = rel.get("_to_id", "")

                # Resolve to target types/ids
                if from_neo4j_id not in id_mapping or to_neo4j_id not in id_mapping:
                    # One or both nodes were filtered out
                    continue

                from_type, from_id = id_mapping[from_neo4j_id]
                to_type, to_id = id_mapping[to_neo4j_id]

                # Extract relationship properties - only include explicitly mapped properties
                # (Neo4j exports often have large embedding vectors that we don't want)
                rel_props = {}
                if rel_prop_mapping:
                    for neo4j_prop, target_prop in rel_prop_mapping.items():
                        if neo4j_prop in rel:
                            rel_props[target_prop] = rel[neo4j_prop]

                relationships.append(
                    RelationshipRecord(
                        from_type=from_type,
                        from_id=from_id,
                        to_type=to_type,
                        to_id=to_id,
                        name=target_rel_type,
                        properties=rel_props,
                    )
                )

        logger.info(
            f"Built {nodes_data.total_count()} nodes ({', '.join(f'{t}:{len(n)}' for t, n in nodes_data.items())}) "
            f"and {len(relationships)} relationships"
        )

        return nodes_data, relationships
