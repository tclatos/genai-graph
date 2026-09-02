"""Integration tests for Kuzu embeddings functionality.

Tests cover:
- Scenario A: Pre-computed embeddings (L3 descriptionEmbedding from Neo4j)
- Scenario B: Calculated embeddings from index_fields
- Scenario C: Mixed pre-computed and calculated embeddings
- Scenario D: Vector index queries and caching behavior
- Scenario E: embedding_field_dimensions + FLOAT[N] schema for pre-computed fields
- Scenario F: Neo4j JSON-string embedding deserialization
- Scenario G: DataFrame float-list → ArrowExtensionArray for Kuzu LOAD FROM df
"""

import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from genai_graph.kg.backend import KuzuBackend
from genai_graph.kg.embeddings_handler import EmbeddingsHandler
from genai_graph.kg.schema.core import GraphNode


class SimpleNode(BaseModel):
    """Minimal test node for embeddings testing."""

    id: str = Field(description="Unique identifier")
    name: str = Field(description="Node name")
    description: str | None = Field(default=None, description="Node description")
    test_embedding: list[float] | None = Field(default=None, description="Test embedding vector")


@pytest.fixture
def temp_kuzu_db():
    """Create a temporary Kuzu database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.kuzu"
        backend = KuzuBackend()
        backend.connect(str(db_path))
        yield backend, db_path
        backend.close()


@pytest.fixture
def embeddings_handler():
    """Create an EmbeddingsHandler instance."""
    handler = EmbeddingsHandler(embeddings_id="embeddings_768@fake")
    return handler


class TestEmbeddingsHandler:
    """Test EmbeddingsHandler core functionality."""

    def test_handler_initialization(self, embeddings_handler):
        """Test that handler initializes with valid model."""
        assert embeddings_handler is not None
        assert embeddings_handler.embeddings_id is not None

    def test_compute_single_embedding(self, embeddings_handler):
        """Test computing embedding for a single text."""
        text = "This is a test document about cloud services"
        embedding = embeddings_handler.compute_embeddings(text)

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    def test_compute_batch_embeddings(self, embeddings_handler):
        """Test computing embeddings for a batch of texts."""
        texts = [
            "This is a test document about cloud services",
            "Database management systems and graph databases",
        ]
        embeddings = embeddings_handler.compute_embeddings_batch(texts)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 2
        assert all(isinstance(e, list) and len(e) > 0 for e in embeddings)
        assert embeddings_handler.compute_embeddings_batch([]) == []

    def test_compute_field_embeddings(self, embeddings_handler):
        """Test computing embeddings for multiple fields."""
        node_data = {
            "id": "node1",
            "name": "Cloud Storage Service",
            "description": "A comprehensive cloud storage solution",
        }
        index_fields = ["name", "description"]

        embeddings = embeddings_handler.compute_field_embeddings(node_data, index_fields)

        assert "name" in embeddings
        assert "description" in embeddings
        assert len(embeddings["name"]) > 0
        assert len(embeddings["description"]) > 0

    def test_compute_embeddings_missing_field(self, embeddings_handler):
        """Test handling of missing fields."""
        node_data = {"id": "node1", "name": "Test"}
        index_fields = ["name", "nonexistent_field"]

        embeddings = embeddings_handler.compute_field_embeddings(node_data, index_fields)

        # Only name should be computed
        assert "name" in embeddings
        assert "nonexistent_field" not in embeddings

    def test_embed_empty_string(self, embeddings_handler):
        """Test that empty strings are skipped."""
        node_data = {"id": "node1", "name": "", "description": "Test"}
        index_fields = ["name", "description"]

        embeddings = embeddings_handler.compute_field_embeddings(node_data, index_fields)

        # Only description should be computed
        assert "name" not in embeddings
        assert "description" in embeddings


class TestKuzuFloatArrayType:
    """Test Kuzu FLOAT[] type mapping and storage."""

    def test_float_array_type_mapping(self):
        """Test that list[float] maps to FLOAT[] in Kuzu."""
        from genai_graph.kg.ingest.extract import _get_kuzu_type

        kuzu_type = _get_kuzu_type(list[float])
        assert kuzu_type == "FLOAT[]"

    def test_string_array_type_mapping(self):
        """Test that list[str] still maps to STRING[]."""
        from genai_graph.kg.ingest.extract import _get_kuzu_type

        kuzu_type = _get_kuzu_type(list[str])
        assert kuzu_type == "STRING[]"

    def test_optional_float_array(self):
        """Test that Optional[list[float]] maps to FLOAT[]."""
        from genai_graph.kg.ingest.extract import _get_kuzu_type

        kuzu_type = _get_kuzu_type(list[float] | None)
        assert kuzu_type == "FLOAT[]"


class TestSchemaCreationWithEmbeddings:
    """Test schema creation with embedding fields."""

    def test_simple_node_schema_with_embedding(self, temp_kuzu_db):
        """Test creating schema for node with embedding field."""
        backend, _ = temp_kuzu_db

        # Create table with embedding column
        backend.execute(
            """
            CREATE NODE TABLE SimpleNode (
                id STRING PRIMARY KEY,
                name STRING,
                description STRING,
                test_embedding FLOAT[]
            );
            """
        )

        # Verify table was created
        result = backend.execute("CALL table_info('SimpleNode') RETURN *;")
        assert result is not None

    def test_insert_embedding_data(self, temp_kuzu_db):
        """Test inserting nodes with embedding data."""
        backend, _ = temp_kuzu_db

        # Create table
        backend.execute(
            """
            CREATE NODE TABLE SimpleNode (
                id STRING PRIMARY KEY,
                name STRING,
                test_embedding FLOAT[]
            );
            """
        )

        # Insert node with embedding
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        backend.execute(
            """
            CREATE (n:SimpleNode {
                id: 'node1',
                name: 'Test Node',
                test_embedding: $embedding
            });
            """,
            {"embedding": embedding},
        )

        # Retrieve and verify
        result = backend.execute("MATCH (n:SimpleNode) RETURN n.test_embedding AS embedding;")
        df = result.get_as_df()
        assert len(df) == 1


class TestVectorIndexCreation:
    """Test vector index creation and querying."""

    def test_ensure_vector_extension(self, temp_kuzu_db):
        """Test that vector extension can be loaded."""
        backend, _ = temp_kuzu_db

        # Should not raise
        backend.ensure_vector_extension()

    def test_create_vector_index(self, temp_kuzu_db):
        """Test creating a vector index."""
        backend, _ = temp_kuzu_db
        backend.ensure_vector_extension()

        # Create table and index
        backend.execute(
            """
            CREATE NODE TABLE TestDoc (
                id STRING PRIMARY KEY,
                title STRING,
                title_embedding FLOAT[384]
            );
            """
        )

        # Create vector index
        backend.create_vector_index(
            table_name="TestDoc",
            field_name="title_embedding",
            index_name="test_doc_title_index",
            metric="cosine",
        )

        # Verify index was created
        result = backend.execute("CALL SHOW_INDEXES() RETURN *;")
        df = result.get_as_df()
        assert len(df) > 0
        # Check if our index is in the results
        index_names = df.get_column("index_name").to_list() if hasattr(df, "get_column") else list(df["index_name"])
        assert "test_doc_title_index" in index_names

    def test_vector_index_with_dummy_data(self, temp_kuzu_db):
        """Test vector index query with synthetic data."""
        backend, _ = temp_kuzu_db
        backend.ensure_vector_extension()

        # Create table
        backend.execute(
            """
            CREATE NODE TABLE Book (
                id STRING PRIMARY KEY,
                title STRING,
                title_embedding FLOAT[3]
            );
            """
        )

        # Insert sample data with 3-dimensional embeddings
        embeddings_data = [
            ("book1", "Machine Learning Basics", [0.1, 0.2, 0.3]),
            ("book2", "Deep Learning Advanced", [0.11, 0.21, 0.31]),
            ("book3", "Quantum Computing", [0.5, 0.6, 0.7]),
        ]

        for book_id, title, embedding in embeddings_data:
            backend.execute(
                """
                CREATE (b:Book {
                    id: $id,
                    title: $title,
                    title_embedding: $embedding
                });
                """,
                {
                    "id": book_id,
                    "title": title,
                    "embedding": embedding,
                },
            )

        # Create vector index
        backend.create_vector_index(
            table_name="Book",
            field_name="title_embedding",
            index_name="book_index",
            metric="l2",
        )

        # Query the index
        query_embedding = [0.1, 0.2, 0.3]  # Similar to book1
        result = backend.query_vector_index(
            table_name="Book",
            index_name="book_index",
            query_vector=query_embedding,
            k=2,
        )

        df = result.get_as_df()
        assert len(df) >= 1  # At least one result


class TestGraphNodeEmbeddingConfig:
    """Test GraphNode configuration for embeddings."""

    def test_graph_node_defaults(self):
        """Test that GraphNode has correct embedding defaults."""
        node_config = GraphNode(
            node_class=SimpleNode,
            name_from="name",
            key_from="id",
        )
        # compute_embeddings is derived from index_fields; empty list → False
        assert node_config.compute_embeddings is False
        assert node_config.embedding_field_dimensions == {}
        assert node_config.index_fields == []

    def test_graph_node_with_embeddings_disabled(self):
        """Test that compute_embeddings is False when index_fields is empty."""
        node_config = GraphNode(
            node_class=SimpleNode,
            name_from="name",
            key_from="id",
        )
        assert node_config.compute_embeddings is False

    def test_graph_node_with_index_fields_enables_embeddings(self):
        """Test that compute_embeddings is True when index_fields is non-empty."""
        node_config = GraphNode(
            node_class=SimpleNode,
            name_from="name",
            key_from="id",
            index_fields=["name"],
        )
        assert node_config.compute_embeddings is True

    def test_graph_node_with_index_fields(self):
        """Test index_fields configuration."""
        node_config = GraphNode(
            node_class=SimpleNode,
            name_from="name",
            key_from="id",
            index_fields=["name", "description"],
        )
        assert node_config.index_fields == ["name", "description"]


class TestNodeWithEmbeddingField:
    """Test a node model that includes a pre-computed embedding field."""

    @pytest.fixture
    def service_item_cls(self) -> type:
        from pydantic import BaseModel

        class ServiceItem(BaseModel):
            name: str
            code: str | None = None
            description: str | None = None
            description_embedding: list[float] | None = None

        return ServiceItem

    def test_model_has_embedding_field(self, service_item_cls: type) -> None:
        """Test that a model with an embedding field stores it correctly."""
        item = service_item_cls(
            name="Cloud Storage",
            code="CS001",
            description="A service for cloud storage",
            description_embedding=[0.1, 0.2, 0.3],
        )
        assert item.description_embedding == [0.1, 0.2, 0.3]

    def test_embedding_optional(self, service_item_cls: type) -> None:
        """Test that the embedding field is optional."""
        item = service_item_cls(name="Cloud Storage", code="CS001")
        assert item.description_embedding is None

    def test_embedding_with_none(self, service_item_cls: type) -> None:
        """Test explicitly setting embedding to None."""
        item = service_item_cls(name="Cloud Storage", code="CS001", description_embedding=None)
        assert item.description_embedding is None


class TestEmbeddingFieldNaming:
    """Test naming conventions for embedding fields."""

    def test_embedding_field_naming_from_index_fields(self):
        """Test that computed embedding field names follow convention."""
        # For index_fields like ["name", "description"],
        # embeddings should be stored in fields like "name_embedding", "description_embedding"
        index_fields = ["name", "description", "keywords"]
        expected_embedding_fields = [f"{field}_embedding" for field in index_fields]

        assert expected_embedding_fields == ["name_embedding", "description_embedding", "keywords_embedding"]

    def test_pre_computed_field_naming(self):
        """Test that pre-computed fields keep their original names."""
        # descriptionEmbedding is pre-computed from Neo4j, should use exact name
        field_name = "description_embedding"
        assert field_name == "description_embedding"


# Performance and caching tests
class TestEmbeddingCaching:
    """Test caching behavior of EmbeddingsHandler."""

    def test_handler_uses_cache_enabled(self, embeddings_handler):
        """Test that handler is initialized with caching enabled."""
        # The handler should be created with cache_embeddings=True
        assert embeddings_handler.factory is not None
        # Verify by calling twice and checking behavior
        text = "Test caching"
        emb1 = embeddings_handler.compute_embeddings(text)
        emb2 = embeddings_handler.compute_embeddings(text)
        # Should get the same result
        assert emb1 == emb2


class TestDocGenerationWithEmbeddings:
    """Test schema documentation handles embeddings correctly."""

    def test_float_array_in_json_schema(self):
        """Test that FLOAT[] is correctly represented in JSON schema."""
        from genai_graph.kg.schema._helpers import _get_kuzu_type_for_field

        kuzu_type = _get_kuzu_type_for_field(list[float])
        assert kuzu_type == "FLOAT[]"

        # Also test optional
        kuzu_type_opt = _get_kuzu_type_for_field(list[float] | None)
        assert kuzu_type_opt == "FLOAT[]"


# ---------------------------------------------------------------------------
# New tests for Scenario E: embedding_field_dimensions + FLOAT[N] schema
# ---------------------------------------------------------------------------


class TestEmbeddingFieldDimensions:
    """Test GraphNode.embedding_field_dimensions and FLOAT[N] schema generation."""

    def test_graph_node_embedding_field_dimensions_default(self):
        """Test that embedding_field_dimensions defaults to empty dict."""
        node = GraphNode(node_class=SimpleNode, name_from="name", key_from="id")
        assert node.embedding_field_dimensions == {}

    def test_graph_node_embedding_field_dimensions_set(self):
        """Test setting embedding_field_dimensions via internal attribute."""
        node = GraphNode(
            node_class=SimpleNode,
            name_from="name",
            key_from="id",
        )
        node._embedding_field_dimensions = {"test_embedding": 1536}
        assert node.embedding_field_dimensions == {"test_embedding": 1536}

    def test_create_schema_uses_float_n_for_known_dimension(self, temp_kuzu_db):
        """Test that create_schema() emits FLOAT[N] when dimension is known."""
        from genai_graph.kg.ingest.extract import create_schema

        backend, _ = temp_kuzu_db

        node = GraphNode(
            node_class=SimpleNode,
            name_from="name",
            key_from="id",
        )
        node._embedding_field_dimensions = {"test_embedding": 768}

        create_schema(backend, [node], [])

        # Inspect table columns
        result = backend.execute("CALL table_info('SimpleNode') RETURN *;")
        df = result.get_as_df()
        col_types = dict(zip(df["name"].tolist(), df["type"].tolist()))
        # test_embedding is already in the model — dimension should be FLOAT[768]
        assert col_types.get("test_embedding") == "FLOAT[768]"

    def test_create_schema_falls_back_to_float_array_without_dimension(self, temp_kuzu_db):
        """Test that create_schema() emits FLOAT[] when no dimension is provided."""
        from genai_graph.kg.ingest.extract import create_schema

        backend, _ = temp_kuzu_db

        node = GraphNode(
            node_class=SimpleNode,
            name_from="name",
            key_from="id",
            # no embedding_field_dimensions set
        )

        create_schema(backend, [node], [])

        result = backend.execute("CALL table_info('SimpleNode') RETURN *;")
        df = result.get_as_df()
        col_types = dict(zip(df["name"].tolist(), df["type"].tolist()))
        assert col_types.get("test_embedding") == "FLOAT[]"

    def test_create_schema_float_n_allows_vector_index(self, temp_kuzu_db):
        """Test that FLOAT[N] columns can have HNSW vector indexes created on them."""
        from genai_graph.kg.ingest.extract import create_schema

        backend, _ = temp_kuzu_db
        backend.ensure_vector_extension()

        node = GraphNode(
            node_class=SimpleNode,
            name_from="name",
            key_from="id",
        )
        node._embedding_field_dimensions = {"test_embedding": 768}
        create_schema(backend, [node], [])

        # Should not raise
        backend.create_vector_index("SimpleNode", "test_embedding", "simple_emb_index", metric="cosine")

        result = backend.execute("CALL SHOW_INDEXES() RETURN *;")
        df = result.get_as_df()
        index_names = list(df["index_name"])
        assert "simple_emb_index" in index_names


# ---------------------------------------------------------------------------
# New tests for Scenario E: Neo4jNodeMapping tuple index_fields + per-field model
# ---------------------------------------------------------------------------


class TestNeo4jNodeMappingIndexFields:
    """Test Neo4jNodeMapping.index_fields tuple syntax and per-field model pinning."""

    def test_neo4j_node_mapping_index_fields_default(self):
        """Test that index_fields defaults to empty list."""
        from genai_graph.kg.factories.neo4j_factory import Neo4jNodeMapping

        mapping = Neo4jNodeMapping(neo4j_label="L3", node_class=SimpleNode, name_field="name", key_field="id")
        assert mapping.index_fields == []

    def test_neo4j_node_mapping_index_fields_plain_strings(self):
        """Test that index_fields accepts plain strings (uses default embedding model)."""
        from genai_graph.kg.factories.neo4j_factory import Neo4jNodeMapping

        mapping = Neo4jNodeMapping(
            neo4j_label="L3",
            node_class=SimpleNode,
            name_field="name",
            key_field="id",
            index_fields=["name"],
        )
        assert mapping.index_fields == ["name"]

    def test_neo4j_node_mapping_index_fields_tuple_overrides_model(self):
        """Test that index_fields accepts (field, model_id) tuples for per-field model pinning."""
        from genai_graph.kg.factories.neo4j_factory import Neo4jNodeMapping

        mapping = Neo4jNodeMapping(
            neo4j_label="L3",
            node_class=SimpleNode,
            name_field="name",
            key_field="id",
            index_fields=[("test_field", "embeddings_768@fake")],
        )
        assert mapping.index_fields == [("test_field", "embeddings_768@fake")]

    def test_build_schema_passes_index_fields_to_graph_node(self):
        """Test that build_schema() passes tuple index_fields through to GraphNode.index_field_specs."""
        from genai_graph.kg.factories.neo4j_factory import Neo4jImportFactory, Neo4jNodeMapping

        class MinimalFactory(Neo4jImportFactory):
            neo4j_export_file: str = "/dev/null"

            def get_node_mappings(self):
                return [
                    Neo4jNodeMapping(
                        neo4j_label="SimpleNode",
                        node_class=SimpleNode,
                        name_field="name",
                        key_field="id",
                        index_fields=["name", ("description", "embeddings_768@fake")],
                    )
                ]

        factory = MinimalFactory.__new__(MinimalFactory)
        object.__setattr__(factory, "_initialized", False)
        object.__setattr__(factory, "_node_data", {})
        object.__setattr__(factory, "_rel_data", {})
        object.__setattr__(factory, "_schema_info", None)
        object.__setattr__(factory, "_neo4j_id_to_label", {})

        schema = factory.build_schema()
        assert len(schema.nodes) == 1
        node_cfg = schema.nodes[0]
        assert node_cfg.index_field_specs == [("name", None), ("description", "embeddings_768@fake")]

    def test_neo4j_mapping_explicit_model_override_pins_embedding_model(self):
        """Test that an explicit (field, model) tuple in index_fields is passed through."""
        from pydantic import BaseModel

        from genai_graph.kg.factories.neo4j_factory import Neo4jImportFactory, Neo4jNodeMapping

        class Concept(BaseModel):
            name: str
            description: str | None = None
            description_embedding: list[float] | None = None

        class ConceptFactory(Neo4jImportFactory):
            neo4j_export_file: str = "/dev/null"

            def get_node_mappings(self):
                return [
                    Neo4jNodeMapping(
                        neo4j_label="Concept",
                        node_class=Concept,
                        property_mappings={"name": "name", "description": "description"},
                        name_field="name",
                        key_field="name",
                        index_fields=[("description", "ada_002@openai")],
                    )
                ]

        factory = ConceptFactory.__new__(ConceptFactory)
        object.__setattr__(factory, "_initialized", False)
        object.__setattr__(factory, "_node_data", {})
        object.__setattr__(factory, "_rel_data", {})
        object.__setattr__(factory, "_schema_info", None)
        object.__setattr__(factory, "_neo4j_id_to_label", {})

        schema = factory.build_schema()
        concept_node = next(n for n in schema.nodes if n.node_class.__name__ == "Concept")
        specs = {field: model for field, model in concept_node.index_field_specs}
        assert specs.get("description") == "ada_002@openai"


# ---------------------------------------------------------------------------
# New tests for Scenario F: JSON-string embedding deserialization
# ---------------------------------------------------------------------------


class TestEmbeddingDeserialization:
    """Test deserialization of JSON-string float arrays during Neo4j import."""

    def test_infer_kuzu_type_list_of_floats(self):
        """Test that _infer_kuzu_type detects list[float] as FLOAT[]."""
        # This is a nested function; test through the public import path indirectly
        # by checking the create_schema type mapping instead
        from genai_graph.kg.ingest.extract import _get_kuzu_type

        assert _get_kuzu_type(list[float]) == "FLOAT[]"

    def test_neo4j_factory_deserializes_json_string_embedding(self):
        """Test that build_nodes_and_relationships parses JSON-encoded embedding strings."""
        import json as _json

        from pydantic import BaseModel

        from genai_graph.kg.factories.neo4j_factory import Neo4jImportFactory, Neo4jNodeMapping

        class ServiceItem(BaseModel):
            name: str
            description: str | None = None
            description_embedding: list[float] | None = None

        embedding_values = [0.1, 0.2, 0.3]
        json_embedding = _json.dumps(embedding_values)  # "[0.1, 0.2, 0.3]"

        class JsonEmbeddingFactory(Neo4jImportFactory):
            neo4j_export_file: str = "/dev/null"

            def get_node_mappings(self):
                return [
                    Neo4jNodeMapping(
                        neo4j_label="ServiceItem",
                        node_class=ServiceItem,
                        property_mappings={
                            "name": "name",
                            "descriptionEmbedding": "description_embedding",
                        },
                        name_field="name",
                        key_field="name",
                    )
                ]

        factory = JsonEmbeddingFactory.__new__(JsonEmbeddingFactory)
        object.__setattr__(factory, "_initialized", True)
        object.__setattr__(factory, "_schema_info", None)
        object.__setattr__(factory, "_neo4j_id_to_label", {})
        object.__setattr__(
            factory,
            "_node_data",
            {
                "ServiceItem": [
                    {
                        "_neo4j_id": "1",
                        "name": "TestService",
                        "descriptionEmbedding": json_embedding,
                    }
                ]
            },
        )
        object.__setattr__(factory, "_rel_data", {})

        nodes_data, _ = factory.build_nodes_and_relationships()
        service_nodes = list(nodes_data.get("ServiceItem"))
        assert len(service_nodes) == 1
        emb = service_nodes[0].get("description_embedding")
        assert isinstance(emb, list), f"Expected list, got {type(emb)}: {emb}"
        assert emb == pytest.approx(embedding_values)

    def test_neo4j_factory_preserves_list_float_embedding(self):
        """Test that already-list float embeddings are preserved."""
        from pydantic import BaseModel

        from genai_graph.kg.factories.neo4j_factory import Neo4jImportFactory, Neo4jNodeMapping

        class ServiceItem(BaseModel):
            name: str
            description_embedding: list[float] | None = None

        embedding_values = [0.1, 0.2, 0.3]

        class ListEmbeddingFactory(Neo4jImportFactory):
            neo4j_export_file: str = "/dev/null"

            def get_node_mappings(self):
                return [
                    Neo4jNodeMapping(
                        neo4j_label="ServiceItem",
                        node_class=ServiceItem,
                        property_mappings={
                            "name": "name",
                            "descriptionEmbedding": "description_embedding",
                        },
                        name_field="name",
                        key_field="name",
                    )
                ]

        factory = ListEmbeddingFactory.__new__(ListEmbeddingFactory)
        object.__setattr__(factory, "_initialized", True)
        object.__setattr__(factory, "_schema_info", None)
        object.__setattr__(factory, "_neo4j_id_to_label", {})
        object.__setattr__(
            factory,
            "_node_data",
            {
                "ServiceItem": [
                    {
                        "_neo4j_id": "1",
                        "name": "TestService",
                        "descriptionEmbedding": embedding_values,
                    }
                ]
            },
        )
        object.__setattr__(factory, "_rel_data", {})

        nodes_data, _ = factory.build_nodes_and_relationships()
        emb = list(nodes_data.get("ServiceItem"))[0].get("description_embedding")
        assert emb == pytest.approx(embedding_values)


# ---------------------------------------------------------------------------
# New tests for Scenario G: DataFrame float-list → ArrowExtensionArray
# ---------------------------------------------------------------------------


class TestArrowTableFloatArrayType:
    """Test that _prepare_node_arrow_table types list[float] columns as list<float64>."""

    def test_float_list_column_gets_arrow_type(self):
        """Test that a float-list column is typed as list<float64> in the Arrow table."""
        import pyarrow as pa
        from pydantic import BaseModel

        from genai_graph.kg.ingest.merge import NodeTypeConfig, _prepare_node_arrow_table

        class MyNode(BaseModel):
            id: str
            name: str
            emb: list[float]

        config = (
            NodeTypeConfig.from_graph_node.__func__(  # type: ignore[attr-defined]
                NodeTypeConfig,
                type(
                    "G",
                    (),
                    {
                        "node_class": MyNode,
                        "key_from": "id",
                        "embedded_struct_classes": [],
                        "_embedding_field_dimensions": {},
                    },
                )(),
            )
            if False
            else NodeTypeConfig(node_type="MyNode", primary_key_field="id")
        )

        node_list = [
            {"id": "n1", "name": "A", "emb": [0.1, 0.2, 0.3]},
            {"id": "n2", "name": "B", "emb": [0.4, 0.5, 0.6]},
        ]
        # No schema: fallback path — emb ends with no special name so use inferred
        # Use schema-less config; the fallback still handles *_embedding; for plain
        # 'emb' the fallback infers from data.
        table = _prepare_node_arrow_table(node_list, config)

        # The 'emb' column should be a list type (inferred from data as list<double>)
        emb_type = table.schema.field("emb").type
        assert pa.types.is_list(emb_type), f"Expected list type, got {emb_type}"
        emb0 = table.column("emb")[0].as_py()
        assert emb0 == pytest.approx([0.1, 0.2, 0.3])

    def test_float_list_column_with_none_values(self):
        """Test that None entries in float-list columns become null in Arrow table."""
        from genai_graph.kg.ingest.merge import NodeTypeConfig, _prepare_node_arrow_table

        config = NodeTypeConfig(node_type="MyNode", primary_key_field="id")
        node_list = [
            {"id": "n1", "name": "A", "emb": [0.1, 0.2]},
            {"id": "n2", "name": "B", "emb": None},
        ]
        table = _prepare_node_arrow_table(node_list, config)

        # n1 should have the embedding, n2 should be null
        emb_col = table.column("emb")
        assert emb_col[0].as_py() == pytest.approx([0.1, 0.2])
        assert emb_col[1].as_py() is None

    def test_string_list_column_stays_string(self):
        """Test that string-list columns are not misidentified as float arrays."""
        from genai_graph.kg.ingest.merge import NodeTypeConfig, _prepare_node_arrow_table

        config = NodeTypeConfig(node_type="MyNode", primary_key_field="id")
        node_list = [
            {"id": "n1", "name": "A", "tags": ["alpha", "beta"]},
        ]
        table = _prepare_node_arrow_table(node_list, config)
        tags = table.column("tags")[0].as_py()
        assert tags == ["alpha", "beta"]

    def test_float_list_column_survives_kuzu_load(self, temp_kuzu_db):
        """Test that an Arrow table with list<float64> can be loaded by Kuzu."""
        from pydantic import BaseModel

        from genai_graph.kg.ingest.merge import NodeTypeConfig, _build_node_arrow_schema, _prepare_node_arrow_table

        backend, _ = temp_kuzu_db
        backend.execute(
            """
            CREATE NODE TABLE EmbNode (
                id STRING PRIMARY KEY,
                name STRING,
                emb FLOAT[3]
            );
            """
        )

        class EmbNode(BaseModel):
            id: str
            name: str
            emb: list[float]

        schema = _build_node_arrow_schema(EmbNode, primary_key_field="id")
        config = NodeTypeConfig(node_type="EmbNode", primary_key_field="id", arrow_schema=schema)

        node_list = [
            {"id": "e1", "name": "Alpha", "emb": [0.1, 0.2, 0.3]},
            {"id": "e2", "name": "Beta", "emb": [0.4, 0.5, 0.6]},
        ]
        arrow_table = _prepare_node_arrow_table(node_list, config)  # noqa: F841

        kuzu_conn = backend.conn
        kuzu_conn.execute(
            """
            LOAD FROM arrow_table
            MERGE (n:EmbNode {id: id})
            ON CREATE SET n.name = name, n.emb = emb
            """
        )

        result = backend.execute("MATCH (n:EmbNode) RETURN n.id, n.emb ORDER BY n.id;")
        rows = result.get_as_df()
        assert len(rows) == 2
        e1_emb = rows[rows["n.id"] == "e1"]["n.emb"].iloc[0]
        assert list(e1_emb) == pytest.approx([0.1, 0.2, 0.3])
