"""Test improved error reporting for common KG creation issues."""

import pytest
from pydantic import BaseModel

from genai_graph.kg.backend import KuzuBackend
from genai_graph.kg.schema import GraphNode


class SimpleNode(BaseModel):
    """Test node with specific fields."""

    name: str
    valid_field: str


def test_missing_key_field_error_message():
    """Test that missing key field shows available fields."""
    node = GraphNode(
        node_class=SimpleNode,
        name_from="name",
        key_from="missing_field",  # This field doesn't exist
        description="Test node",
    )

    data = {
        "name": "Test",
        "valid_field": "value",
        "extra_field1": "data1",
        "extra_field2": "data2",
    }

    with pytest.raises(ValueError) as exc_info:
        node.get_key_value(data, "SimpleNode")

    error_msg = str(exc_info.value)
    # Check that error message includes helpful information
    assert "Key field 'missing_field' not found or empty" in error_msg
    assert "Available fields:" in error_msg
    assert "name" in error_msg  # Should show available fields
    assert "key_from='AUTO_ID'" in error_msg  # Should suggest AUTO_ID


def test_missing_key_field_shows_field_preview():
    """Test that error message previews available fields."""
    node = GraphNode(
        node_class=SimpleNode,
        name_from="name",
        key_from="id",
        description="Test node",
    )

    # Create data with many fields (16 total: 15 + name)
    data = {f"field_{i}": f"value_{i}" for i in range(15)}
    data["name"] = "Test"

    with pytest.raises(ValueError) as exc_info:
        node.get_key_value(data, "SimpleNode")

    error_msg = str(exc_info.value)
    # Should show preview of fields (first 10)
    assert "Available fields:" in error_msg
    assert "(and 6 more)" in error_msg  # 16 total - 10 shown = 6 more


def test_auto_id_never_fails():
    """Test that AUTO_ID always generates a valid key."""
    node = GraphNode(
        node_class=SimpleNode,
        name_from="name",
        key_from="AUTO_ID",
        description="Test node",
    )

    # Even with empty data (except name), AUTO_ID should work
    data = {"name": "Test"}

    key = node.get_key_value(data, "SimpleNode")
    assert key  # Should generate a UUID
    assert len(key) == 36  # UUID format


def test_computed_key_empty_error():
    """Test that computed key shows helpful error when empty."""
    node = GraphNode(
        node_class=SimpleNode,
        name_from="name",
        key_from=lambda data, node_type: data.get("missing_field", ""),
        description="Test node",
    )

    data = {"name": "Test"}

    with pytest.raises(ValueError) as exc_info:
        node.get_key_value(data, "SimpleNode")

    error_msg = str(exc_info.value)
    assert "Computed key is empty" in error_msg


# ---------------------------------------------------------------------------
# merge_nodes_batch error-enhancement path (real Ladybug backend, no mocks)
# ---------------------------------------------------------------------------


class _SchemaNode(BaseModel):
    id: str
    name: str
    score: float
    bad_field: str | None = None


class TestMergeNodesBatchErrorHandler:
    """Cover the error-enhancement branch in merge_nodes_batch using a real database.

    The 'Cannot find property' error is triggered for real by merging nodes whose
    Pydantic model declares a column that the actual DB table does not have.
    """

    def _registry_for(self, node_type: str = "_SchemaNode"):
        from genai_graph.kg.ingest.merge import NodeTypeConfig, NodeTypeRegistry, _build_node_arrow_schema

        schema = _build_node_arrow_schema(_SchemaNode, primary_key_field="id")
        config = NodeTypeConfig(node_type=node_type, primary_key_field="id", arrow_schema=schema)
        registry = NodeTypeRegistry()
        registry.register(config)
        return registry

    def _nodes(self, node_type: str = "_SchemaNode"):
        from genai_graph.kg.ingest.merge import NodeDataCollection

        nodes = NodeDataCollection()
        nodes.add(node_type, {"id": "n1", "name": "A", "score": 1.0, "bad_field": "x"})
        return nodes

    def test_cannot_find_property_logs_schema_fields(self, graph_backend: KuzuBackend):
        """Error-enhancement path must not raise AttributeError on config.field_names.

        Regression test: before the fix, the except block accessed config.field_types
        (removed attribute) and raised AttributeError, masking the real DB error.
        """
        from genai_graph.kg.ingest.merge import merge_nodes_batch

        # Real table WITHOUT the bad_field column declared in the Pydantic model
        graph_backend.execute("CREATE NODE TABLE _SchemaNode(id STRING, name STRING, score DOUBLE, PRIMARY KEY(id))")

        # Must raise the real DB error (Cannot find property bad_field), NOT an AttributeError
        with pytest.raises(RuntimeError, match="Cannot find property"):
            merge_nodes_batch(graph_backend, self._nodes(), self._registry_for())

    def test_generic_error_logged_without_attribute_error(self, graph_backend: KuzuBackend):
        """Generic (non-property) errors must propagate without AttributeError."""
        from genai_graph.kg.ingest.merge import merge_nodes_batch

        # Table does not exist at all -> generic binder error, not 'Cannot find property'
        with pytest.raises(RuntimeError) as exc_info:
            merge_nodes_batch(graph_backend, self._nodes(), self._registry_for())
        assert "Cannot find property" not in str(exc_info.value)

    def test_broken_formatter_does_not_mask_original_error(self, graph_backend: KuzuBackend):
        """A bug in the error-formatter must NEVER replace the original exception.

        This is the root cause of the original issue: the except block accessed
        config.field_types (gone after refactor), which raised AttributeError and
        replaced the real DB error in the caller's error log. The formatter is now
        wrapped in its own try/except so the original exception always propagates.
        """
        from unittest.mock import patch

        from genai_graph.kg.ingest.merge import merge_nodes_batch

        graph_backend.execute("CREATE NODE TABLE _SchemaNode(id STRING, name STRING, score DOUBLE, PRIMARY KEY(id))")
        registry = self._registry_for()
        config = registry.get("_SchemaNode")

        # Simulate a future regression: field_names is deleted/broken
        with patch.object(
            type(config),
            "field_names",
            new_callable=lambda: property(lambda self: (_ for _ in ()).throw(AttributeError("field_names gone"))),
        ):
            # The ORIGINAL RuntimeError must still propagate, NOT AttributeError from formatter
            with pytest.raises(RuntimeError, match="Cannot find property"):
                merge_nodes_batch(graph_backend, self._nodes(), registry)
