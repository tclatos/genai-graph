"""Unit tests for the orchestration DAG, caching, and models."""

from __future__ import annotations

import threading

import pytest

from genai_graph.orchestration.dag import resolve_import_dag
from genai_graph.orchestration.models import BundleResult, ImportResult, WarningsCollector

# ---------------------------------------------------------------------------
# WarningsCollector tests
# ---------------------------------------------------------------------------


class TestWarningsCollector:
    def test_add_dedup(self) -> None:
        wc = WarningsCollector(source="test")
        wc.add("warning A")
        wc.add("warning B")
        wc.add("warning A")  # duplicate
        assert wc.count == 2
        assert wc.warnings == ["warning A", "warning B"]

    def test_merge_with_prefix(self) -> None:
        parent = WarningsCollector(source="parent")
        parent.add("parent warning")

        child = WarningsCollector(source="child_kg")
        child.add("child warning 1")
        child.add("child warning 2")

        parent.merge(child)

        assert parent.count == 3
        assert "[child_kg] child warning 1" in parent.warnings
        assert "[child_kg] child warning 2" in parent.warnings

    def test_merge_no_source(self) -> None:
        parent = WarningsCollector()
        child = WarningsCollector()
        child.add("w1")

        parent.merge(child)
        assert parent.warnings == ["w1"]

    def test_empty(self) -> None:
        wc = WarningsCollector()
        assert wc.count == 0
        assert wc.warnings == []

    def test_serialization(self) -> None:
        wc = WarningsCollector(source="s", warnings=["a", "b"])
        data = wc.model_dump()
        wc2 = WarningsCollector.model_validate(data)
        assert wc2.source == "s"
        assert wc2.warnings == ["a", "b"]


# ---------------------------------------------------------------------------
# ImportDag / resolve_import_dag tests
# ---------------------------------------------------------------------------


class _FakeConfig:
    """Minimal stub that has model_dump() like KgProfileConfig."""

    def __init__(self, imports: list[str] | None = None) -> None:
        self._imports = imports or []

    def model_dump(self) -> dict:
        return {"imports": self._imports, "graphs": []}


class TestResolveImportDag:
    def test_no_imports(self) -> None:
        configs = {"root": _FakeConfig()}
        dag = resolve_import_dag("root", configs)
        assert dag.root == "root"
        assert dag.execution_order == []

    def test_single_import(self) -> None:
        configs = {
            "root": _FakeConfig(imports=["dep_a"]),
            "dep_a": _FakeConfig(),
        }
        dag = resolve_import_dag("root", configs)
        assert len(dag.execution_order) == 1
        assert dag.execution_order[0].config_name == "dep_a"

    def test_diamond_dependency(self) -> None:
        """Diamond: root → A, B; both A and B → C."""
        configs = {
            "root": _FakeConfig(imports=["A", "B"]),
            "A": _FakeConfig(imports=["C"]),
            "B": _FakeConfig(imports=["C"]),
            "C": _FakeConfig(),
        }
        dag = resolve_import_dag("root", configs)
        names = [n.config_name for n in dag.execution_order]
        # C must appear before both A and B
        assert names.index("C") < names.index("A")
        assert names.index("C") < names.index("B")
        # Each appears exactly once
        assert sorted(names) == ["A", "B", "C"]

    def test_chain_dependency(self) -> None:
        """Chain: root → A → B → C."""
        configs = {
            "root": _FakeConfig(imports=["A"]),
            "A": _FakeConfig(imports=["B"]),
            "B": _FakeConfig(imports=["C"]),
            "C": _FakeConfig(),
        }
        dag = resolve_import_dag("root", configs)
        names = [n.config_name for n in dag.execution_order]
        assert names == ["C", "B", "A"]

    def test_circular_import_detected(self) -> None:
        configs = {
            "root": _FakeConfig(imports=["A"]),
            "A": _FakeConfig(imports=["B"]),
            "B": _FakeConfig(imports=["A"]),
        }
        with pytest.raises(ValueError, match="Circular import"):
            resolve_import_dag("root", configs)

    def test_missing_config_treated_as_leaf(self) -> None:
        """If a referenced import doesn't exist in configs, it's treated as a leaf."""
        configs = {
            "root": _FakeConfig(imports=["missing_dep"]),
        }
        dag = resolve_import_dag("root", configs)
        assert len(dag.execution_order) == 1
        assert dag.execution_order[0].config_name == "missing_dep"

    def test_root_not_in_execution_order(self) -> None:
        configs = {
            "root": _FakeConfig(imports=["A"]),
            "A": _FakeConfig(),
        }
        dag = resolve_import_dag("root", configs)
        assert all(n.config_name != "root" for n in dag.execution_order)

    def test_dict_configs(self) -> None:
        """Also works with plain dicts instead of model objects."""
        configs = {
            "root": {"imports": ["dep"], "graphs": []},
            "dep": {"imports": [], "graphs": []},
        }
        dag = resolve_import_dag("root", configs)
        assert len(dag.execution_order) == 1


# ---------------------------------------------------------------------------
# BundleResult / ImportResult tests
# ---------------------------------------------------------------------------


class TestResultModels:
    def test_bundle_result_defaults(self) -> None:
        br = BundleResult()
        assert br.stats.total_processed == 0
        assert br.warnings.count == 0

    def test_import_result(self) -> None:
        ir = ImportResult(config_name="crm_export", nodes_imported=10, rels_imported=5)
        assert ir.config_name == "crm_export"
        assert not ir.skipped


# ---------------------------------------------------------------------------
# CacheFingerprints tests (Phase 2)
# ---------------------------------------------------------------------------


class TestCacheFingerprints:
    def test_matches_all_none(self) -> None:
        from genai_graph.kg.export.artifacts import CacheFingerprints, ParquetManifest

        fp = CacheFingerprints()
        manifest = ParquetManifest(
            config_name="x",
            exported_at="2025-01-01",
            node_tables=[],
            rel_tables=[],
            node_count=0,
            rel_count=0,
        )
        assert fp.matches(manifest)

    def test_matches_when_equal(self) -> None:
        from genai_graph.kg.export.artifacts import CacheFingerprints, ParquetManifest

        fp = CacheFingerprints(schema_fingerprint="abc123", factory_config_hash="def456")
        manifest = ParquetManifest(
            config_name="x",
            exported_at="2025-01-01",
            node_tables=[],
            rel_tables=[],
            node_count=0,
            rel_count=0,
            schema_fingerprint="abc123",
            factory_config_hash="def456",
        )
        assert fp.matches(manifest)

    def test_mismatch_on_schema(self) -> None:
        from genai_graph.kg.export.artifacts import CacheFingerprints, ParquetManifest

        fp = CacheFingerprints(schema_fingerprint="new_hash")
        manifest = ParquetManifest(
            config_name="x",
            exported_at="2025-01-01",
            node_tables=[],
            rel_tables=[],
            node_count=0,
            rel_count=0,
            schema_fingerprint="old_hash",
        )
        assert not fp.matches(manifest)
        reasons = fp.mismatch_reasons(manifest)
        assert len(reasons) == 1
        assert "schema structure" in reasons[0]

    def test_mismatch_multiple_reasons(self) -> None:
        from genai_graph.kg.export.artifacts import CacheFingerprints, ParquetManifest

        fp = CacheFingerprints(
            schema_fingerprint="new_schema",
            factory_config_hash="new_config",
            source_content_hash="new_content",
        )
        manifest = ParquetManifest(
            config_name="x",
            exported_at="2025-01-01",
            node_tables=[],
            rel_tables=[],
            node_count=0,
            rel_count=0,
            schema_fingerprint="old_schema",
            factory_config_hash="old_config",
            source_content_hash="old_content",
        )
        assert not fp.matches(manifest)
        reasons = fp.mismatch_reasons(manifest)
        assert len(reasons) == 3

    def test_none_current_does_not_mismatch(self) -> None:
        """If current fingerprint is None (could not compute), don't invalidate."""
        from genai_graph.kg.export.artifacts import CacheFingerprints, ParquetManifest

        fp = CacheFingerprints(schema_fingerprint=None)
        manifest = ParquetManifest(
            config_name="x",
            exported_at="2025-01-01",
            node_tables=[],
            rel_tables=[],
            node_count=0,
            rel_count=0,
            schema_fingerprint="some_hash",
        )
        assert fp.matches(manifest)

    def test_legacy_manifest_without_fingerprints(self) -> None:
        """Manifests without fingerprint fields are treated as valid (backward compat)."""
        from genai_graph.kg.export.artifacts import CacheFingerprints, ParquetManifest

        fp = CacheFingerprints(schema_fingerprint="abc123")
        manifest = ParquetManifest(
            config_name="x",
            exported_at="2025-01-01",
            node_tables=[],
            rel_tables=[],
            node_count=0,
            rel_count=0,
        )
        # manifest has None fingerprints — still matches
        assert fp.matches(manifest)

    def test_serialization_roundtrip(self) -> None:
        from genai_graph.kg.export.artifacts import CacheFingerprints

        fp = CacheFingerprints(schema_fingerprint="abc", factory_config_hash="def", source_content_hash="ghi")
        data = fp.model_dump()
        fp2 = CacheFingerprints.model_validate(data)
        assert fp2.schema_fingerprint == "abc"
        assert fp2.factory_config_hash == "def"
        assert fp2.source_content_hash == "ghi"


# ---------------------------------------------------------------------------
# GraphSchema.fingerprint() tests (Phase 2)
# ---------------------------------------------------------------------------


class TestGraphSchemaFingerprint:
    def _make_schema(self, *, extra_node: bool = False):
        """Build a minimal GraphSchema for testing fingerprint stability."""
        from pydantic import BaseModel as PydanticBaseModel

        from genai_graph.kg.schema.core import GraphNode, GraphRelation, GraphSchema

        class Person(PydanticBaseModel):
            name: str
            age: int | None = None

        class Company(PydanticBaseModel):
            name: str

        nodes = [
            GraphNode(node_class=Person, name_from="name"),
            GraphNode(node_class=Company, name_from="name"),
        ]
        person_node, company_node = nodes[0], nodes[1]
        if extra_node:

            class Project(PydanticBaseModel):
                name: str

            nodes.append(GraphNode(node_class=Project, name_from="name"))

        rels = [
            GraphRelation(from_node=person_node, to_node=company_node, name="WORKS_AT"),
        ]

        return GraphSchema(root_model_class=Person, nodes=nodes, relations=rels)

    def test_deterministic(self) -> None:
        s1 = self._make_schema()
        s2 = self._make_schema()
        assert s1.fingerprint() == s2.fingerprint()

    def test_changes_with_extra_node(self) -> None:
        s1 = self._make_schema()
        s2 = self._make_schema(extra_node=True)
        assert s1.fingerprint() != s2.fingerprint()

    def test_returns_hex_string(self) -> None:
        s = self._make_schema()
        fp = s.fingerprint()
        assert isinstance(fp, str)
        assert len(fp) > 0
        # xxh3_64 produces 16-char hex
        int(fp, 16)  # Should not raise


# ---------------------------------------------------------------------------
# KgFactory.config_fingerprint() tests (Phase 2)
# ---------------------------------------------------------------------------


class TestKgFactoryConfigFingerprint:
    def test_deterministic(self) -> None:
        from typing import ClassVar

        from pydantic import BaseModel as PydanticBaseModel

        from genai_graph.kg.factories.base import KgFactory
        from genai_graph.kg.schema.core import GraphNode, GraphSchema

        class Dummy(PydanticBaseModel):
            name: str

        class FakeFactory(KgFactory):
            TOP_CLASS: ClassVar[type[PydanticBaseModel] | None] = Dummy
            data_root: str = "/tmp/data"
            include: list[str] = ["*.json"]

            def get_struct_data_by_key(self, key: str) -> PydanticBaseModel | None:
                return None

            def build_schema(self) -> GraphSchema:
                return GraphSchema(
                    root_model_class=Dummy,
                    nodes=[GraphNode(node_class=Dummy, name_from="name")],
                    relations=[],
                )

        f1 = FakeFactory(data_root="/tmp/a")
        f2 = FakeFactory(data_root="/tmp/a")
        assert f1.config_fingerprint() == f2.config_fingerprint()

        f3 = FakeFactory(data_root="/tmp/b")
        assert f1.config_fingerprint() != f3.config_fingerprint()


# ---------------------------------------------------------------------------
# ParquetCollector thread safety (Phase 3)
# ---------------------------------------------------------------------------


class TestParquetCollectorThreadSafety:
    def test_concurrent_add_nodes(self) -> None:
        import pyarrow as pa

        from genai_graph.kg.ingest.merge import ParquetCollector

        collector = ParquetCollector()
        barrier = threading.Barrier(4)

        def add_nodes(thread_id: int) -> None:
            barrier.wait()
            table = pa.table({"name": [f"node_{thread_id}"]})
            collector.add_nodes("Person", table)

        threads = [threading.Thread(target=add_nodes, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert collector.get_node_count() == 4

    def test_concurrent_add_relationships(self) -> None:
        import pyarrow as pa

        from genai_graph.kg.ingest.merge import ParquetCollector

        collector = ParquetCollector()
        barrier = threading.Barrier(4)

        def add_rels(thread_id: int) -> None:
            barrier.wait()
            table = pa.table({"from_id": [f"a_{thread_id}"], "to_id": [f"b_{thread_id}"]})
            collector.add_relationships("KNOWS", table)

        threads = [threading.Thread(target=add_rels, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert collector.get_relationship_count() == 4
