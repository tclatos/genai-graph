"""Unit tests for the KG backend abstraction layer (no database required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from genai_graph.kg.backend import (
    KuzuBackend,
    LadybugBackend,
    Neo4jBackend,
    _normalize_db_config,
    create_backend,
    create_in_memory_backend,
)


class TestCreateBackend:
    def test_ladybug(self) -> None:
        backend = create_backend("ladybug")
        assert isinstance(backend, LadybugBackend)
        assert isinstance(backend, KuzuBackend)

    def test_kuzu(self) -> None:
        backend = create_backend("kuzu")
        assert isinstance(backend, LadybugBackend)
        assert isinstance(backend, KuzuBackend)

    def test_kuzu_case_insensitive(self) -> None:
        assert isinstance(create_backend("LADYBUG"), LadybugBackend)
        assert isinstance(create_backend("KUZU"), KuzuBackend)

    def test_neo4j(self) -> None:
        assert isinstance(create_backend("neo4j"), Neo4jBackend)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend type"):
            create_backend("cassandra")

    def test_in_memory(self) -> None:
        backend = create_in_memory_backend()
        assert isinstance(backend, LadybugBackend)
        # Connection is live
        backend.execute("CREATE NODE TABLE T(id STRING, PRIMARY KEY(id))")
        backend.close()


class TestNormalizeDbConfig:
    def test_bare_string_defaults_to_ladybug(self) -> None:
        assert _normalize_db_config("/tmp/x.db") == {"type": "ladybug", "path": "/tmp/x.db"}

    def test_mapping_passthrough(self) -> None:
        cfg = {"type": "neo4j", "path": "bolt://localhost"}
        assert _normalize_db_config(cfg) == cfg


class TestKuzuBackendBasics:
    def test_execute_without_connect_raises(self) -> None:
        backend = LadybugBackend()
        with pytest.raises(RuntimeError, match="Not connected"):
            backend.execute("MATCH (n) RETURN n")

    def test_get_query_language(self) -> None:
        assert LadybugBackend().get_query_language() == "Cypher"

    def test_close_resets_connection(self) -> None:
        backend = create_in_memory_backend()
        backend.close()
        assert backend.conn is None
        assert backend.db is None


class TestNeo4jBackendPlaceholder:
    """Neo4jBackend is a placeholder — every method must raise NotImplementedError."""

    def test_connect(self) -> None:
        with pytest.raises(NotImplementedError):
            Neo4jBackend().connect("bolt://localhost")

    def test_execute(self) -> None:
        with pytest.raises(NotImplementedError):
            Neo4jBackend().execute("MATCH (n) RETURN n")

    def test_create_node_table(self) -> None:
        with pytest.raises(NotImplementedError):
            Neo4jBackend().create_node_table("T", {"id": "STRING"}, "id")

    def test_create_relationship_table(self) -> None:
        with pytest.raises(NotImplementedError):
            Neo4jBackend().create_relationship_table("R", "A", "B")

    def test_drop_table(self) -> None:
        with pytest.raises(NotImplementedError):
            Neo4jBackend().drop_table("T")

    def test_insert_node(self) -> None:
        with pytest.raises(NotImplementedError):
            Neo4jBackend().insert_node("T", {"id": "1"})

    def test_insert_relationship(self) -> None:
        with pytest.raises(NotImplementedError):
            Neo4jBackend().insert_relationship("R", "A", "1", "B", "2")

    def test_close(self) -> None:
        with pytest.raises(NotImplementedError):
            Neo4jBackend().close()

    def test_get_query_language(self) -> None:
        assert Neo4jBackend().get_query_language() == "Cypher"


class TestExecuteGetAsDf:
    """Exercise the multi-result handling of execute_get_as_df with stub executors."""

    class _SingleResultBackend(KuzuBackend):
        """Backend returning a fake single result object."""

        def execute(self, query: str, parameters: dict | None = None):  # type: ignore[override]
            import pandas as pd

            class _Res:
                def get_as_df(self) -> pd.DataFrame:
                    return pd.DataFrame({"a": [1, 2]})

            return _Res()

    def test_single_result(self) -> None:
        backend = self._SingleResultBackend()
        df = backend.execute_get_as_df("MATCH (n) RETURN n")
        assert list(df["a"]) == [1, 2]

    def test_single_result_without_get_as_df_raises(self) -> None:
        class _BadBackend(KuzuBackend):
            def execute(self, query: str, parameters: dict | None = None):  # type: ignore[override]
                return 42

        with pytest.raises(AttributeError, match="does not support get_as_df"):
            _BadBackend().execute_get_as_df("MATCH (n) RETURN n")

    def test_empty_list_result(self) -> None:
        class _EmptyBackend(KuzuBackend):
            def execute(self, query: str, parameters: dict | None = None):  # type: ignore[override]
                return []

        df = _EmptyBackend().execute_get_as_df("MATCH (n) RETURN n")
        assert df.empty

    def test_multi_result_union_false_raises(self) -> None:
        import pandas as pd

        class _MultiBackend(KuzuBackend):
            def execute(self, query: str, parameters: dict | None = None):  # type: ignore[override]
                class _Res:
                    def get_as_df(self) -> pd.DataFrame:
                        return pd.DataFrame({"a": [1]})

                return [_Res(), _Res()]

        with pytest.raises(ValueError, match="union=False"):
            _MultiBackend().execute_get_as_df("Q1; Q2", union=False)

    def test_multi_result_union_concatenates(self) -> None:
        import pandas as pd

        class _MultiBackend(KuzuBackend):
            def execute(self, query: str, parameters: dict | None = None):  # type: ignore[override]
                class _Res:
                    def __init__(self, v: int) -> None:
                        self.v = v

                    def get_as_df(self) -> pd.DataFrame:
                        return pd.DataFrame({"a": [self.v]})

                return [_Res(1), _Res(2)]

        df = _MultiBackend().execute_get_as_df("Q1; Q2", union=True)
        assert list(df["a"]) == [1, 2]


class TestDeleteBackendStorage:
    def test_delete_file_and_wal(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """delete_backend_storage_from_config removes the DB file and its WAL."""
        from genai_graph.kg import backend as backend_mod

        db_file = tmp_path / "test.db"
        db_file.write_bytes(b"db")
        wal_file = tmp_path / "test.db.wal"
        wal_file.write_bytes(b"wal")

        monkeypatch.setattr(
            backend_mod,
            "get_backend_storage_path_from_config",
            lambda config_key="default", kg_config_name=None: db_file,
        )
        backend_mod.delete_backend_storage_from_config()

        assert not db_file.exists()
        assert not wal_file.exists()

    def test_delete_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from genai_graph.kg import backend as backend_mod

        db_dir = tmp_path / "db_dir"
        db_dir.mkdir()
        (db_dir / "nodes").write_bytes(b"x")

        monkeypatch.setattr(
            backend_mod,
            "get_backend_storage_path_from_config",
            lambda config_key="default", kg_config_name=None: db_dir,
        )
        backend_mod.delete_backend_storage_from_config()
        assert not db_dir.exists()

    def test_delete_nonexistent_with_orphan_wal(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from genai_graph.kg import backend as backend_mod

        db_file = tmp_path / "ghost.db"
        wal_file = tmp_path / "ghost.db.wal"
        wal_file.write_bytes(b"wal")

        monkeypatch.setattr(
            backend_mod,
            "get_backend_storage_path_from_config",
            lambda config_key="default", kg_config_name=None: db_file,
        )
        backend_mod.delete_backend_storage_from_config()  # must not raise
        assert not wal_file.exists()
