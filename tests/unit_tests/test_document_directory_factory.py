"""Unit tests for generic Folder/Document node types and DocumentDirectoryFactory."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
class TestDocumentModel:
    """Tests for the generic Document model."""

    def test_required_fields(self) -> None:
        from genai_graph.kg.nodes.document import Document

        doc = Document(content_hash="abc123", path="/data/file.md", filename="file.md")
        assert doc.path == "/data/file.md"
        assert doc.filename == "file.md"

    def test_optional_defaults(self) -> None:
        from genai_graph.kg.nodes.document import Document

        doc = Document(content_hash="abc123", path="/data/file.md", filename="file.md")
        assert doc.file_size is None
        assert doc.mime_type is None
        assert doc.token_count == 0
        assert doc.section_count == 0
        assert doc.access_level == "public"
        assert doc.allowed_roles == []
        assert doc.allowed_users == []

    def test_access_control(self) -> None:
        from genai_graph.kg.nodes.document import Document

        doc = Document(
            content_hash="secret123",
            path="/data/secret.md",
            filename="secret.md",
            access_level="confidential",
            allowed_roles=["admin"],
        )
        assert doc.access_level == "confidential"
        assert doc.allowed_roles == ["admin"]


@pytest.mark.unit
class TestFolderModel:
    """Tests for the generic Folder model."""

    def test_required_fields(self) -> None:
        from genai_graph.kg.nodes.document import Folder

        folder = Folder(folder_id="folder_1", uri="/data/docs", name="docs")
        assert folder.folder_id == "folder_1"
        assert folder.kind == "directory"


@pytest.mark.unit
class TestGraphNodeSingletons:
    """Tests for generic FolderNode and DocumentNode singletons."""

    def test_document_node_keys(self) -> None:
        from genai_graph.kg.nodes.document import Document, DocumentNode

        assert DocumentNode.key_from == "content_hash"
        assert DocumentNode.name_from == "filename"
        assert DocumentNode.node_class is Document
        assert DocumentNode.explicitly_defined is True

    def test_folder_node_keys(self) -> None:
        from genai_graph.kg.nodes.document import Folder, FolderNode

        assert FolderNode.key_from == "folder_id"
        assert FolderNode.node_class is Folder

    def test_contains_relation(self) -> None:
        from genai_graph.kg.nodes.document import CONTAINS_DOC, DocumentNode, FolderNode

        assert CONTAINS_DOC.name == "CONTAINS"
        assert CONTAINS_DOC.from_node is FolderNode
        assert CONTAINS_DOC.to_node is DocumentNode


@pytest.mark.unit
class TestDocumentDirectoryFactory:
    """Tests for the DocumentDirectoryFactory (Document-only)."""

    def test_build_schema_has_document(self) -> None:
        from genai_graph.kg.factories.document_factory import DocumentDirectoryFactory

        factory = DocumentDirectoryFactory(data_root="/tmp/docs")
        schema = factory.build_schema()

        node_names = [n.node_class.__name__ for n in schema.nodes]
        assert "Document" in node_names
        assert "Chunk" not in node_names

    def test_build_document_from_file(self, tmp_path: Path) -> None:
        from genai_graph.kg.factories.document_factory import DocumentDirectoryFactory

        test_file = tmp_path / "sample.md"
        test_file.write_text("# Hello\n\nThis is a test document.")

        factory = DocumentDirectoryFactory(data_root=str(tmp_path))
        doc = factory.get_struct_data_by_key(str(test_file))

        assert doc is not None
        assert doc.filename == "sample.md"
        assert doc.path == str(test_file)
        assert doc.file_size > 0

    def test_get_keys_discovers_files(self, tmp_path: Path) -> None:
        from genai_graph.kg.factories.document_factory import DocumentDirectoryFactory

        (tmp_path / "a.md").write_text("# A")
        (tmp_path / "b.txt").write_text("B content")
        (tmp_path / "skip.json").write_text("{}")

        factory = DocumentDirectoryFactory(
            data_root=str(tmp_path),
            include=["*.md", "*.txt"],
        )
        keys = factory.get_keys()

        filenames = [Path(k).name for k in keys]
        assert "a.md" in filenames
        assert "b.txt" in filenames
        assert "skip.json" not in filenames

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        from genai_graph.kg.factories.document_factory import DocumentDirectoryFactory

        factory = DocumentDirectoryFactory(data_root=str(tmp_path))
        result = factory.get_struct_data_by_key(str(tmp_path / "nonexistent.md"))

        assert result is None

    def test_sample_queries_present(self) -> None:
        from genai_graph.kg.factories.document_factory import DocumentDirectoryFactory

        factory = DocumentDirectoryFactory(data_root="/tmp")
        queries = factory.get_sample_queries()

        assert len(queries) > 0
        assert any("Document" in q for q in queries)
