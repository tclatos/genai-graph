"""Unit tests for DocumentGraphFactory (source resolution + document parsing)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from genai_graph.kg.factories.document_graph_factory import DocumentGraphFactory


@pytest.mark.unit
class TestDocumentGraphFactorySchema:
    def test_build_schema_has_document_and_section(self) -> None:
        factory = DocumentGraphFactory(sources=["/tmp/does-not-matter"])
        schema = factory.build_schema()

        node_names = [n.node_class.__name__ for n in schema.nodes]
        rel_names = [r.name for r in schema.relations]

        assert "Folder" in node_names
        assert "Document" in node_names
        assert "MarkdownSection" in node_names
        assert "CONTAINS" in rel_names
        assert "HAS_SUBFOLDER" in rel_names
        assert "HAS_SECTION" in rel_names
        assert "HAS_SUBSECTION" in rel_names


@pytest.mark.unit
class TestDocumentGraphFactorySources:
    def test_directory_source(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("# A\n\ntext\n")
        (tmp_path / "b.md").write_text("# B\n\ntext\n")
        (tmp_path / "ignore.txt").write_text("not markdown")

        factory = DocumentGraphFactory(sources=[str(tmp_path)])
        keys = factory.get_keys()

        assert len(keys) == 2
        assert all(k.endswith(".md") for k in keys)

    def test_single_file_source(self, tmp_path: Path) -> None:
        md_file = tmp_path / "one.md"
        md_file.write_text("# One\n\ntext\n")

        factory = DocumentGraphFactory(sources=[str(md_file)])
        keys = factory.get_keys()

        assert keys == [str(md_file)]

    def test_zip_source_is_extracted_and_scanned(self, tmp_path: Path) -> None:
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "doc.md").write_text("# Doc\n\ntext\n")

        zip_path = tmp_path / "archive.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(src_dir / "doc.md", arcname="doc.md")

        factory = DocumentGraphFactory(sources=[str(zip_path)], cache_dir=str(tmp_path / "cache"))
        keys = factory.get_keys()

        assert len(keys) == 1
        assert keys[0].endswith("doc.md")

    def test_mixed_sources_deduplicated(self, tmp_path: Path) -> None:
        md_file = tmp_path / "one.md"
        md_file.write_text("# One\n\ntext\n")

        factory = DocumentGraphFactory(sources=[str(tmp_path), str(md_file)])
        keys = factory.get_keys()

        assert len(keys) == 1


@pytest.mark.unit
class TestDocumentGraphFactoryParsing:
    def test_get_struct_data_by_key_builds_tree(self, tmp_path: Path) -> None:
        md_file = tmp_path / "doc.md"
        md_file.write_text("# Title\n\nintro\n\n## Sub\n\nbody\n")

        factory = DocumentGraphFactory(sources=[str(tmp_path)])
        bundle = factory.get_struct_data_by_key(str(md_file))

        assert bundle is not None
        assert bundle.document.path == str(md_file)
        assert bundle.document.filename == "doc.md"
        assert bundle.document.section_count == 2
        assert [s.title for s in bundle.sections] == ["Title", "Sub"]
        assert bundle.sections[0].parent_section_id is None
        assert bundle.sections[1].parent_section_id == bundle.sections[0].section_id
        content_hash = bundle.document.content_hash
        assert bundle.sections[0].section_id == f"{content_hash}::0"

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        factory = DocumentGraphFactory(sources=[str(tmp_path)])
        assert factory.get_struct_data_by_key(str(tmp_path / "missing.md")) is None


@pytest.mark.unit
class TestDocumentGraphFactoryFolderTree:
    def test_nested_subfolder_ancestor_chain(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        md_file = nested / "doc.md"
        md_file.write_text("# Title\n\ntext\n")

        factory = DocumentGraphFactory(sources=[str(tmp_path)])
        bundle = factory.get_struct_data_by_key(str(md_file))

        assert bundle is not None
        # root -> a -> b (immediate parent), 3 levels
        assert [f.name for f in bundle.folders] == [tmp_path.name, "a", "b"]
        assert bundle.folders[0].parent_folder_id is None
        assert bundle.folders[1].parent_folder_id == bundle.folders[0].folder_id
        assert bundle.folders[2].parent_folder_id == bundle.folders[1].folder_id
        assert bundle.document.folder_id == bundle.folders[-1].folder_id

    def test_folder_id_changes_when_content_changes(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        md_file = sub / "doc.md"
        md_file.write_text("# Title\n\ntext\n")

        factory = DocumentGraphFactory(sources=[str(tmp_path)])
        bundle_before = factory.get_struct_data_by_key(str(md_file))
        assert bundle_before is not None
        folder_id_before = bundle_before.folders[-1].folder_id

        md_file.write_text("# Title\n\nchanged text\n")
        factory_after = DocumentGraphFactory(sources=[str(tmp_path)])
        bundle_after = factory_after.get_struct_data_by_key(str(md_file))
        assert bundle_after is not None
        folder_id_after = bundle_after.folders[-1].folder_id

        assert folder_id_before != folder_id_after

    def test_unrelated_sibling_rename_does_not_change_folder_id(self, tmp_path: Path) -> None:
        sub_a = tmp_path / "a"
        sub_a.mkdir()
        (sub_a / "doc.md").write_text("# A\n\ntext\n")
        sub_b = tmp_path / "b"
        sub_b.mkdir()
        (sub_b / "doc.md").write_text("# B\n\ntext\n")

        factory = DocumentGraphFactory(sources=[str(tmp_path)])
        bundle_a_before = factory.get_struct_data_by_key(str(sub_a / "doc.md"))
        assert bundle_a_before is not None
        folder_a_id_before = bundle_a_before.folders[-1].folder_id

        (sub_b / "doc.md").rename(sub_b / "renamed.md")

        factory_after = DocumentGraphFactory(sources=[str(tmp_path)])
        bundle_a_after = factory_after.get_struct_data_by_key(str(sub_a / "doc.md"))
        assert bundle_a_after is not None
        folder_a_id_after = bundle_a_after.folders[-1].folder_id

        assert folder_a_id_before == folder_a_id_after
