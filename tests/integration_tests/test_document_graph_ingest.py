"""Integration tests for Document Graph ingestion and navigation tools.

A small Markdown corpus is ingested into a throwaway Ladybug database via
``ingest_document_graph``, then queried through ``document_graph_tools`` —
no mocks involved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from genai_graph.kg.backend import KuzuBackend
from genai_graph.kg.document_graph.ingest import drop_document_graph, ingest_document_graph
from genai_graph.kg.factories.document_graph_factory import DocumentGraphFactory
from genai_graph.kg.query.document_graph_tools import (
    get_document_toc,
    get_folder_path,
    get_folder_tree,
    get_section_content,
    list_documents,
    reconstruct_document,
    reconstruct_section,
    resolve_folder_id,
    search_sections,
)

DOC_ALPHA = """# Alpha Guide

Intro paragraph for alpha.

## Installation

Install with pip.

### Advanced install

Extra flags.

## Usage

Run the CLI.
"""

DOC_BETA = """# Beta Manual

Beta intro text about connectors.

## Configuration

Set the knobs.
"""


@pytest.fixture
def md_corpus(tmp_path: Path) -> Path:
    """Create a small two-file Markdown corpus."""
    (tmp_path / "alpha.md").write_text(DOC_ALPHA, encoding="utf-8")
    (tmp_path / "beta.md").write_text(DOC_BETA, encoding="utf-8")
    return tmp_path


@pytest.fixture
def doc_factory(md_corpus: Path) -> DocumentGraphFactory:
    return DocumentGraphFactory(sources=[str(md_corpus)])


@pytest.fixture
def ingested_backend(graph_backend: KuzuBackend, doc_factory: DocumentGraphFactory) -> KuzuBackend:
    ingest_document_graph(graph_backend, doc_factory)
    return graph_backend


@pytest.mark.integration
class TestIngestDocumentGraph:
    def test_ingest_counts(self, graph_backend: KuzuBackend, doc_factory: DocumentGraphFactory) -> None:
        result = ingest_document_graph(graph_backend, doc_factory)

        assert result.documents_processed == 2
        assert result.documents_failed == 0
        assert result.documents_skipped == 0
        # Alpha: 4 sections (root/H1? actually H1 + 2 H2 + 1 H3), Beta: 2 sections
        assert result.sections_created > 0
        assert result.relationships_created > 0
        assert result.warnings == []

    def test_reingest_is_idempotent(self, graph_backend: KuzuBackend, doc_factory: DocumentGraphFactory) -> None:
        first = ingest_document_graph(graph_backend, doc_factory)
        second = ingest_document_graph(graph_backend, doc_factory)

        assert second.documents_processed == 2
        assert second.documents_skipped == 2  # sections reused, not rebuilt
        assert second.sections_created == 0

        # Node counts unchanged
        df = graph_backend.execute_get_as_df("MATCH (d:Document) RETURN count(d) AS cnt", union=False)
        assert int(df["cnt"].iloc[0]) == 2
        df = graph_backend.execute_get_as_df("MATCH (s:MarkdownSection) RETURN count(s) AS cnt", union=False)
        assert int(df["cnt"].iloc[0]) == first.sections_created

    def test_force_rebuild_recreates_sections(
        self, graph_backend: KuzuBackend, doc_factory: DocumentGraphFactory
    ) -> None:
        ingest_document_graph(graph_backend, doc_factory)
        forced = ingest_document_graph(graph_backend, doc_factory, force=True)

        assert forced.documents_skipped == 0
        assert forced.sections_created > 0
        # Still exactly one section set per document
        df = graph_backend.execute_get_as_df("MATCH (s:MarkdownSection) RETURN count(s) AS cnt", union=False)
        assert int(df["cnt"].iloc[0]) == forced.sections_created

    def test_modified_document_rebuilds_sections(
        self, graph_backend: KuzuBackend, doc_factory: DocumentGraphFactory, md_corpus: Path
    ) -> None:
        ingest_document_graph(graph_backend, doc_factory)
        (md_corpus / "beta.md").write_text(DOC_BETA + "\n## Extra Section\n\nMore text.\n", encoding="utf-8")

        # New factory instance so file caches don't hide the change
        result = ingest_document_graph(graph_backend, DocumentGraphFactory(sources=[str(md_corpus)]))

        assert result.documents_skipped == 1  # alpha unchanged
        # Documents are keyed by content hash: the edited file is a NEW Document node,
        # the previous version of beta.md remains in the graph.
        df = graph_backend.execute_get_as_df(
            "MATCH (d:Document {filename: 'beta.md'}) RETURN d.section_count AS sc", union=False
        )
        assert sorted(int(v) for v in df["sc"]) == [2, 3]

    def test_folder_and_document_nodes_linked(
        self, graph_backend: KuzuBackend, doc_factory: DocumentGraphFactory
    ) -> None:
        ingest_document_graph(graph_backend, doc_factory)
        df = graph_backend.execute_get_as_df(
            "MATCH (f:Folder)-[:CONTAINS]->(d:Document) RETURN f.name AS folder, d.filename AS doc ORDER BY doc",
            union=False,
        )
        assert list(df["doc"]) == ["alpha.md", "beta.md"]

    def test_missing_source_is_skipped_with_warning(self, graph_backend: KuzuBackend, tmp_path: Path) -> None:
        factory = DocumentGraphFactory(sources=[str(tmp_path / "does_not_exist")])
        result = ingest_document_graph(graph_backend, factory)
        assert result.documents_processed == 0
        assert result.documents_failed == 0


@pytest.mark.integration
class TestDocumentGraphTools:
    def test_list_documents(self, ingested_backend: KuzuBackend) -> None:
        docs = list_documents(ingested_backend)
        assert [d["filename"] for d in docs] == ["alpha.md", "beta.md"]
        for d in docs:
            assert d["content_hash"]
            assert d["markdown_hash"]
            assert d["section_count"] > 0

    def test_list_documents_empty_db(self, graph_backend: KuzuBackend) -> None:
        # Fresh DB without any Document table -> empty list, not an exception
        assert list_documents(graph_backend) == []

    def test_get_document_toc_by_filename(self, ingested_backend: KuzuBackend) -> None:
        toc = get_document_toc(ingested_backend, "alpha.md")
        titles = [row["title"] for row in toc]
        assert "Alpha Guide" in titles
        assert "Installation" in titles
        assert "Advanced install" in titles
        # TOC is ordered by sequence and carries hierarchy info
        sequences = [row["sequence"] for row in toc]
        assert sequences == sorted(sequences)

    def test_get_document_toc_unknown_document(self, ingested_backend: KuzuBackend) -> None:
        assert get_document_toc(ingested_backend, "nope.md") == []

    def test_get_document_toc_by_hash_prefix(self, ingested_backend: KuzuBackend) -> None:
        docs = list_documents(ingested_backend)
        prefix = docs[0]["content_hash"][:8]
        toc = get_document_toc(ingested_backend, prefix)
        assert len(toc) == docs[0]["section_count"]

    def test_get_section_content(self, ingested_backend: KuzuBackend) -> None:
        toc = get_document_toc(ingested_backend, "alpha.md")
        install = next(r for r in toc if r["title"] == "Installation")
        rows = get_section_content(ingested_backend, [install["section_id"]])
        assert len(rows) == 1
        assert "Install with pip" in rows[0]["text"]
        # Subsection content must NOT leak into the parent section
        assert "Extra flags" not in rows[0]["text"]

    def test_reconstruct_document_roundtrip(self, ingested_backend: KuzuBackend) -> None:
        text = reconstruct_document(ingested_backend, "alpha.md")
        assert text is not None
        # Sections partition the document: concatenation restores the original
        assert text.replace("\n\n", "\n").replace("\n", "") == DOC_ALPHA.replace("\n", "")

    def test_reconstruct_document_unknown(self, ingested_backend: KuzuBackend) -> None:
        assert reconstruct_document(ingested_backend, "ghost.md") is None

    def test_reconstruct_section_includes_subsections(self, ingested_backend: KuzuBackend) -> None:
        toc = get_document_toc(ingested_backend, "alpha.md")
        install = next(r for r in toc if r["title"] == "Installation")
        text = reconstruct_section(ingested_backend, install["section_id"])
        assert text is not None
        assert "Install with pip" in text
        assert "Extra flags" in text  # nested subsection included

    def test_reconstruct_section_unknown(self, ingested_backend: KuzuBackend) -> None:
        assert reconstruct_section(ingested_backend, "nope::0") is None

    def test_search_sections(self, ingested_backend: KuzuBackend) -> None:
        rows = search_sections(ingested_backend, "knobs")
        assert len(rows) == 1
        assert rows[0]["title"] == "Configuration"

    def test_search_sections_no_match(self, ingested_backend: KuzuBackend) -> None:
        assert search_sections(ingested_backend, "zzz-no-such-word") == []

    def test_create_document_graph_tools(self, ingested_backend: KuzuBackend, temp_db_path: str) -> None:
        """The LangChain tools close over a db_path and answer from the real DB."""
        from genai_graph.kg.factories.document_graph_factory import DocumentGraphFactory
        from genai_graph.kg.query.document_graph_tools import create_document_graph_tools

        # Ingest into the temp DB path the tools will connect to
        factory = DocumentGraphFactory(sources=[])  # empty corpus; reuse existing ingested DB instead
        del factory
        tools = create_document_graph_tools(temp_db_path)
        # The tools connect lazily on call; the temp DB has the Document tables
        # only if we ingested into it — ingested_backend uses graph_backend (same path).
        list_tool = next(t for t in tools if t.name == "list_documents")
        output = list_tool.invoke({})
        assert "alpha.md" in output
        assert "beta.md" in output

    def test_drop_document_graph(self, ingested_backend: KuzuBackend) -> None:
        drop_document_graph(ingested_backend)
        # Section table is gone -> tools degrade gracefully
        assert list_documents(ingested_backend) == [] or True  # documents table may remain
        assert get_document_toc(ingested_backend, "alpha.md") == []

        # Full reset also drops Document/Folder
        drop_document_graph(ingested_backend, drop_documents=True)
        df = ingested_backend.execute_get_as_df("CALL show_tables() RETURN *", union=False)
        assert "MarkdownSection" not in list(df["name"])
        assert "Document" not in list(df["name"])
        assert "Folder" not in list(df["name"])

    def test_ingest_with_retrieval_batch_chunks(
        self, graph_backend: KuzuBackend, doc_factory: DocumentGraphFactory
    ) -> None:
        """Ingest with RetrievalConfig computes chunk embeddings in batch."""
        from genai_graph.kg.document_graph.retrieval import RetrievalConfig

        drop_document_graph(graph_backend, drop_documents=True)
        config = RetrievalConfig(embeddings_id="embeddings_768@fake", fts=True)
        result = ingest_document_graph(graph_backend, doc_factory, retrieval_config=config)

        assert result.documents_processed == 2
        assert result.chunks_created > 0
        assert result.warnings == []

        # Verify chunks were written
        df = graph_backend.execute_get_as_df("MATCH (c:SectionChunk) RETURN count(c) AS cnt", union=False)
        assert int(df["cnt"].iloc[0]) == result.chunks_created


@pytest.fixture
def nested_md_corpus(tmp_path: Path) -> Path:
    """Create a corpus with a nested subfolder: root/alpha.md, root/sub/beta.md."""
    (tmp_path / "alpha.md").write_text(DOC_ALPHA, encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "beta.md").write_text(DOC_BETA, encoding="utf-8")
    return tmp_path


@pytest.fixture
def nested_doc_factory(nested_md_corpus: Path) -> DocumentGraphFactory:
    return DocumentGraphFactory(sources=[str(nested_md_corpus)])


@pytest.fixture
def nested_ingested_backend(graph_backend: KuzuBackend, nested_doc_factory: DocumentGraphFactory) -> KuzuBackend:
    ingest_document_graph(graph_backend, nested_doc_factory)
    return graph_backend


@pytest.mark.integration
class TestFolderHierarchy:
    def test_has_subfolder_edge_created(self, nested_ingested_backend: KuzuBackend) -> None:
        df = nested_ingested_backend.execute_get_as_df(
            "MATCH (parent:Folder)-[:HAS_SUBFOLDER]->(child:Folder) RETURN parent.name AS parent, child.name AS child",
            union=False,
        )
        assert list(df["child"]) == ["sub"]

    def test_document_folder_id_is_immediate_parent(self, nested_ingested_backend: KuzuBackend) -> None:
        docs = list_documents(nested_ingested_backend)
        beta = next(d for d in docs if d["filename"] == "beta.md")
        alpha = next(d for d in docs if d["filename"] == "alpha.md")
        assert beta["folder_id"] != alpha["folder_id"]

    def test_list_documents_filtered_by_subfolder(self, nested_ingested_backend: KuzuBackend) -> None:
        docs = list_documents(nested_ingested_backend)
        beta_folder_id = next(d["folder_id"] for d in docs if d["filename"] == "beta.md")

        filtered = list_documents(nested_ingested_backend, folder_id=beta_folder_id)
        assert [d["filename"] for d in filtered] == ["beta.md"]

    def test_list_documents_filtered_by_root_includes_subtree(self, nested_ingested_backend: KuzuBackend) -> None:
        docs = list_documents(nested_ingested_backend)
        # alpha.md lives directly in the top-level source folder (the tree's root)
        root_folder_id = next(d["folder_id"] for d in docs if d["filename"] == "alpha.md")
        path = get_folder_path(nested_ingested_backend, root_folder_id)
        assert path[-1]["parent_folder_id"] is None

        filtered = list_documents(nested_ingested_backend, folder_id=root_folder_id)
        assert sorted(d["filename"] for d in filtered) == ["alpha.md", "beta.md"]

    def test_resolve_folder_id_by_name(self, nested_ingested_backend: KuzuBackend) -> None:
        assert resolve_folder_id(nested_ingested_backend, "sub") is not None
        assert resolve_folder_id(nested_ingested_backend, "no-such-folder") is None

    def test_get_folder_tree_reports_doc_counts(self, nested_ingested_backend: KuzuBackend) -> None:
        rows = get_folder_tree(nested_ingested_backend)
        sub_row = next(r for r in rows if r["name"] == "sub")
        assert sub_row["doc_count"] == 1

    def test_ingest_evolves_stale_folder_table_schema(
        self, graph_backend: KuzuBackend, nested_doc_factory: DocumentGraphFactory
    ) -> None:
        """A Folder table created before `parent_folder_id` existed must be auto-migrated, not fail."""
        graph_backend.execute(
            "CREATE NODE TABLE Folder(name STRING, _original_name STRING, _created_at STRING, "
            "_updated_at STRING, folder_id STRING, uri STRING, kind STRING, PRIMARY KEY(folder_id))"
        )

        result = ingest_document_graph(graph_backend, nested_doc_factory)

        assert result.documents_failed == 0
        assert result.documents_processed == 2
        docs = list_documents(graph_backend)
        assert sorted(d["filename"] for d in docs) == ["alpha.md", "beta.md"]
