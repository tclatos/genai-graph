"""Integration tests for PDF markdownization, document graph ingestion, and querying.

Tests end-to-end injection and retrieval for `sample-pdf-a4-size.pdf` stored in `tests/data/`:
1. Converts PDF to Markdown via MarkItDownConverter.
2. Ingests the document into a Ladybug database via DocumentGraphFactory & ingest_document_graph.
3. Exercises TOC generation, keyword indexing, and section search.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from genai_tk.extra.markdownize.markitdown_converter import MarkItDownConverter

from genai_graph.kg.backend import KuzuBackend
from genai_graph.kg.document_graph.ingest import ingest_document_graph
from genai_graph.kg.document_graph.outline_extract import (
    BranchOutline,
    OutlineConfig,
    OutlineEntry,
)
from genai_graph.kg.factories.document_graph_factory import DocumentGraphFactory
from genai_graph.kg.query.document_graph_tools import (
    document_toc_yaml,
    get_document,
    get_document_toc,
    get_section_content,
    list_documents,
    search_sections,
)

SAMPLE_PDF_PATH = Path(__file__).parent.parent / "data" / "sample-pdf-a4-size.pdf"


@pytest.fixture(scope="module")
def sample_pdf_file() -> Path:
    """Ensure the sample PDF file exists."""
    if not SAMPLE_PDF_PATH.exists():
        import httpx

        url = "https://sample-files.com/downloads/documents/pdf/sample-pdf-a4-size.pdf"
        SAMPLE_PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
        resp = httpx.get(url, follow_redirects=True, timeout=30.0)
        resp.raise_for_status()
        SAMPLE_PDF_PATH.write_bytes(resp.content)
    return SAMPLE_PDF_PATH


@pytest.fixture
def converted_md_corpus(tmp_path: Path, sample_pdf_file: Path) -> Path:
    """Convert sample PDF to Markdown and save in a test corpus folder."""
    import asyncio

    converter = MarkItDownConverter()
    md_content = asyncio.run(converter.convert(sample_pdf_file))
    md_file = tmp_path / "sample_doc.md"
    md_file.write_text(md_content, encoding="utf-8")
    return tmp_path


def _fake_call_branch_llm(**kwargs: Any) -> BranchOutline:
    branch_headings = kwargs.get("branch_headings", [])
    entries = []
    for title, level, _ in branch_headings:
        entries.append(
            OutlineEntry(
                title=title,
                level=level,
                description=f"Market analysis and details for {title}.",
                summary=f"Substantial metrics and findings under {title}.",
                keywords=[title.lower(), "metrics", "analysis"],
            )
        )
    return BranchOutline(sections=entries)


def _fake_synthesize_doc_summary(**kwargs: Any) -> tuple[str, str]:  # noqa: ARG001
    return (
        "A sample business report on project objectives.",
        "Full summary of sample PDF report covering objectives, findings, and results.",
    )


@pytest.mark.integration
class TestPdfIngestIntegration:
    """End-to-end integration test for converting PDF, ingesting into Ladybug, and navigating."""

    def test_pdf_conversion_and_algorithmic_ingest(self, graph_backend: KuzuBackend, converted_md_corpus: Path) -> None:
        """Test algorithmic ingestion of markdown generated from sample-pdf-a4-size.pdf."""
        factory = DocumentGraphFactory(sources=[str(converted_md_corpus)])
        result = ingest_document_graph(graph_backend, factory)

        assert result.documents_processed == 1
        assert result.documents_failed == 0
        assert result.sections_created >= 5
        assert result.relationships_created >= 4

        # Verify Document node
        docs = list_documents(graph_backend)
        assert len(docs) == 1
        assert docs[0]["filename"] == "sample_doc.md"

        # Verify MarkdownSection nodes exist and are queryable
        toc_rows = get_document_toc(graph_backend, docs[0]["content_hash"])
        assert len(toc_rows) >= 5

        # Verify Section content retrieval
        first_sec = toc_rows[0]
        content_rows = get_section_content(graph_backend, [first_sec["section_id"]])
        assert len(content_rows) == 1
        assert len(content_rows[0]["text"]) > 0

    def test_pdf_llm_outline_ingest_with_keywords_and_search(
        self,
        graph_backend: KuzuBackend,
        converted_md_corpus: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test LLM-based outline extraction with section keywords and FTS search."""
        monkeypatch.setattr("genai_graph.kg.document_graph.outline_extract._context_window_for", lambda llm_id: None)
        monkeypatch.setattr("genai_graph.kg.document_graph.outline_extract._call_branch_llm", _fake_call_branch_llm)
        monkeypatch.setattr(
            "genai_graph.kg.document_graph.outline_extract._synthesize_document_summary", _fake_synthesize_doc_summary
        )

        outline_config = OutlineConfig(llm="test_llm@fake")
        factory = DocumentGraphFactory(sources=[str(converted_md_corpus)], outline_config=outline_config)
        result = ingest_document_graph(graph_backend, factory)

        assert result.documents_processed == 1
        assert result.sections_created >= 5
        assert result.sections_summarized >= 1

        # Check document summary and description
        doc_info = get_document(graph_backend, "sample_doc.md")
        assert doc_info is not None
        assert "sample business" in (doc_info.get("description") or "").lower()

        # Check TOC YAML formatting with keywords and summaries
        toc_yaml = document_toc_yaml(graph_backend, "sample_doc.md", include_summaries=True)
        assert "sample_doc.md" in toc_yaml
        assert "sections" in toc_yaml
        assert "Item 1" in toc_yaml or "Market" in toc_yaml

        # Check section search across content and keywords
        hits = search_sections(graph_backend, query="Market Analysis", mode="bm25")
        assert len(hits) >= 1
