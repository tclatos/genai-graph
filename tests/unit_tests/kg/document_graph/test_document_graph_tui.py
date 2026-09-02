"""Tests for the Document Graph Textual TUI (`cli docgraph tui`)."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Markdown, Tree

from genai_graph.kg.backend import KuzuBackend
from genai_graph.kg.document_graph.ingest import ingest_document_graph
from genai_graph.kg.factories.document_graph_factory import DocumentGraphFactory
from genai_graph.kg.query.document_graph_tui import DocumentGraphApp, _dedupe_documents, _read_origin_path

DOC_TEXT = """<!-- source: /path/to/original_guide.pdf -->
# Guide Title

Introduction paragraph with general information.

## Section 1

Content of section 1.

## Section 2

Content of section 2.

### Subsection 2.1

Content of subsection 2.1.
"""


@pytest.fixture
def sample_tui_db(temp_db_path: str, tmp_path: Path) -> str:
    (tmp_path / "guide.md").write_text(DOC_TEXT, encoding="utf-8")
    backend = KuzuBackend()
    backend.connect(temp_db_path)
    ingest_document_graph(backend, DocumentGraphFactory(sources=[str(tmp_path)]))
    return temp_db_path


def test_read_origin_path(tmp_path: Path) -> None:
    f = tmp_path / "test.md"
    f.write_text("<!-- source: /docs/annual_report.pdf -->\n# Report\n", encoding="utf-8")
    assert _read_origin_path(str(f)) == "/docs/annual_report.pdf"

    f2 = tmp_path / "no_origin.md"
    f2.write_text("# Plain Markdown\n", encoding="utf-8")
    assert _read_origin_path(str(f2)) is None
    assert _read_origin_path(None) is None


def test_dedupe_documents() -> None:
    rows = [
        {"filename": "doc.md", "path": "dir/doc.md", "section_count": 2, "markdown_hash": "hash1"},
        {"filename": "doc.md", "path": "dir/doc.md", "section_count": 5, "markdown_hash": "hash2"},
    ]
    deduped = _dedupe_documents(rows)
    assert len(deduped) == 1
    assert deduped[0]["markdown_hash"] == "hash2"
    assert deduped[0]["section_count"] == 5


@pytest.mark.anyio
async def test_document_graph_tui_navigation(sample_tui_db: str) -> None:
    app = DocumentGraphApp(sample_tui_db)
    async with app.run_test() as pilot:
        tree = app.query_one(Tree)
        content = app.query_one("#content", Markdown)

        # 1. Expand root and folder
        tree.root.expand()
        await pilot.pause()
        folder_node = tree.root.children[0]
        folder_node.expand()
        await pilot.pause()

        # 2. Select document node
        assert len(folder_node.children) == 1
        doc_node = folder_node.children[0]
        tree.select_node(doc_node)

        # Allow worker to finish loading content
        for _ in range(50):
            await pilot.pause(0.02)
            if not app.query_one("#loading").has_class("active"):
                break

        assert "Guide Title" in content._markdown
        assert "Introduction paragraph" in content._markdown

        # 3. Expand doc node to reveal sections
        doc_node.expand()
        await pilot.pause()
        assert len(doc_node.children) >= 1

        # 4. Select a section node
        sec_node = doc_node.children[0]
        tree.select_node(sec_node)
        for _ in range(50):
            await pilot.pause(0.02)
            if not app.query_one("#loading").has_class("active"):
                break

        assert len(content._markdown) > 0


@pytest.mark.anyio
async def test_document_graph_tui_rapid_navigation(sample_tui_db: str) -> None:
    app = DocumentGraphApp(sample_tui_db)
    async with app.run_test() as pilot:
        tree = app.query_one(Tree)
        tree.root.expand()
        await pilot.pause()
        folder_node = tree.root.children[0]
        folder_node.expand()
        await pilot.pause()
        doc_node = folder_node.children[0]
        doc_node.expand()
        await pilot.pause()

        # Rapidly click through multiple nodes
        tree.select_node(doc_node)
        for child in doc_node.children:
            tree.select_node(child)

        # Allow final worker to complete
        for _ in range(50):
            await pilot.pause(0.02)
            if not app.query_one("#loading").has_class("active"):
                break

        content = app.query_one("#content", Markdown)
        assert content.display is True
        assert len(content._markdown) > 0
