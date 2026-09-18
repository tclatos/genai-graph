"""Unit and integration tests for Document Graph Streamlit UI components and DocBench CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from genai_graph.core.commands_docbench import DocBenchCommands
from genai_graph.core.commands_docgraph import DocGraphCommands
from genai_graph.kg.backend import KuzuBackend
from genai_graph.kg.document_graph.ingest import ingest_document_graph
from genai_graph.kg.factories.document_graph_factory import DocumentGraphFactory
from genai_graph.webapp.pages.demos.docgraph_browser import _get_configured_profiles
from genai_graph.webapp.ui_components.docgraph_view import (
    build_tree_select_nodes,
    extract_markdown_images,
    fetch_database_stats,
    fetch_document_sections_full,
    fetch_folders_and_documents,
    fetch_section_markdown_async,
    format_section_expander_label,
    resolve_image_path,
)

SAMPLE_DOC_TEXT = """<!-- source: /path/to/annual_report.pdf -->
# Annual Financial Report 2026

Executive summary of annual corporate performance and key achievements.

## Financial Overview

Overview of financial performance across all quarters.

| Metric | Q1 | Q2 | Total |
|---|---|---|---|
| Revenue ($M) | 120 | 145 | 265 |
| Net Income ($M) | 30 | 38 | 68 |

<!-- Image: revenue_chart.png (hash: 7890abcd) -->
![Revenue Chart](revenue_chart.png "Revenue Growth by Quarter")

### Capital Expenditure

Capital allocation in cloud infrastructure and AI compute.
"""


@pytest.fixture
def sample_docgraph_db(temp_db_path: str, tmp_path: Path) -> tuple[str, Path]:
    doc_file = tmp_path / "annual_report.md"
    doc_file.write_text(SAMPLE_DOC_TEXT, encoding="utf-8")

    # Create dummy image file
    img_file = tmp_path / "revenue_chart.png"
    img_file.write_bytes(b"dummy image data")

    backend = KuzuBackend()
    backend.connect(temp_db_path)
    ingest_document_graph(backend, DocumentGraphFactory(sources=[str(tmp_path)]))
    return temp_db_path, tmp_path


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------


def test_format_section_expander_label_basic() -> None:
    section = {
        "title": "Introduction",
        "level": 1,
        "token_count": 250,
        "line_start": 5,
        "line_end": 20,
        "summary": "Covers company background and core missions.",
        "has_table": False,
        "has_image": False,
        "has_graph": False,
    }
    label = format_section_expander_label(section)

    assert "[H1]" in label
    assert "Introduction" in label
    assert "Covers company background" in label
    assert "250 tokens" in label
    assert "L5-L20" in label


def test_format_section_expander_label_with_indicators() -> None:
    section = {
        "title": "Quarterly Breakdown",
        "level": 2,
        "token_count": 890,
        "line_start": 30,
        "line_end": 75,
        "description": "Financial metrics with revenue tables and architecture charts",
        "has_table": True,
        "has_image": True,
        "has_graph": True,
    }
    label = format_section_expander_label(section)

    assert "[H2]" in label
    assert "Quarterly Breakdown" in label
    assert "📊 Table" in label
    assert "🖼️ Image" in label
    assert "🕸️ Graph" in label
    assert "890 tokens" in label


def test_build_tree_select_nodes_hierarchy() -> None:
    folders = [
        {"folder_id": "f1", "name": "Financials", "parent_folder_id": None, "doc_count": 1},
    ]
    docs = [
        {
            "markdown_hash": "doc_hash_1",
            "filename": "q1_report.md",
            "folder_id": "f1",
            "section_count": 2,
            "token_count": 500,
        },
    ]
    sections_by_doc = {
        "doc_hash_1": [
            {
                "section_id": "doc_hash_1::0",
                "title": "Overview",
                "level": 1,
                "parent_section_id": None,
                "token_count": 200,
                "has_table": False,
                "has_image": False,
            },
            {
                "section_id": "doc_hash_1::1",
                "title": "Details",
                "level": 2,
                "parent_section_id": "doc_hash_1::0",
                "token_count": 300,
                "has_table": True,
                "has_image": True,
            },
        ]
    }

    tree_nodes = build_tree_select_nodes(folders, docs, sections_by_doc=sections_by_doc, include_sections=True)

    assert len(tree_nodes) == 1
    folder_node = tree_nodes[0]
    assert folder_node["value"] == "folder:f1"
    assert "Financials" in folder_node["label"]

    assert len(folder_node["children"]) == 1
    doc_node = folder_node["children"][0]
    assert doc_node["value"] == "doc:doc_hash_1"
    assert "q1_report.md" in doc_node["label"]

    assert len(doc_node["children"]) == 1
    root_sec_node = doc_node["children"][0]
    assert root_sec_node["value"] == "sec:doc_hash_1::0"
    assert "Overview" in root_sec_node["label"]

    assert len(root_sec_node["children"]) == 1
    child_sec_node = root_sec_node["children"][0]
    assert child_sec_node["value"] == "sec:doc_hash_1::1"
    assert "Details" in child_sec_node["label"]
    assert "📊" in child_sec_node["label"]
    assert "🖼️" in child_sec_node["label"]


def test_extract_markdown_images() -> None:
    text = """
    Here is an image: ![Diagram](https://example.com/arch.png "System Arch")
    And a mistral comment: <!-- Image: chart.png (hash: 1234abcd) -->
    And HTML: <img src="logo.png" />
    """
    imgs = extract_markdown_images(text)
    assert len(imgs) == 3
    assert imgs[0]["url"] == "https://example.com/arch.png"
    assert imgs[1]["url"] == "chart.png"
    assert imgs[2]["url"] == "logo.png"


def test_resolve_image_path(tmp_path: Path) -> None:
    # URL
    assert resolve_image_path("https://example.com/test.png", None) == "https://example.com/test.png"

    # Local file relative to doc
    doc_file = tmp_path / "doc.md"
    doc_file.write_text("# Doc", encoding="utf-8")
    img_file = tmp_path / "sample.png"
    img_file.write_bytes(b"data")

    resolved = resolve_image_path("sample.png", "sample.png", doc_path=str(doc_file))
    assert resolved == str(img_file.resolve())


# ---------------------------------------------------------------------------
# Async Integration Tests against real Ladybug DB
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_async_fetch_docgraph_data(sample_docgraph_db: tuple[str, Path]) -> None:
    db_path, _ = sample_docgraph_db
    backend = KuzuBackend()
    backend.connect(db_path)

    stats = await fetch_database_stats(backend)
    assert stats["documents"] == 1
    assert stats["sections"] >= 3
    assert stats["total_tokens"] > 0

    folders, docs = await fetch_folders_and_documents(backend)
    assert len(docs) == 1
    doc_hash = docs[0]["markdown_hash"]

    sections = await fetch_document_sections_full(backend, doc_hash)
    assert len(sections) >= 3

    # Check table / image detection in sections
    has_table_found = any(s["has_table"] for s in sections)
    has_image_found = any(s["has_image"] for s in sections)
    assert has_table_found
    assert has_image_found


# ---------------------------------------------------------------------------
# CLI Command Tests
# ---------------------------------------------------------------------------


def test_docbench_cli_help() -> None:
    app = typer.Typer()
    DocBenchCommands().register(app)
    runner = CliRunner()

    res = runner.invoke(app, ["docbench", "--help"])
    assert res.exit_code == 0
    assert "web" in res.output
    assert "cli" in res.output
    assert "tui" in res.output
    assert "list" in res.output


def test_docbench_cli_list(sample_docgraph_db: tuple[str, Path]) -> None:
    db_path, _ = sample_docgraph_db
    app = typer.Typer()
    DocBenchCommands().register(app)
    runner = CliRunner()

    res = runner.invoke(app, ["docbench", "list", "--db", db_path])
    assert res.exit_code == 0
    assert "annual_report.md" in res.output


def test_docgraph_web_help() -> None:
    app = typer.Typer()
    DocGraphCommands().register(app)
    runner = CliRunner()

    res = runner.invoke(app, ["docgraph", "web", "--help"])
    assert res.exit_code == 0
    assert "--port" in res.output
    assert "--host" in res.output
    assert "--profile" in res.output


def test_get_configured_profiles() -> None:
    profiles = _get_configured_profiles()
    assert isinstance(profiles, list)
    assert len(profiles) >= 1
    assert "default" in profiles


def test_build_tree_select_nodes_unwraps_synthetic_root() -> None:
    docs = [
        {
            "markdown_hash": "h1",
            "filename": "guide.md",
            "section_count": 2,
            "token_count": 100,
        }
    ]
    sections_by_doc = {
        "h1": [
            {
                "section_id": "h1::0",
                "title": "(document root)",
                "level": 0,
                "parent_section_id": None,
                "sequence": 0,
                "token_count": 100,
            },
            {
                "section_id": "h1::1",
                "title": "Getting Started",
                "level": 1,
                "parent_section_id": "h1::0",
                "sequence": 1,
                "token_count": 60,
            },
            {
                "section_id": "h1::2",
                "title": "Installation",
                "level": 2,
                "parent_section_id": "h1::1",
                "sequence": 2,
                "token_count": 40,
            },
        ]
    }
    nodes = build_tree_select_nodes(folders=[], documents=docs, sections_by_doc=sections_by_doc, include_sections=True)
    assert len(nodes) == 1
    doc_node = nodes[0]
    assert len(doc_node["children"]) == 1
    # Real heading "Getting Started" is direct child of document, not "(document root)"
    sec1 = doc_node["children"][0]
    assert sec1["value"] == "sec:h1::1"
    assert "Getting Started" in sec1["label"]
    assert len(sec1["children"]) == 1
    assert sec1["children"][0]["value"] == "sec:h1::2"


@pytest.mark.anyio
async def test_async_fetch_section_markdown(sample_docgraph_db: tuple[str, Path]) -> None:
    db_path, _ = sample_docgraph_db
    backend = KuzuBackend()
    backend.connect(db_path)

    _, docs = await fetch_folders_and_documents(backend)
    doc_hash = docs[0]["markdown_hash"]
    sections = await fetch_document_sections_full(backend, doc_hash)
    sec_id = sections[1]["section_id"]

    md_text = await fetch_section_markdown_async(backend, sec_id)
    assert md_text is not None
    assert len(md_text) > 0
