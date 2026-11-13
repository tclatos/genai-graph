"""Unit tests for the Document Graph and PDF inspector component."""

from __future__ import annotations

from pathlib import Path

from genai_graph.ui.streamlit.components.docgraph_inspector import find_document_pdf


def test_find_document_pdf_exact_stem(tmp_path: Path) -> None:
    """find_document_pdf resolves a PDF matching <stem>.pdf."""
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    sample_pdf = pdf_dir / "manual_a.pdf"
    sample_pdf.write_bytes(b"%PDF-1.4...")

    resolved = find_document_pdf(
        doc_filename="manual_a_pdf.md",
        additional_dirs=[pdf_dir],
    )
    assert resolved is not None
    assert resolved.resolve() == sample_pdf.resolve()


def test_find_document_pdf_origin_comment(tmp_path: Path) -> None:
    """find_document_pdf reads <!-- source: /path/to/source.pdf --> when present."""
    pdf_file = tmp_path / "original_source.pdf"
    pdf_file.write_bytes(b"%PDF-1.4...")

    md_file = tmp_path / "converted.md"
    md_file.write_text(f"<!-- source: {pdf_file.resolve()} -->\n# Content", encoding="utf-8")

    resolved = find_document_pdf(
        doc_filename="converted.md",
        doc_path=str(md_file.resolve()),
    )
    assert resolved is not None
    assert resolved.resolve() == pdf_file.resolve()


def test_find_document_pdf_missing(tmp_path: Path) -> None:
    """Returns None when no matching PDF is found."""
    resolved = find_document_pdf(
        doc_filename="non_existent_doc.md",
        additional_dirs=[tmp_path],
    )
    assert resolved is None
