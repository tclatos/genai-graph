"""Unit tests for batch markdownize and PDF discovery in benchmark builder."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from genai_graph.bench.build_graph import (
    MD_FILENAME_SUFFIX,
    find_pdf_path,
    markdownize_targets_batch,
)


def test_find_pdf_path_resolves_directly_and_in_subdirs(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()

    # Case 1: Direct PDF
    pdf1 = pdf_dir / "doc1.pdf"
    pdf1.write_bytes(b"%PDF-1.4 " + b"x" * 2000)
    assert find_pdf_path("doc1", pdf_dir) == pdf1
    assert find_pdf_path("doc1.pdf", pdf_dir) == pdf1

    # Case 2: In documents/ subfolder
    sub_docs = pdf_dir / "documents"
    sub_docs.mkdir()
    pdf2 = sub_docs / "doc2.pdf"
    pdf2.write_bytes(b"%PDF-1.4 " + b"x" * 2000)
    assert find_pdf_path("doc2", pdf_dir) == pdf2

    # Case 3: Missing raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        find_pdf_path("nonexistent", pdf_dir)


def test_markdownize_targets_batch_uses_batch_converter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    saved_md_dir = tmp_path / "saved_markdown"
    saved_md_dir.mkdir()

    pdf1 = pdf_dir / "doc1.pdf"
    pdf2 = pdf_dir / "doc2.pdf"
    pdf1.write_bytes(b"%PDF-1.4 " + b"x" * 2000)
    pdf2.write_bytes(b"%PDF-1.4 " + b"x" * 2000)

    mock_conv = MagicMock()
    mock_conv.batch_convert.return_value = {
        str(pdf1): "## Page 1\n\nDoc 1 content",
        str(pdf2): "## Page 1\n\nDoc 2 content",
    }

    from genai_tk.extra.markdownize.factory import ConverterFactory

    monkeypatch.setattr(ConverterFactory, "create", lambda _: mock_conv)

    results = markdownize_targets_batch(
        ["doc1", "doc2"],
        pdfs_dir=pdf_dir,
        saved_markdown_dir=saved_md_dir,
        markdownize_profile="medium",
    )

    assert len(results) == 2
    assert (saved_md_dir / f"doc1{MD_FILENAME_SUFFIX}").read_text(encoding="utf-8") == "## Page 1\n\nDoc 1 content"
    assert (saved_md_dir / f"doc2{MD_FILENAME_SUFFIX}").read_text(encoding="utf-8") == "## Page 1\n\nDoc 2 content"
    mock_conv.batch_convert.assert_called_once()


def test_markdownize_targets_batch_skips_cached_and_converts_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    saved_md_dir = tmp_path / "saved_markdown"
    saved_md_dir.mkdir()

    # doc1 is already cached
    cached_doc1 = saved_md_dir / f"doc1{MD_FILENAME_SUFFIX}"
    cached_doc1.write_text("Cached doc1 content", encoding="utf-8")

    # doc2 is missing
    pdf2 = pdf_dir / "doc2.pdf"
    pdf2.write_bytes(b"%PDF-1.4 " + b"x" * 2000)

    mock_conv = MagicMock()
    mock_conv.convert.return_value = "Doc 2 converted content"

    from genai_tk.extra.markdownize.factory import ConverterFactory

    monkeypatch.setattr(ConverterFactory, "create", lambda _: mock_conv)

    results = markdownize_targets_batch(
        ["doc1", "doc2"],
        pdfs_dir=pdf_dir,
        saved_markdown_dir=saved_md_dir,
        markdownize_profile="medium",
    )

    assert len(results) == 2
    assert cached_doc1.read_text(encoding="utf-8") == "Cached doc1 content"
    assert (saved_md_dir / f"doc2{MD_FILENAME_SUFFIX}").read_text(encoding="utf-8") == "Doc 2 converted content"
