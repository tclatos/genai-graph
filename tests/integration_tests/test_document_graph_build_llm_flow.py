"""Flow-level integration test for the `document_graph_flow` LLM build path.

Drives the Prefect flow in-process (no CLI); the LLM call boundary
(``outline_extract._call_llm``) and the context-window / token-count helpers are
monkeypatched — the only true external dependencies — so the tests exercise the
full extract -> merge -> ingest -> DB path, including the over-context-window
degradation to algorithmic parsing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from genai_graph.kg.backend import KuzuBackend
from genai_graph.kg.document_graph.outline_extract import DocumentOutline, OutlineConfig, OutlineEntry
from genai_graph.kg.query.document_graph_tools import get_document, get_document_toc
from genai_graph.orchestration.document_graph_flow import document_graph_flow

# The flow has no `@task` calls, so invoking `document_graph_flow.fn(...)` runs the
# exact same extract -> merge -> ingest wiring as the decorated entrypoint, without
# requiring a Prefect API server (which is an ephemeral subprocess and flaky in a
# test sandbox). The returned dict and DB effects are identical.
_build_flow = document_graph_flow.fn

DOC = """# Guide

## Setup

{filler}

## Usage

Run the CLI.
"""


def _fake_outline(filename: str) -> DocumentOutline:
    """A content-free outline whose titles match the Markdown headings verbatim."""
    return DocumentOutline(
        document_description=f"Description of {filename}.",
        document_summary=f"Summary of {filename}.",
        sections=[
            OutlineEntry(title="Guide", level=1, description="The whole guide."),
            OutlineEntry(
                title="Setup", level=2, description="How to install.", summary="Install via pip then configure."
            ),
            OutlineEntry(title="Usage", level=2, description="How to run.", summary=None),
        ],
    )


def _fake_call_llm(
    *, llm_id: str, filename: str, raw: str, config: OutlineConfig, max_tokens: int | None
) -> DocumentOutline:
    return _fake_outline(filename)


@pytest.mark.integration
class TestDocumentGraphBuildLLMFlow:
    def test_llm_path_writes_sections_and_summaries(
        self, temp_db_path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "guide.md").write_text(DOC.format(filler="Install with pip. " * 60), encoding="utf-8")

        monkeypatch.setattr("genai_graph.kg.document_graph.outline_extract._context_window_for", lambda llm_id: None)
        monkeypatch.setattr("genai_graph.kg.document_graph.outline_extract._call_llm", _fake_call_llm)

        result = _build_flow(
            sources=[str(tmp_path)],
            db_path=temp_db_path,
            llm="fake@fake",
            outline_cache_dir=str(tmp_path / "outline_cache"),
        )

        assert result["documents_processed"] == 1
        assert result["documents_failed"] == 0
        assert result["files_degraded"] == 0
        assert result["sections_summarized"] >= 1

        backend = KuzuBackend()
        backend.connect(temp_db_path)
        try:
            rows = get_document_toc(backend, "guide.md")
            described = [r for r in rows if int(r["level"]) > 0]
            assert described
            assert all(r["description"] for r in described)
            assert all(r["summary_source"] == "llm" for r in described)
            assert any(r["summary"] for r in described)

            doc = get_document(backend, "guide.md")
            assert doc is not None
            assert doc["description"] == "Description of guide.md."
            assert doc["summary"] == "Summary of guide.md."
        finally:
            backend.close()

    def test_over_context_window_degrades_to_algo(
        self, temp_db_path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "guide.md").write_text(DOC.format(filler="Install with pip. " * 60), encoding="utf-8")

        calls: list[dict] = []

        def spying_call_llm(**kwargs) -> DocumentOutline:
            calls.append(kwargs)
            return _fake_outline(kwargs["filename"])

        monkeypatch.setattr("genai_graph.kg.document_graph.outline_extract.count_tokens", lambda text: 100_000)
        monkeypatch.setattr("genai_graph.kg.document_graph.outline_extract._context_window_for", lambda llm_id: 1000)
        monkeypatch.setattr("genai_graph.kg.document_graph.outline_extract._call_llm", spying_call_llm)

        result = _build_flow(
            sources=[str(tmp_path)],
            db_path=temp_db_path,
            llm="fake@fake",
            outline_cache_dir=str(tmp_path / "outline_cache"),
        )

        # The document is still ingested (algorithmic structure), just without summaries.
        assert result["documents_processed"] == 1
        assert result["files_degraded"] == 1
        assert result["sections_summarized"] == 0
        assert any("degrad" in w.lower() for w in result["warnings"])
        assert calls == []  # no LLM call was made

        backend = KuzuBackend()
        backend.connect(temp_db_path)
        try:
            rows = get_document_toc(backend, "guide.md")
            sections = [r for r in rows if int(r["level"]) > 0]
            assert sections  # algorithmic structure still produced sections
            assert all(r["description"] is None for r in sections)
            assert all(r["summary"] is None for r in sections)
            assert all(r["summary_source"] is None for r in sections)
        finally:
            backend.close()
