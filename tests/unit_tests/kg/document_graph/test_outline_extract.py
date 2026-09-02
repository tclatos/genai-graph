"""Unit tests for outline extraction (LLM boundary monkeypatched, no database)."""

from __future__ import annotations

from pathlib import Path

import pytest

from genai_graph.kg.document_graph.outline_extract import (
    DocumentOutline,
    OutlineConfig,
    OutlineEntry,
    _cache_path,
    _resolve_llm_id,
    extract_outline,
)


def _outline() -> DocumentOutline:
    """A minimal content-free outline returned by the fake LLM."""
    return DocumentOutline(
        document_description="A doc.",
        document_summary="A doc summary.",
        sections=[
            OutlineEntry(title="Title", level=1, description="Company overview and fiscal-year basis."),
            OutlineEntry(
                title="Section A", level=2, description="Product lines and target markets.", summary="A summary."
            ),
        ],
    )


def _config(tmp_path: Path) -> OutlineConfig:
    return OutlineConfig(llm="fake@fake", cache_root=str(tmp_path))


@pytest.mark.unit
class TestExtractOutline:
    def test_success_extracts_and_caches(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("genai_graph.kg.document_graph.outline_extract._context_window_for", lambda llm_id: None)
        monkeypatch.setattr("genai_graph.kg.document_graph.outline_extract._call_llm", lambda **kwargs: _outline())
        config = _config(tmp_path)
        warnings: list[str] = []

        # Heading-bearing markdown: the LLM outline (Title, Section A) is aligned
        # back onto the detected headings (verbatim title + algorithmic level).
        md_text = "# Title\n\n## Section A\n\nbody\n"
        result = extract_outline(md_text, "deadbeef", "doc.md", config, warnings=warnings)

        assert result.outline is not None
        assert result.degraded is False
        assert result.llm_calls == 1
        assert len(result.outline.sections) == 2
        assert result.outline.sections[0].title == "Title"
        assert result.outline.sections[0].level == 1  # algorithmic level is authoritative
        assert result.outline.sections[0].description == "Company overview and fiscal-year basis."
        assert list(tmp_path.rglob("*.json"))  # cache file written

    def test_prompt_uses_template_variables_not_baked_text(self, tmp_path: Path) -> None:
        # The user message must reference {raw}/{headings}/{filename} as template
        # variables (filled at invoke time), not bake the source text into the
        # template string — otherwise Markdown braces like LaTeX `^{(1)}` would be
        # parsed as prompt-template variables and crash rendering.
        from genai_graph.kg.document_graph.outline_extract import _build_prompt

        system, user = _build_prompt(filename="doc.md", raw="SHOULD_NOT_APPEAR", config=_config(tmp_path))

        assert "{filename}" in user
        assert "{raw}" in user
        assert "{headings}" in user
        assert "SHOULD_NOT_APPEAR" not in user

    def test_cache_hit_avoids_llm_call(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[dict] = []

        def fake_call_llm(**kwargs) -> DocumentOutline:
            calls.append(kwargs)
            return _outline()

        monkeypatch.setattr("genai_graph.kg.document_graph.outline_extract._context_window_for", lambda llm_id: None)
        monkeypatch.setattr("genai_graph.kg.document_graph.outline_extract._call_llm", fake_call_llm)
        config = _config(tmp_path)
        warnings: list[str] = []

        first = extract_outline("body", "deadbeef", "doc.md", config, warnings=warnings)
        second = extract_outline("body", "deadbeef", "doc.md", config, warnings=warnings)

        assert first.llm_calls == 1
        assert second.llm_calls == 0  # cache hit resets the per-invocation count
        assert len(calls) == 1  # the LLM was called only once

    def test_over_context_window_degrades_without_llm_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called: list[dict] = []

        def fake_call_llm(**kwargs) -> DocumentOutline:
            called.append(kwargs)
            return _outline()

        monkeypatch.setattr("genai_graph.kg.document_graph.outline_extract.count_tokens", lambda text: 100_000)
        monkeypatch.setattr("genai_graph.kg.document_graph.outline_extract._context_window_for", lambda llm_id: 1000)
        monkeypatch.setattr("genai_graph.kg.document_graph.outline_extract._call_llm", fake_call_llm)
        config = _config(tmp_path)
        config.context_safety_ratio = 0.9
        warnings: list[str] = []

        result = extract_outline("body", "deadbeef", "doc.md", config, warnings=warnings)

        assert result.degraded is True
        assert result.outline is None
        assert result.reason == "context_window_overflow"
        assert warnings  # surfaced a human-readable degrade reason
        assert called == []  # no LLM call made

    def test_llm_failure_degrades(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_call_llm(**kwargs) -> DocumentOutline:
            raise ValueError("boom")

        monkeypatch.setattr("genai_graph.kg.document_graph.outline_extract._context_window_for", lambda llm_id: None)
        monkeypatch.setattr("genai_graph.kg.document_graph.outline_extract._call_llm", fake_call_llm)
        config = _config(tmp_path)
        warnings: list[str] = []

        result = extract_outline("body", "deadbeef", "doc.md", config, warnings=warnings)

        assert result.degraded is True
        assert result.reason == "llm_call_failed"
        assert any("LLM call failed" in w for w in warnings)

    def test_corrupt_cache_is_re_extracted(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("genai_graph.kg.document_graph.outline_extract._context_window_for", lambda llm_id: None)
        monkeypatch.setattr("genai_graph.kg.document_graph.outline_extract._call_llm", lambda **kwargs: _outline())
        config = _config(tmp_path)
        warnings: list[str] = []

        # Poison the cache with garbage so _load_cached returns None.
        cache_path = _cache_path(config, _resolve_llm_id(config), "deadbeef")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("{ not valid json", encoding="utf-8")

        result = extract_outline("body", "deadbeef", "doc.md", config, warnings=warnings)

        assert result.outline is not None
        assert result.llm_calls == 1  # re-extracted despite a present-but-corrupt cache


@pytest.mark.unit
class TestRestatementFilter:
    def test_restatement_descriptions_are_dropped(self) -> None:
        from genai_graph.kg.document_graph.outline_extract import _is_title_restatement

        assert _is_title_restatement("PART I", "Begins Part I of the annual report.")
        assert _is_title_restatement("Gaming Segment", "Introduces the Gaming segment.")
        assert _is_title_restatement("Our Strategy", "Outlines our strategy.")
        assert not _is_title_restatement("Data Center Products", "Lists server CPUs, GPUs and DPUs.")
        assert not _is_title_restatement(
            "Non-custom products", "Off-the-shelf CPUs/GPUs recognized on delivery (ASC 606)."
        )
        assert not _is_title_restatement("Title", None)

    def test_clean_outline_drops_restatements_keeps_substantive(self, tmp_path: Path) -> None:
        from genai_graph.kg.document_graph.outline_extract import _clean_outline

        outline = DocumentOutline(
            document_description="A doc.",
            document_summary="A doc summary.",
            sections=[
                OutlineEntry(title="PART I", level=1, description="Begins Part I of the annual report."),
                OutlineEntry(title="Data Center Products", level=2, description="Server CPUs, GPUs and DPUs."),
                OutlineEntry(title="Index", level=1, description=None),
            ],
        )
        cleaned = _clean_outline(outline, _config(tmp_path))
        descs = [s.description for s in cleaned.sections]
        assert descs[0] is None  # restatement dropped
        assert descs[1] == "Server CPUs, GPUs and DPUs."  # substantive kept
        assert descs[2] is None  # null stays null
