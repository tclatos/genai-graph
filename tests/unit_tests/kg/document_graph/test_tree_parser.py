"""Unit tests for the Document Graph Markdown parser."""

from __future__ import annotations

import pytest

from genai_graph.kg.document_graph.tree_parser import (
    _infer_levels,
    _outline_depth,
    _strip_surrounding_emphasis,
    parse_markdown_tree,
)


@pytest.mark.unit
class TestParseMarkdownTree:
    def test_empty_document_has_single_root_section(self) -> None:
        sections = parse_markdown_tree("")
        assert len(sections) == 1
        assert sections[0].level == 0

    def test_no_headings_document_has_single_root_section(self) -> None:
        raw = "Just a paragraph, no headings."
        sections = parse_markdown_tree(raw)
        assert len(sections) == 1
        assert sections[0].level == 0
        assert sections[0].text == raw

    def test_document_is_reconstructable_from_sections(self) -> None:
        raw = "# Title\n\nIntro text.\n\n## Section A\n\nBody A.\n\n## Section B\n\nBody B.\n"
        sections = parse_markdown_tree(raw)
        assert "\n".join(s.text for s in sections) == raw.rstrip("\n")

    def test_flat_headings(self) -> None:
        raw = "# Title\n\nIntro text.\n\n## Section A\n\nBody A.\n\n## Section B\n\nBody B.\n"
        sections = parse_markdown_tree(raw)

        assert [s.title for s in sections] == ["Title", "Section A", "Section B"]
        assert [s.level for s in sections] == [1, 2, 2]
        # Section A and B are siblings under Title
        assert sections[0].parent_index is None
        assert sections[1].parent_index == 0
        assert sections[2].parent_index == 0

    def test_nested_headings_parent_index(self) -> None:
        raw = "# H1\n## H2a\n### H3\ntext\n## H2b\n"
        sections = parse_markdown_tree(raw)

        assert [s.title for s in sections] == ["H1", "H2a", "H3", "H2b"]
        assert sections[0].parent_index is None  # H1 root
        assert sections[1].parent_index == 0  # H2a -> H1
        assert sections[2].parent_index == 1  # H3 -> H2a
        assert sections[3].parent_index == 0  # H2b -> H1 (pops H3 and H2a off the stack)

    def test_line_start_and_line_end(self) -> None:
        raw = "\n".join(
            [
                "# Title",  # line 1
                "intro",  # line 2
                "## Section A",  # line 3
                "body a line 1",  # line 4
                "body a line 2",  # line 5
                "## Section B",  # line 6
                "body b",  # line 7
            ]
        )
        sections = parse_markdown_tree(raw)
        title, section_a, section_b = sections

        assert title.line_start == 1
        # "Title" is H1; its *own* text spans only until the next heading of
        # any level (nested subsections own their own text ranges).
        assert title.line_end == 2
        assert section_a.line_start == 3
        assert section_a.line_end == 5  # ends right before "## Section B"
        assert section_b.line_start == 6
        assert section_b.line_end == 7  # end of file

    def test_code_fence_does_not_produce_false_headings(self) -> None:
        raw = "# Real Heading\n\n```markdown\n# Not a heading\n## Also not one\n```\n\n## Real Section\n"
        sections = parse_markdown_tree(raw)

        assert [s.title for s in sections] == ["Real Heading", "Real Section"]

    def test_heading_inside_blockquote_is_ignored(self) -> None:
        raw = "# Top\n\n> # Quoted heading\n\n## Bottom\n"
        sections = parse_markdown_tree(raw)

        assert [s.title for s in sections] == ["Top", "Bottom"]

    def test_token_count_is_positive(self) -> None:
        raw = "# Title\n\nSome words here for counting tokens.\n"
        sections = parse_markdown_tree(raw)
        assert sections[0].token_count > 0

    def test_untitled_heading_gets_placeholder_title(self) -> None:
        raw = "#\n\nbody\n"
        sections = parse_markdown_tree(raw)
        assert sections[0].title == "(untitled H1)"


@pytest.mark.unit
class TestInferLevels:
    def test_well_structured_markdown_keeps_its_levels(self) -> None:
        # A real H1/H2/H3 hierarchy is not degenerate, so the Markdown levels are
        # authoritative and must be returned unchanged (the AMD 10-K case).
        titles = ["PART I", "Our Industry", "Data Center Market"] * 5
        md_levels = [1, 2, 3] * 5
        assert _infer_levels(titles, md_levels) == md_levels

    def test_degenerate_flat_with_numbering_re_derives(self) -> None:
        # All-H1 (degenerate) but coherently numbered -> levels come from the outline
        # numbers (1 -> 1, 1.1 -> 2), unnumbered nest one below the last numbered.
        titles = ["1. Intro", "Body", "1.1 Detail", "2. Next"]
        assert _infer_levels(titles, [1, 1, 1, 1]) == [1, 2, 2, 1]

    def test_degenerate_flat_without_numbering_stays_flat(self) -> None:
        titles = ["Alpha", "Beta", "Gamma", "Delta"]
        assert _infer_levels(titles, [1, 1, 1, 1]) == [1, 1, 1, 1]

    def test_interest_rate_is_not_an_outline_number(self) -> None:
        assert _outline_depth("3.924% Senior Notes") is None
        assert _outline_depth("1. Financial Statements") == 1
        assert _outline_depth("3.4 Device life cycle") == 2
        assert _outline_depth("10 Foo") == 1


@pytest.mark.unit
class TestStripSurroundingEmphasis:
    def test_balanced_wrap_is_stripped(self) -> None:
        assert (
            _strip_surrounding_emphasis("***Original Equipment Manufacturers***") == "Original Equipment Manufacturers"
        )
        assert _strip_surrounding_emphasis("**Advanced Micro Devices, Inc.**") == "Advanced Micro Devices, Inc."
        assert _strip_surrounding_emphasis("*Non-custom products*") == "Non-custom products"

    def test_dangling_unbalanced_is_stripped(self) -> None:
        assert (
            _strip_surrounding_emphasis("**Certification of Chief Executive Officer")
            == "Certification of Chief Executive Officer"
        )

    def test_backticks_and_internal_math_are_preserved(self) -> None:
        assert _strip_surrounding_emphasis("See `footnote`") == "See `footnote`"
        assert _strip_surrounding_emphasis("Note: 2 * 3") == "Note: 2 * 3"
