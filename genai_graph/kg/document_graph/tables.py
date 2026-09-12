"""Table extraction and parsing for Document Graph sections."""

from __future__ import annotations

import re
from pathlib import Path

from genai_graph.kg.document_graph.tree_parser import _estimate_token_count
from genai_graph.kg.nodes.document_section import MarkdownSection, Table

# Matches HTML tables: <table ...>...</table>
_HTML_TABLE_PATTERN = re.compile(r"<table(?:\s+[^>]*)?>(.*?)</table>", re.DOTALL | re.IGNORECASE)

# Matches HTML caption: <caption>...</caption> or <figcaption>...</caption>
_HTML_CAPTION_PATTERN = re.compile(
    r"<caption>(.*?)</caption>|<figcaption>(.*?)</figcaption>|<p[^>]*><em>(.*?)</em></p>",
    re.IGNORECASE,
)

# Matches Markdown pipe table lines
_MD_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_MD_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")

# Caption pattern (e.g. "**Table 1: Title**", "*Table 2 - ...*", "Table 1. ...")
_TABLE_CAPTION_PREFIX_PATTERN = re.compile(
    r"^(?:[\*_]{1,2})?\s*(?:Table|Tab\.)\s*[\d\w\.\-:]*\s*[:\-\.]?\s*",
    re.IGNORECASE,
)

_ITALIC_BOLD_WRAPPER = re.compile(r"^[\*_]{1,2}(.*?)[\*_]{1,2}$")


def _clean_caption_text(text: str) -> str:
    """Clean markdown styling and HTML tags from a caption string."""
    text = text.strip()
    html_match = _HTML_CAPTION_PATTERN.search(text)
    if html_match:
        text = html_match.group(1) or html_match.group(2) or html_match.group(3) or text

    italic_match = _ITALIC_BOLD_WRAPPER.match(text)
    if italic_match:
        text = italic_match.group(1)

    text = re.sub(r"[\*_`]", "", text)
    return text.strip()


def _find_caption_before_pos(text: str, pos: int) -> str | None:
    """Scan lines preceding a table to extract an immediate table caption/title."""
    sub = text[:pos]
    lines = sub.splitlines()
    for line in reversed(lines[-4:]):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith("!["):
            break
        if _TABLE_CAPTION_PREFIX_PATTERN.match(stripped):
            cleaned = _clean_caption_text(stripped)
            if len(cleaned) >= 3:
                return cleaned
    return None


def extract_section_tables(
    section: MarkdownSection,
    markdown_file_path: Path | None = None,  # noqa: ARG001
) -> list[Table]:
    """Extract all tables (HTML and Markdown) referenced in a MarkdownSection.

    Args:
        section: The MarkdownSection to parse.
        markdown_file_path: Optional path to the containing markdown file.

    Returns:
        List of Table nodes found within the section.
    """
    text = section.text or ""
    if not text or ("<table" not in text.lower() and "|" not in text):
        return []

    tables: list[Table] = []
    table_index = 0

    # 1. Extract HTML tables
    for match in _HTML_TABLE_PATTERN.finditer(text):
        full_table = match.group(0)
        inner_content = match.group(1)
        match_start = match.start()

        # Check caption inside HTML table
        caption = None
        cap_match = _HTML_CAPTION_PATTERN.search(inner_content)
        if cap_match:
            raw_cap = cap_match.group(1) or cap_match.group(2) or cap_match.group(3)
            if raw_cap:
                caption = _clean_caption_text(raw_cap)

        if not caption:
            caption = _find_caption_before_pos(text, match_start)

        name = caption or f"Table {table_index + 1}"
        tok_count = _estimate_token_count(full_table)

        table_obj = Table(
            table_id=f"{section.section_id}::t{table_index}",
            section_id=section.section_id,
            markdown_hash=section.markdown_hash,
            table_index=table_index,
            name=name,
            table_format="html",
            content=full_table,
            caption=caption,
            token_count=tok_count,
        )
        tables.append(table_obj)
        table_index += 1

    # 2. Extract Markdown pipe tables (only if no HTML tables or outside HTML table spans)
    lines = text.splitlines()
    current_table_lines: list[str] = []
    current_table_start_line = 0
    has_separator = False

    def _flush_md_table(t_lines: list[str], start_line_idx: int) -> None:
        nonlocal table_index
        if len(t_lines) < 2:
            return
        table_str = "\n".join(t_lines)
        # Check caption in preceding lines
        preceding_lines = lines[max(0, start_line_idx - 3) : start_line_idx]
        caption = None
        for pline in reversed(preceding_lines):
            pstr = pline.strip()
            if not pstr:
                continue
            if _TABLE_CAPTION_PREFIX_PATTERN.match(pstr):
                caption = _clean_caption_text(pstr)
                break

        name = caption or f"Table {table_index + 1}"
        tok_count = _estimate_token_count(table_str)

        table_obj = Table(
            table_id=f"{section.section_id}::t{table_index}",
            section_id=section.section_id,
            markdown_hash=section.markdown_hash,
            table_index=table_index,
            name=name,
            table_format="markdown",
            content=table_str,
            caption=caption,
            token_count=tok_count,
        )
        tables.append(table_obj)
        table_index += 1

    in_html = False
    for idx, line in enumerate(lines):
        s = line.strip()
        if "<table" in s.lower():
            in_html = True
        if "</table>" in s.lower():
            in_html = False
            continue
        if in_html:
            continue

        if _MD_TABLE_ROW_RE.match(s):
            if not current_table_lines:
                current_table_start_line = idx
            current_table_lines.append(line)
            if _MD_TABLE_SEP_RE.match(s):
                has_separator = True
        else:
            if current_table_lines and has_separator:
                _flush_md_table(current_table_lines, current_table_start_line)
            current_table_lines = []
            has_separator = False

    if current_table_lines and has_separator:
        _flush_md_table(current_table_lines, current_table_start_line)

    return tables
