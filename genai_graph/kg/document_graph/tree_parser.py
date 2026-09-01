"""Document Graph — parse Markdown text into a flat, ordered list of sections.

Uses ``markdown-it-py`` (a real CommonMark parser) instead of regex so that
headings inside fenced code blocks, inline code, or blockquotes are handled
correctly, and each heading's source line number is known precisely.

Each section owns a *non-overlapping* slice of the document: its lines run from
its heading up to the line before the next heading of **any** level (a nested
subsection's lines therefore belong to the subsection, not the parent). Any text
before the first heading — or a document with no headings at all — is captured by
a synthetic level-0 root section. As a result every document yields at least one
section and can be reconstructed exactly by concatenating the sections'
``text`` in ``sequence`` order.
"""

from __future__ import annotations

import re
from collections import Counter

from pydantic import BaseModel, Field

# Title used for the synthetic level-0 section that captures a document's
# preamble (text before the first heading) or a heading-less document.
ROOT_SECTION_TITLE = "(document root)"

# Words that are page markers from a PDF/Doc → Markdown conversion (e.g.
# "## Page 12"). These carry no structural meaning and must not become sections.
_PAGE_MARKER_RE = re.compile(r"^page\s+\d+$", re.IGNORECASE)

# Multilingual Table of Contents (TOC) header pattern:
# Matches common TOC titles in English, French, German, Spanish, Italian, etc.
_TOC_HEADER_RE = re.compile(
    r"^\s*(?:#*\s*)?(?:"
    r"table\s+of\s+contents|"
    r"table\s+des\s+mati[èe]res|"
    r"sommaire|"
    r"inhaltsverzeichnis|"
    r"table\s+du\s+contenu|"
    r"list\s+of\s+contents|"
    r"contents|"
    r"índice|"
    r"indice"
    r")\s*$",
    re.IGNORECASE,
)

# Generic structured document headings pattern (language/domain-agnostic):
# Matches:
# - PART / Part / SECTION / Section / CHAPTER / Chapter / ARTICLE / Article / ITEM / Item + numbers
# - NOTE / Note / EXHIBIT / Exhibit / ANNEX / Annex / ANNEXE / Annexe / APPENDIX / Appendix + numbers
# - Common corporate financial statement titles (Statements of Income, Balance Sheets, Bilans, etc.)
# - SIGNATURE / SIGNATURES
_HEURISTIC_HEADING_RE = re.compile(
    r"^(?:"
    r"(?:(?:PART|Part|SECTION|Section|CHAPTER|Chapter|ARTICLE|Article|ITEM|Item)\s+[IVX\d]+(?:\.\d+)?[A-Za-z]?\.?)"
    r"|(?:(?:NOTE|Note|EXHIBIT|Exhibit|ANNEX|Annex|ANNEXE|Annexe|APPENDIX|Appendix)\s+[A-Za-z\d]+(?:\.\d+)?\.?)"
    r"|(?:\b(?:CONDENSED\s+)?(?:CONSOLIDATED\s+)?(?:STATEMENTS?\s+OF|BALANCE\s+SHEETS?|BILAN|COMPTE\s+DE\s+R[ÉE]SULTAT|TABLEAU\s+DE\s+FLUX)\b)"
    r"|(?:SIGNATURES?)"
    r")",
    re.IGNORECASE,
)

# Common non-heading uppercase tokens to ignore during standalone uppercase title matching
_NON_HEADING_UPPERCASE_TOKENS = frozenset(
    {
        "PAGE",
        "N/A",
        "NONE",
        "TOTAL",
        "ALL RIGHTS RESERVED",
        "CONFIDENTIAL",
        "UNAUDITED",
    }
)

# Surrounding Markdown emphasis/whitespace, stripped before comparing a heading
# title to another for de-duplication (e.g. "**Advanced Micro Devices, Inc.**"
# and "Advanced Micro Devices, Inc." collapse together).
_DEDUP_EMPHASIS_RE = re.compile(r"[*_`]{1,3}")
_DEDUP_WS_RE = re.compile(r"\s+")

# Leading outline number of a heading, ignoring surrounding Markdown emphasis
# (e.g. "**3.4 Device life cycle**" -> "3.4"). The depth (dot-separated component
# count) gives the heading's logical level in a numbered document. The number
# must be followed by whitespace, an optional trailing ".", or end-of-string, so
# values prefixed with "%" (interest rates like "3.924% Senior Notes") are NOT
# mistaken for outline numbers.
_OUTLINE_NUMBER_RE = re.compile(r"^\**\s*(\d+(?:\.\d+)*)(?:\.)?(?=\s|$)")

# Surrounding Markdown emphasis that wraps a whole heading title
# (e.g. "***Original Equipment Manufacturers***" or "**Advanced Micro Devices,
# Inc.**"). Stripped from the stored title so the table of contents is clean; the
# section ``text`` still carries the raw line, so the document stays
# byte-for-byte reconstructable.
_SURROUNDING_EMPHASIS_RE = re.compile(r"^([*_`]{1,3})(.*?)\1$")
# Unbalanced/dangling emphasis a converter left at an edge with no closer
# (e.g. "**Certification of Chief Executive Officer"). Only ``*``/``_`` (not
# backticks, which are often inline code) and only when glued to text, so a
# standalone separator like ``* * *`` is left untouched.
_LEADING_DANGLING_RE = re.compile(r"^[*_]{1,3}(?=\S)")
_TRAILING_DANGLING_RE = re.compile(r"(?<=\S)[*_]{1,3}$")


def _strip_surrounding_emphasis(title: str) -> str:
    """Remove Markdown emphasis that wraps (or dangles at the edge of) a heading title.

    Balanced wrapping (``***foo***``) is unwrapped first; then any unbalanced
    leading/trailing ``*``/``_`` run glued to text is trimmed, so a converter's
    dangling ``**`` does not pollute the stored title.
    """
    s = title.strip()
    m = _SURROUNDING_EMPHASIS_RE.match(s)
    while m:
        s = m.group(2).strip()
        m = _SURROUNDING_EMPHASIS_RE.match(s)
    s = _LEADING_DANGLING_RE.sub("", s)
    s = _TRAILING_DANGLING_RE.sub("", s)
    return s


def _outline_depth(title: str) -> int | None:
    """Depth of a heading's leading outline number (``3.4`` -> 2), or None if unnumbered."""
    match = _OUTLINE_NUMBER_RE.match(title)
    if not match:
        return None
    return match.group(1).count(".") + 1


def _normalize_title_for_dedup(title: str) -> str:
    """Normalize a heading title for repeat-detection (emphasis/whitespace-stripped, lowercased)."""
    stripped = _DEDUP_EMPHASIS_RE.sub("", title or "")
    return _DEDUP_WS_RE.sub(" ", stripped).strip().lower()


def _dedupe_page_header_headings(
    headings: list[tuple[str, int, int]], raw_lines: list[str]
) -> list[tuple[str, int, int]]:
    """Drop repeated page-header headings whose own body is empty.

    PDF/Office → Markdown conversions repeat a company-name or title line as a
    ``#`` heading before every statement (e.g. ``# **Advanced Micro Devices, Inc.**``
    ahead of each financial statement). Each such heading owns no body — the next
    heading follows on a nearby line — so it would become a near-empty section
    (the "empty sections" of the algorithmic path). Drop a heading when its body
    (the lines between it and the next heading) is entirely blank AND its
    normalized title has already appeared earlier in the document. Real
    sub-headings, first occurrences, and headings with body are always kept, so no
    content is lost and the document stays byte-for-byte reconstructable.
    """
    kept: list[tuple[str, int, int]] = []
    seen: set[str] = set()
    n = len(headings)
    for i, (title, level, line_start) in enumerate(headings):
        next_start = headings[i + 1][2] if i + 1 < n else None
        norm = _normalize_title_for_dedup(title)
        is_repeat = bool(norm) and norm in seen
        empty_body = False
        if next_start is not None:
            # Body lines: 0-indexed slice from just after the heading line up to
            # (excluding) the next heading's line.
            body = raw_lines[line_start : next_start - 1]
            empty_body = all(not line.strip() for line in body)
        if is_repeat and empty_body:
            continue
        kept.append((title, level, line_start))
        seen.add(norm)
    return kept


def _infer_levels(titles: list[str], md_levels: list[int]) -> list[int]:
    """Derive logical heading levels, trusting Markdown unless it is degenerate.

    PDF/Doc -> Markdown conversions sometimes emit inconsistent ``#`` levels (the
    same "3.1"/"3.5" heading may come out as H4 or H1). When the Markdown levels
    are *degenerate* (a single distinct level, or one level covering >=85% of
    headings) AND the document carries a coherent outline-number scheme, the
    outline number is the reliable structure signal: a heading's level is its
    number's depth (``1`` -> 1, ``1.1`` -> 2), and an unnumbered heading nests one
    level below the most recent numbered heading.

    When the Markdown levels are *not* degenerate, they are authoritative and
    returned unchanged: many converters (and hand-authored Markdown) already encode
    the real H1-H6 hierarchy, and a handful of coincidentally numeric titles
    (interest rates, exhibit indices) must not be allowed to flatten it.
    """
    if not md_levels:
        return md_levels
    counts = Counter(md_levels)
    modal_count = counts.most_common(1)[0][1]
    distinct = len(counts)
    degenerate = distinct <= 1 or modal_count / len(md_levels) >= 0.85
    if not degenerate:
        return md_levels

    depths = [_outline_depth(t) for t in titles]
    if sum(d is not None for d in depths) < 3:
        return md_levels

    levels: list[int] = []
    last_numbered_level = 0
    for depth in depths:
        if depth is not None:
            level = depth
            last_numbered_level = level
        else:
            level = last_numbered_level + 1
        levels.append(level)
    return levels


class FlatSection(BaseModel):
    """A single section, before hierarchy is resolved into graph edges."""

    title: str = Field(..., description="Heading text (or the root-section title)")
    level: int = Field(..., description="Heading level, 0 (synthetic root) or 1 (H1) to 6 (H6)")
    line_start: int = Field(..., description="1-indexed source line where this section's own text starts")
    line_end: int = Field(..., description="1-indexed source line where this section's own text ends (inclusive)")
    text: str = Field(..., description="Own Markdown text: heading line + body up to the next heading (any level)")
    token_count: int = Field(..., description="Approximate token count (whitespace/punctuation based estimate)")
    parent_index: int | None = Field(
        default=None, description="Index of the parent section within the same flat list, or None for a root section"
    )
    description: str | None = Field(default=None, description="One-sentence routing description of the section")
    summary: str | None = Field(default=None, description="Short paragraph summary (substantial sections only)")
    summary_source: str | None = Field(
        default=None, description="How description/summary were produced (e.g. 'llm'), or None when not yet set"
    )


def _estimate_token_count(text: str) -> int:
    """Rough token-count estimate (word + punctuation split) — no tokenizer dependency."""
    return len(re.findall(r"\w+|[^\w\s]", text))


def _detect_heuristic_headings(raw: str) -> list[tuple[str, int, int]]:
    """Detect headings from document text heuristics when no Markdown '#' headings exist.

    Identifies standard structured document headings (Parts, Items, Chapters, Articles,
    Notes, Financial Statements) and standalone uppercase titles surrounded by blank lines.
    """
    lines = raw.splitlines()
    headings: list[tuple[str, int, int]] = []
    in_toc_block = False
    toc_lines_passed = 0

    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            if in_toc_block:
                toc_lines_passed += 1
                if toc_lines_passed > 80:
                    in_toc_block = False
            continue

        # Filter out lines that are inside the initial Table of Contents summary block
        if _TOC_HEADER_RE.match(s) and i < 300:
            in_toc_block = True
            toc_lines_passed = 0
            continue
        if in_toc_block:
            toc_lines_passed += 1
            if toc_lines_passed > 100 or i > 350:
                in_toc_block = False
            else:
                continue

        prev_blank = i == 0 or not lines[i - 1].strip()
        next_blank = i == len(lines) - 1 or not lines[i + 1].strip()

        # Rule 1: Structured document prefixes (Part, Item, Section, Chapter, Note, etc.)
        if _HEURISTIC_HEADING_RE.match(s) and len(s) < 140:
            if re.match(r"^(?:PART|Part|SECTION|Section|CHAPTER|Chapter)\b", s, re.IGNORECASE):
                level = 1
            elif (
                re.match(r"^(?:ITEM|Item|ARTICLE|Article)\b", s, re.IGNORECASE)
                or "STATEMENTS" in s.upper()
                or "BALANCE" in s.upper()
                or "BILAN" in s.upper()
            ):
                level = 2
            elif re.match(
                r"^(?:Note|NOTE|EXHIBIT|Exhibit|ANNEX|Annex|ANNEXE|Annexe|APPENDIX|Appendix)\b", s, re.IGNORECASE
            ):
                level = 3
            else:
                level = 2
            title = _strip_surrounding_emphasis(s)
            headings.append((title, level, i + 1))
            continue

        # Rule 2: Standalone multi-word uppercase titles surrounded by blank lines
        words = s.split()
        if s.isupper() and prev_blank and next_blank and not s.isdigit() and not s.startswith("<!--"):
            if 2 <= len(words) <= 10 and 6 <= len(s) <= 80:
                if not any(token in s for token in _NON_HEADING_UPPERCASE_TOKENS):
                    title = _strip_surrounding_emphasis(s)
                    headings.append((title, 2, i + 1))

    return headings


def detect_headings(raw: str) -> list[tuple[str, int, int]]:
    """Find the document's top-level headings and their logical levels.

    Uses ``markdown-it-py`` so headings inside fenced code blocks, inline code, or
    blockquotes are ignored, and each heading's source line number is known
    precisely. Page-marker headings (``Page 12``) are dropped — they are PDF/Doc
    conversion artifacts with no structural meaning. For numbered documents the
    unreliable source ``#`` levels are re-derived from the outline numbers.
    When no Markdown headings exist in the document, heuristics for SEC / financial
    filing headings (Items, Parts, Notes, Financial Statements) are used.

    Returns:
        ``(title, level, line_start)`` tuples in document order, where
        ``line_start`` is 1-indexed. Empty when the document has no headings.
    """
    from markdown_it import MarkdownIt

    md = MarkdownIt("commonmark", {"html": True}).enable("table").enable("strikethrough")
    tokens = md.parse(raw)

    headings: list[tuple[str, int, int]] = []  # (title, level, line_start 1-indexed)
    depth = 0
    for i, tok in enumerate(tokens):
        if tok.type == "heading_open" and depth == 0:
            level = int(tok.tag[1:])  # "h2" -> 2
            line_start = (tok.map[0] if tok.map else 0) + 1
            title = ""
            if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                title = _strip_surrounding_emphasis(tokens[i + 1].content.strip())
            # Drop page-marker headings ("Page 12"): their text stays inline.
            if not _PAGE_MARKER_RE.match(title):
                headings.append((title, level, line_start))
        depth += tok.nesting

    if not headings:
        headings = _detect_heuristic_headings(raw)

    inferred_levels = _infer_levels([h[0] for h in headings], [h[1] for h in headings])
    inferred = [(title, inferred_levels[idx], line_start) for idx, (title, _, line_start) in enumerate(headings)]
    return _dedupe_page_header_headings(inferred, raw.splitlines())


def slice_sections(raw: str, headings: list[tuple[str, int, int]]) -> list[FlatSection]:
    """Slice *raw* into non-overlapping sections anchored at *headings*.

    Each heading's section owns its heading line plus body up to the line before
    the next heading of ANY level (a nested subsection's lines belong to the
    subsection, not the parent). Any text before the first heading — or a document
    with no headings at all — is captured by a synthetic level-0 root section.
    ``headings`` is ``(title, level, line_start)`` in document order, with
    ``line_start`` 1-indexed. The result is never empty, and concatenating the
    sections' ``text`` in order reconstructs *raw*.
    """
    lines = raw.splitlines()
    total_lines = len(lines)

    sections: list[FlatSection] = []

    first_heading_line = headings[0][2] if headings else None
    if first_heading_line is None or first_heading_line > 1:
        root_end = (first_heading_line - 1) if first_heading_line is not None else total_lines
        root_end = max(root_end, 1)
        root_text = "\n".join(lines[0:root_end])
        sections.append(
            FlatSection(
                title=ROOT_SECTION_TITLE,
                level=0,
                line_start=1,
                line_end=root_end,
                text=root_text,
                token_count=_estimate_token_count(root_text),
            )
        )

    for idx, (title, level, line_start) in enumerate(headings):
        next_line = headings[idx + 1][2] if idx + 1 < len(headings) else total_lines + 1
        line_end = next_line - 1
        text = "\n".join(lines[line_start - 1 : line_end])
        sections.append(
            FlatSection(
                title=title or f"(untitled H{level})",
                level=level,
                line_start=line_start,
                line_end=max(line_end, line_start),
                text=text,
                token_count=_estimate_token_count(text),
            )
        )

    stack: list[int] = []
    for idx, section in enumerate(sections):
        while stack and sections[stack[-1]].level >= section.level:
            stack.pop()
        section.parent_index = stack[-1] if stack else None
        stack.append(idx)

    return sections


def parse_markdown_tree(raw: str) -> list[FlatSection]:
    """Parse *raw* Markdown into a flat, order-preserving list of sections.

    Every document yields at least one section. Section line ranges partition the
    document with no overlap, so concatenating the sections' ``text`` in
    ``sequence`` order reconstructs the original document.

    Args:
        raw: Full Markdown document text.

    Returns:
        Flat list of `FlatSection` in document order (never empty). Parent/child
        hierarchy is resolved with a level-based stack and stored as `parent_index`.
    """
    return slice_sections(raw, detect_headings(raw))
