"""Reconcile an LLM outline with the Markdown to produce section nodes.

The LLM (:mod:`genai_graph.kg.document_graph.outline_extract`) emits a
content-free outline: an ordered list of sections, each with a *verbatim*
heading title, a level, and a description/summary. It does not emit line
numbers (a flash model miscounts them on a large document) or section text.

`merge_outline` recovers the line ranges and text itself by *anchoring* each
outline title back to the Markdown: a forward, sequential line scan finds the
next line whose text matches the title, and the existing
:func:`genai_graph.kg.document_graph.tree_parser.slice_sections` then slices
the document into non-overlapping sections exactly as the algorithmic parser
does — so the result is byte-for-byte reconstructable and shape-compatible with
the algorithmic path. When a title cannot be matched to a line, the nearest
algorithmic heading (from :func:`~genai_graph.kg.document_graph.tree_parser.detect_headings`)
is used as a fallback anchor; if that also fails, the entry's description is
folded into the most recent matched section rather than dropping a section.

If no outline title can be reconciled at all, the algorithmic structure is used
verbatim with no summaries (a "structure reconciliation" degradation).
"""

from __future__ import annotations

import re

from loguru import logger

from genai_graph.kg.document_graph.outline_extract import DocumentOutline, OutlineEntry
from genai_graph.kg.document_graph.tree_parser import FlatSection, slice_sections

# Leading Markdown heading hashes, list bullets, blockquote markers, and
# outline numbers, plus surrounding emphasis/whitespace — stripped before
# comparing a heading title to a source line.
_LEADING_NOISE_RE = re.compile(r"^\s*(?:#{1,6}\s*|[-*+]\s+|>\s*|\d+[.)]\s*)")
_EMPHASIS_RE = re.compile(r"[*_`]{1,3}")
_WS_RE = re.compile(r"\s+")


def _normalize_title(text: str) -> str:
    """Normalize a heading title or source line for tolerant matching."""
    stripped = _LEADING_NOISE_RE.sub("", text or "")
    stripped = _EMPHASIS_RE.sub("", stripped)
    return _WS_RE.sub(" ", stripped).strip().lower()


def _titles_match(a: str, b: str) -> bool:
    """True when two titles are equal, or one contains the other, after normalization."""
    na, nb = _normalize_title(a), _normalize_title(b)
    if not na or not nb:
        return False
    return na == nb or na in nb or nb in na


def _find_heading_line(lines: list[str], title: str, cursor: int) -> int | None:
    """Return the next 0-based line index at/after *cursor* whose text matches *title*."""
    for i in range(cursor, len(lines)):
        if _titles_match(lines[i], title):
            return i
    return None


def _fold_into(section: FlatSection, entry: OutlineEntry) -> None:
    """Append an unmatched entry's description/summary onto an existing section."""
    if entry.description:
        prefix = section.description or ""
        section.description = f"{prefix} {entry.description}".strip() if prefix else entry.description
        section.summary_source = "llm"
    if entry.summary:
        prefix_s = section.summary or ""
        section.summary = f"{prefix_s} {entry.summary}".strip() if prefix_s else entry.summary


def merge_outline(raw: str, outline: DocumentOutline, algo_headings: list[tuple[str, int, int]]) -> list[FlatSection]:
    """Anchor an LLM outline onto *raw* to produce a flat, ordered section list.

    Post-refactor the cached outline is already *aligned* to the detected
    headings (one ``OutlineEntry`` per heading, in document order, carrying the
    heading's verbatim title and detected level — see
    :func:`genai_graph.kg.document_graph.outline_extract._align_outline`). In
    that case the sections slice directly on ``algo_headings`` and the LLM's
    ``description``/``summary`` attach by index: no fragile title-to-line scan,
    and the detected title/level is authoritative.

    When the entry and heading counts disagree (pre-refactor caches or
    hand-crafted test outlines), fall back to the original title-anchored
    reconciliation: each entry's verbatim title is matched to a source line by
    a forward sequential scan; unmatched entries fold into the most recent
    matched section; total failure degrades to the algorithmic structure.

    Args:
        raw: Full Markdown document text.
        outline: The LLM's content-free outline (titles + levels + descriptions).
        algo_headings: ``(title, level, line_start)`` from ``detect_headings``;
        authoritative title/level, and the wholesale fallback structure.

    Returns:
        Flat list of `FlatSection` in document order (never empty). Sections
        the LLM described carry ``description``/``summary`` and
        ``summary_source='llm'``; undescribed sections and the synthetic root do not.
    """
    n_entries = len(outline.sections)

    # Fast path: aligned outline (one entry per detected heading, in order).
    # The detected title/level is authoritative; attach LLM desc/summary by index.
    if n_entries == len(algo_headings) and n_entries > 0:
        sections = slice_sections(raw, algo_headings)
        heading_sections = [s for s in sections if s.level > 0]
        # slice_sections yields exactly one level>0 section per heading (plus the
        # synthetic root), so this zips 1:1 with the aligned outline entries.
        for entry, section in zip(outline.sections, heading_sections, strict=True):
            if entry.description:
                section.description = entry.description
            if entry.summary:
                section.summary = entry.summary
            if entry.description or entry.summary:
                section.summary_source = "llm"
        return sections

    # Fallback: title-anchored reconciliation for unaligned / hand-crafted outlines.
    lines = raw.splitlines()

    # If document has a TOC preamble, skip past it so we don't match entries inside the TOC itself
    from genai_graph.kg.document_graph.outline_extract import _extract_toc_excerpt

    _toc_text, _s, end_line = _extract_toc_excerpt(raw)
    cursor = max(0, end_line - 1) if _toc_text else 0  # 0-based line index consumed up to (exclusive)

    # entry_idx -> (line_start 1-indexed, entry); only for matched entries.
    matched: dict[int, tuple[int, OutlineEntry]] = {}
    unmatched_idx: set[int] = set()

    for entry_idx, entry in enumerate(outline.sections):
        line_idx = _find_heading_line(lines, entry.title, cursor) if entry.title else None
        if line_idx is not None:
            matched[entry_idx] = (line_idx + 1, entry)
            cursor = line_idx + 1
            continue
        # Fallback: first algorithmic heading at/after the cursor whose title matches.
        for a_title, _a_level, a_line in algo_headings:
            if a_line - 1 < cursor:
                continue
            if _titles_match(a_title, entry.title):
                matched[entry_idx] = (a_line, entry)
                cursor = a_line
                break
        else:
            unmatched_idx.add(entry_idx)

    if not matched:
        # Structure reconciliation failed entirely: degrade to the algorithmic
        # structure, with no summaries attached.
        logger.warning(
            "Outline reconciliation failed: 0/{} titles matched; using algorithmic structure without summaries.",
            n_entries,
        )
        return slice_sections(raw, algo_headings)

    confirmed = sorted(
        [(line_start, entry.level, entry.title, entry_idx) for entry_idx, (line_start, entry) in matched.items()],
        key=lambda c: c[0],
    )
    headings = [(title, level, line_start) for line_start, level, title, _entry_idx in confirmed]
    sections = slice_sections(raw, headings)

    # Map each confirmed entry to its heading section (skip the synthetic root).
    heading_sections = [s for s in sections if s.level > 0]
    entry_idx_to_section: dict[int, FlatSection] = {}
    for (_line_start, _level, _title, entry_idx), section in zip(confirmed, heading_sections, strict=False):
        entry_idx_to_section[entry_idx] = section
        section.description = matched[entry_idx][1].description
        section.summary = matched[entry_idx][1].summary
        section.summary_source = "llm"

    # Fold unmatched entries into the most recent matched section in entry order.
    last_section: FlatSection | None = None
    for entry_idx, entry in enumerate(outline.sections):
        if entry_idx in entry_idx_to_section:
            last_section = entry_idx_to_section[entry_idx]
        elif entry_idx in unmatched_idx:
            target = last_section if last_section is not None else (sections[0] if sections else None)
            if target is not None:
                _fold_into(target, entry)

    if unmatched_idx:
        logger.info(
            "Outline merge: {}/{} entries unmatched (folded into preceding sections).",
            len(unmatched_idx),
            n_entries,
        )

    return sections
