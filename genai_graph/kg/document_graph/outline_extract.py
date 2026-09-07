"""LLM outline extraction for the Document Graph build.

Takes a Markdown document and asks a (typically cheap, large-context "flash")
model for its **table of contents** plus a one-sentence description of every
section (and a short summary of the substantial ones) — *without* re-emitting
the section content, so the output stays small regardless of document size.

The outline is a content-free JSON artifact, cached by ``markdown_hash`` (and a
policy/LLM hash) so re-runs are free. A later deterministic pass
(:func:`genai_graph.kg.document_graph.outline_merge.merge_outline`) reconciles
the outline's heading anchors against the Markdown to produce the actual
section nodes. Sending the outline separately from the text — instead of asking
the model to repeat each section's body — keeps output tokens low on
million-token documents.

Two failure modes both degrade to "no outline" (the build then falls back to
the algorithmic ``parse_markdown_tree`` for that document, with no summaries):
the document exceeding the model's context window (no LLM call is made), and
the LLM call itself failing.
"""

from __future__ import annotations

import hashlib
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from genai_tk.utils.tokens import count_tokens
from loguru import logger
from pydantic import BaseModel, Field, model_validator

from genai_graph.kg.document_graph.summarize import _clean_text, _is_length_limit_error
from genai_graph.kg.document_graph.tree_parser import (
    _TOC_HEADER_RE,
    FlatSection,
    detect_headings,
    slice_sections,
)

_DEFAULT_LLM_TAG = "default"

# "Page 12" conversion artifacts that leak in from PDF/Office -> Markdown.
_PAGE_MARKER_RE = re.compile(r"(?im)^\s*#*\s*page\s+\d+\s*$")


def _extract_toc_excerpt(raw: str, max_lines: int = 350) -> tuple[str | None, int, int]:
    """Extract candidate Table of Contents / preamble excerpt for fast structure analysis.

    Returns:
        (toc_text, start_line_1_indexed, end_line_1_indexed) or (None, 1, 1) if no TOC header is found.
    """
    lines = raw.splitlines()
    for i, line in enumerate(lines[:500]):
        if _TOC_HEADER_RE.match(line.strip()):
            start = max(0, i)
            end = start + 1
            blank_streak = 0
            for j in range(start + 1, min(len(lines), start + max_lines)):
                lj = lines[j].strip()
                if not lj:
                    blank_streak += 1
                else:
                    if (
                        (blank_streak >= 1 and (lj.startswith("#") or bool(_PAGE_MARKER_RE.match(lj))))
                        or (blank_streak >= 2 and not lj.startswith("|") and not lj.startswith("["))
                    ) and (j > start + 2):
                        end = j
                        break
                    blank_streak = 0
                end = j + 1
            return "\n".join(lines[start:end]), start + 1, end
    return None, 1, 1


class TocPreambleEntry(BaseModel):
    """One section or table in the extracted Table of Contents."""

    title: str = Field(..., description="Heading or section title as listed in the Table of Contents")
    level: int = Field(
        default=1,
        description="Hierarchical level: 1 (Part/Chapter/Major Section), 2 (Item/Section), 3 (Table/Note/Subsection)",
    )
    page: str | None = Field(default=None, description="Page number or identifier if given in the TOC, otherwise None")

    @model_validator(mode="before")
    @classmethod
    def _coerce_entry(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"title": data.strip(), "level": 1}
        if isinstance(data, dict):
            # If page is int or float, convert to str
            if "page" in data and data["page"] is not None and not isinstance(data["page"], str):
                data["page"] = str(data["page"])
        return data


class DocumentTocPreamble(BaseModel):
    """Structured-output schema for TOC extraction from document preamble."""

    document_title: str | None = Field(default=None, description="Title of the document if identified")
    document_description: str | None = Field(
        default=None, description="ONE plain-text sentence, at most 20 words, on the whole document"
    )
    document_summary: str | None = Field(
        default=None, description="2-4 plain-text sentences, at most 60 words, abstracting the whole document"
    )
    entries: list[TocPreambleEntry] = Field(default_factory=list, description="Ordered list of TOC entries")

    @model_validator(mode="before")
    @classmethod
    def _coerce_input(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"entries": data}
        if isinstance(data, dict):
            if "entries" not in data:
                for alt_key in ("sections", "toc", "items", "table_of_contents", "content", "tables"):
                    if alt_key in data and isinstance(data[alt_key], list):
                        data["entries"] = data[alt_key]
                        break
        return data


class OutlineEntry(BaseModel):
    """One section in the LLM's table of contents for a document."""

    title: str = Field(
        ..., description="The heading text EXACTLY as it appears on its own line in the document (used to locate it)."
    )
    level: int = Field(..., description="Heading level, 1 (top) to 6, inferred from numbering/TOC/indentation.")
    description: str | None = Field(
        default=None,
        description="One plain-text sentence (<=20 words) naming concrete subject matter (entities, metrics, "
        "products, scope) found under the heading. null for structural dividers with no real body. Never restate the title.",
    )
    summary: str | None = Field(
        default=None, description="Only for substantial sections: 2-3 plain-text sentences, at most 60 words."
    )


class BranchOutline(BaseModel):
    """Structured output for one branch's section descriptions."""

    sections: list[OutlineEntry] = Field(..., description="Every requested section in this branch, in exact order")

    @model_validator(mode="before")
    @classmethod
    def _coerce_input(cls, data: Any) -> Any:
        # Some models return the entries as a bare JSON array instead of the
        # expected {"sections": [...]} wrapper object.
        if isinstance(data, list):
            return {"sections": data}
        return data


class DocumentSummarySynthesis(BaseModel):
    """Structured output for whole-document description and summary."""

    document_description: str = Field(
        ..., description="ONE plain-text sentence, at most 20 words, on the whole document"
    )
    document_summary: str = Field(
        ..., description="2-4 plain-text sentences, at most 60 words, abstracting the whole document"
    )


class DocumentOutline(BaseModel):
    """Structured-output schema for one outline-extraction LLM call."""

    document_description: str = Field(
        ..., description="ONE plain-text sentence, at most 20 words, on the whole document."
    )
    document_summary: str = Field(
        ..., description="2-4 plain-text sentences, at most 60 words, abstracting the document."
    )
    sections: list[OutlineEntry] = Field(
        ..., description="Every section in document order; titles must appear verbatim."
    )


class OutlineConfig(BaseModel):
    """Policy and LLM settings for outline extraction."""

    llm: str | None = Field(default=None, description="LLM id (name@provider) or tag; None uses kg_build.llms.default")
    structure_strategy: str = Field(
        default="auto",
        description="Structure discovery strategy: 'auto' | 'algo' | 'toc_preamble' | 'llm_full'",
    )
    generate_summaries: bool = Field(
        default=True,
        description="Whether to generate section routing descriptions and summaries with LLM",
    )
    workers: int = Field(
        default=4,
        description="Parallel LLM workers for branch summarization",
    )
    context_safety_ratio: float = Field(
        default=0.9,
        description="Degrade (no LLM call) when the document's token count exceeds this fraction of the context window",
    )
    summary_min_tokens: int = Field(
        default=800, description="Heuristic passed to the prompt for what counts as a 'substantial' section"
    )
    max_description_words: int = Field(default=20, description="Target length of a section/document description")
    max_summary_words: int = Field(default=60, description="Target length of a section/document summary")
    max_description_chars: int = Field(default=180, description="Hard cap applied to a description after cleaning")
    max_summary_chars: int = Field(default=500, description="Hard cap applied to a summary after cleaning")
    llm_max_tokens: int | None = Field(
        default=None,
        description="Explicit max output tokens for the call; raise if a reasoning model exhausts its completion budget.",
    )
    retry_max_tokens: int = Field(
        default=32_000, description="max_tokens for the one automatic retry after a 'length limit reached' failure"
    )
    cache_root: str | None = Field(default=None, description="Directory for the content-addressed outline JSON cache")


class OutlineResult(BaseModel):
    """Outcome of extracting one document's outline (cached on disk)."""

    outline: DocumentOutline | None = Field(default=None, description="The extracted outline, or None when degraded")
    degraded: bool = Field(
        default=False, description="True when no outline was produced (over context window or failure)"
    )
    reason: str | None = Field(default=None, description="Why degradation happened, if it did")
    llm_calls: int = 0


class OutlineStats(BaseModel):
    """Aggregate outcome of the parallel outline pre-pass over a corpus."""

    total_files: int = 0
    degraded_count: int = 0
    llm_calls: int = 0
    warnings: list[str] = Field(default_factory=list)


def _resolve_llm_id(config: OutlineConfig) -> str:
    """Resolve the LLM id from the config or the global default."""
    if config.llm:
        return config.llm
    from genai_tk.config_mgmt.config_mngr import global_config

    return global_config().get_str("kg_build.llms.default", default=_DEFAULT_LLM_TAG) or _DEFAULT_LLM_TAG


def _context_window_for(llm_id: str) -> int | None:
    """Resolve a model's effective context window, or None if it cannot be determined."""
    from genai_tk.core.factories.llm_factory import get_llm_info

    try:
        return get_llm_info(llm_id).effective_context_window
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not resolve context window for {}: {}", llm_id, exc)
        return None


def _policy_hash(config: OutlineConfig, llm_id: str) -> str:
    """Stable short hash of the LLM + policy fields that affect the outline."""
    payload = (
        f"multi-tier-v1|{llm_id}|{config.structure_strategy}|{config.generate_summaries}|"
        f"{config.summary_min_tokens}|{config.max_description_words}|{config.max_summary_words}"
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _llm_tag_for_path(llm_id: str) -> str:
    """Filesystem-safe tag derived from the LLM id (e.g. gpt_4o_mini@edenai -> gpt_4o_mini_edenai)."""
    return llm_id.replace("@", "_").replace("/", "_")


def _cache_path(config: OutlineConfig, llm_id: str, markdown_hash: str) -> Path | None:
    """Return the content-addressed cache path for one document's outline, or None if caching is disabled."""
    if not config.cache_root:
        return None
    root = Path(config.cache_root) / f"{_llm_tag_for_path(llm_id)}__{_policy_hash(config, llm_id)}"
    return root / f"{markdown_hash}.json"


def _condense_table_smart(table_block: str, head_rows: int = 3, tail_rows: int = 2) -> str:
    """Condense a Markdown table by keeping header + first head_rows + last tail_rows with an omission row."""
    lines = table_block.strip().splitlines()
    if len(lines) <= head_rows + tail_rows + 2:
        return table_block

    # Find where header separator is (e.g. |---|---|)
    sep_idx = -1
    for i, line in enumerate(lines[:5]):
        stripped = line.strip()
        if re.match(r"^\s*\|?[\s:|-]+\|?\s*$", stripped) and set(stripped.replace("|", "")) <= set(" :-"):
            sep_idx = i
            break

    if sep_idx == -1:
        header_lines = lines[:1]
        data_lines = lines[1:]
    else:
        header_lines = lines[: sep_idx + 1]
        data_lines = lines[sep_idx + 1 :]

    if len(data_lines) <= head_rows + tail_rows:
        return table_block

    kept_head = data_lines[:head_rows]
    kept_tail = data_lines[-tail_rows:] if tail_rows > 0 else []
    omitted = len(data_lines) - head_rows - len(kept_tail)

    col_names = []
    if header_lines:
        col_names = [c.strip() for c in header_lines[0].split("|") if c.strip()]

    col_info = f"; columns: {', '.join(col_names[:6])}" if col_names else ""
    omission_line = f"| ... ({omitted} table rows omitted for brevity{col_info}) |"

    res = header_lines + kept_head + [omission_line] + kept_tail
    return "\n" + "\n".join(res) + "\n"


def _clean_markdown_for_prompt(raw: str, head_rows: int = 3, tail_rows: int = 2) -> str:
    """Drop ``Page N`` artifacts and condense bulky tables/numeric runs with head + tail row sampling."""
    # 1. Drop page markers
    text = _PAGE_MARKER_RE.sub("", raw)

    # 2. Condense Markdown pipe tables: keep header + head_rows + tail_rows
    table_re = re.compile(r"(?:^[ \t]*\|[^\n]+\|[ \t]*\n){4,}", re.MULTILINE)

    def _replace_table(m: re.Match) -> str:
        return _condense_table_smart(m.group(0), head_rows=head_rows, tail_rows=tail_rows)

    text = table_re.sub(_replace_table, text)

    # 3. Condense runs of 5+ numeric/currency lines (OCR plain-text tabular listings)
    def _condense_numbers(match: re.Match) -> str:
        num_lines = match.group(0).strip().splitlines()
        if len(num_lines) <= head_rows + tail_rows:
            return match.group(0)
        kept_head = num_lines[:head_rows]
        kept_tail = num_lines[-tail_rows:] if tail_rows > 0 else []
        omitted = len(num_lines) - head_rows - len(kept_tail)
        return (
            "\n"
            + "\n".join(kept_head)
            + f"\n[... {omitted} numeric data lines omitted ...]\n"
            + "\n".join(kept_tail)
            + "\n"
        )

    num_run_re = re.compile(
        r"(?:^[ \t]*(?:\$[\s\d,\.\(\)\-]+|\d[\d,\.\(\)\-\%\s]*|\([0-9,\.\s]+\))[ \t]*\n){5,}", re.MULTILINE
    )
    text = num_run_re.sub(_condense_numbers, text)

    return text


# Words that carry no routing signal on their own: generic filing/document
# labels, announcement verbs, and common English connectors. Used to spot
# descriptions that merely rephrase the heading title (a description whose
# significant words are all already in the title adds nothing for routing).
_RESTATEMENT_STOPWORDS = frozenset(
    {
        "section",
        "document",
        "filing",
        "report",
        "annual",
        "information",
        "overview",
        "statement",
        "part",
        "item",
        "note",
        "notes",
        "chapter",
        "page",
        "content",
        "contents",
        "data",
        "details",
        "type",
        "form",
        "kind",
        "following",
        "above",
        "below",
        "begins",
        "describes",
        "provides",
        "lists",
        "summarizes",
        "outlines",
        "explains",
        "introduces",
        "presents",
        "shows",
        "states",
        "indicates",
        "discusses",
        "covers",
        "includes",
        "contains",
        "the",
        "a",
        "an",
        "of",
        "for",
        "to",
        "in",
        "on",
        "and",
        "or",
        "as",
        "is",
        "are",
        "this",
        "these",
        "those",
        "its",
        "their",
        "with",
        "by",
        "from",
        "that",
        "which",
        "such",
        "into",
        "about",
        "under",
        "per",
        "also",
        "both",
        "each",
        "all",
        "any",
        "some",
    }
)


def _significant_words(text: str) -> set[str]:
    """Lowercase alphanumeric tokens of length >= 2, minus the restatement stopword set."""
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) >= 2} - _RESTATEMENT_STOPWORDS


def _is_title_restatement(title: str, description: str | None) -> bool:
    """True when a description adds no new significant word beyond its heading title.

    A description whose significant words are all already in the title (after
    dropping generic filing vocabulary) merely rephrases the heading, so it is
    dropped to a clean ``None`` rather than a useless one-liner. A description
    made entirely of generic words is always a restatement (empty set is a subset
    of any title); a description with real content words is never dropped.
    """
    if not description:
        return False
    return _significant_words(description) <= _significant_words(title)


def _clean_outline(outline: DocumentOutline, config: OutlineConfig) -> DocumentOutline:
    """Strip Markdown noise, drop title-restatements, and hard-truncate descriptions/summaries."""
    cleaned_sections: list[OutlineEntry] = []
    for entry in outline.sections:
        description = _clean_text(entry.description, config.max_description_chars) if entry.description else None
        if description and _is_title_restatement(entry.title, description):
            description = None
        cleaned_sections.append(
            entry.model_copy(
                update={
                    "description": description,
                    "summary": _clean_text(entry.summary, config.max_summary_chars) if entry.summary else None,
                }
            )
        )
    return outline.model_copy(
        update={
            "document_description": _clean_text(outline.document_description, config.max_description_chars),
            "document_summary": _clean_text(outline.document_summary, config.max_summary_chars),
            "sections": cleaned_sections,
        }
    )


# ---------------------------------------------------------------------------
# Heading-anchored enrichment (hybrid granularity)
# ---------------------------------------------------------------------------
#
# The LLM no longer "discovers" the table of contents (it collapsed the 10-K to
# ~27 coarse PART/ITEM sections, ignoring the document's real H1/H2/H3
# sub-headings). Instead the Markdown headings are detected algorithmically
# (:func:`~genai_graph.kg.document_graph.tree_parser.detect_headings`, reliable)
# and the LLM is asked to return ONE description/summary per listed heading, in
# order. The LLM's entries are then aligned back to the detected headings by
# title, so the cached outline carries the heading's verbatim title and its
# Markdown level (authoritative), plus the LLM's description/summary where one
# was provided. Headings the LLM skipped still become sections (with no
# description); LLM entries that match no heading (collapses/hallucinations) are
# dropped. The downstream merge then slices on the detected headings directly.

_LEADING_NOISE_RE = re.compile(r"^\s*(?:#{1,6}\s*|[-*+]\s+|>\s*|\d+[.)]\s*)")
_ENRICH_EMPHASIS_RE = re.compile(r"[*_`]{1,3}")
_ENRICH_WS_RE = re.compile(r"\s+")


def _normalize_title(text: str) -> str:
    """Normalize a heading title or source line for tolerant matching."""
    stripped = _LEADING_NOISE_RE.sub("", text or "")
    stripped = _ENRICH_EMPHASIS_RE.sub("", stripped)
    return _ENRICH_WS_RE.sub(" ", stripped).strip().lower()


def _titles_match(a: str, b: str) -> bool:
    """True when two titles are equal, or one contains the other, after normalization."""
    na, nb = _normalize_title(a), _normalize_title(b)
    if not na or not nb:
        return False
    return na == nb or na in nb or nb in na


def _find_heading_line(lines: list[str], title: str, cursor: int) -> int | None:
    """Return the next 0-based line index at/after *cursor* whose text matches *title*."""
    for i in range(cursor, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        if _titles_match(line, title):
            return i
    return None


def anchor_toc_preamble(
    raw: str,
    entries: list[TocPreambleEntry],
    toc_end_line: int = 1,
) -> list[tuple[str, int, int]]:
    """Anchor TOC preamble entries onto the body of the document to produce heading tuples.

    Scans sequential lines in raw (starting after the TOC preamble) to find matching
    headings in document order.

    Returns:
        list of (title, level, line_start_1_indexed)
    """
    lines = raw.splitlines()
    cursor = max(0, toc_end_line - 1)
    anchored: list[tuple[str, int, int]] = []

    for entry in entries:
        level = max(1, min(6, entry.level))
        title = entry.title.strip()
        if not title:
            continue
        line_idx = _find_heading_line(lines, title, cursor)
        if line_idx is not None:
            anchored.append((title, level, line_idx + 1))
            cursor = line_idx + 1

    return anchored


def _baml_result_to_model(model_cls: type[BaseModel], baml_result: Any) -> BaseModel:
    """Convert a BAML-generated pydantic object into the project model of the same shape."""
    if isinstance(baml_result, model_cls):
        return baml_result
    return model_cls.model_validate(baml_result.model_dump())


def _call_toc_preamble_llm(
    *, llm_id: str, filename: str, toc_text: str, max_tokens: int | None = None
) -> DocumentTocPreamble:
    """Extract table of contents and document overview from preamble text."""
    try:
        from genai_tk.extra.structured.baml_util import create_baml_options

        from genai_graph.baml_client import b

        baml_options = create_baml_options(llm_id) or {}
        baml_result = b.ExtractTocPreamble(
            filename=filename,
            toc_text=toc_text,
            baml_options=baml_options,
        )
        if isinstance(baml_result, DocumentTocPreamble):
            return baml_result
        return DocumentTocPreamble.model_validate(baml_result.model_dump())
    except Exception as exc:  # noqa: BLE001
        logger.debug("BAML preamble TOC extraction unavailable ({}); using LangChain structured output", exc)
        from genai_tk.core.factories.llm_factory import get_llm

        system = """
            You extract the structured Table of Contents from the preamble or beginning of a document.
            Return all sections/items/parts/tables in the EXACT order they appear in the Table of Contents as a JSON object.
            Also return `document_title`, `document_description` (1 sentence, <= 20 words), and `document_summary` (2-4 sentences, <= 60 words).
            For each entry:
            - `title`: the exact heading/section text as listed in the Table of Contents.
            - `level`: hierarchical depth: 1 (top-level Part/Chapter/Major Section), 2 (Item/Section/Sub-chapter), 3 (Table/Chart/Note/Subsection).
            - `page`: reported page number or identifier if given, else null.

            Do not invent sections not present in the Table of Contents.
        """
        user = f"""
            Document: {filename}

            --- Table of Contents excerpt ---
            {toc_text}
            --- end excerpt ---
        """
        llm_kwargs = {"max_tokens": max_tokens} if max_tokens is not None else {}
        structured_llm = get_llm(llm_id, **llm_kwargs).with_structured_output(DocumentTocPreamble)
        result = structured_llm.invoke([("system", system), ("user", user)])
        if isinstance(result, DocumentTocPreamble):
            return result
        return DocumentTocPreamble.model_validate(result)


def extract_toc_from_preamble(
    raw: str,
    filename: str,
    config: OutlineConfig,
    *,
    warnings: list[str],
) -> tuple[list[tuple[str, int, int]], DocumentTocPreamble | None]:
    """Extract TOC entries from document preamble using LLM and anchor them to document lines.

    Returns:
        (anchored_headings, preamble_toc)
    """
    toc_text, _start_line, end_line = _extract_toc_excerpt(raw)
    if not toc_text:
        return [], None

    llm_id = _resolve_llm_id(config)
    try:
        toc = _call_toc_preamble_llm(
            llm_id=llm_id, filename=filename, toc_text=toc_text, max_tokens=config.llm_max_tokens
        )
        anchored = anchor_toc_preamble(raw, toc.entries, toc_end_line=end_line)
        return anchored, toc
    except Exception as exc:  # noqa: BLE001
        msg = f"{filename}: preamble TOC extraction failed: {exc}"
        warnings.append(msg)
        logger.warning(msg)
        return [], None


def _render_headings_block(headings: list[tuple[str, int, int]]) -> str:
    """Render detected headings as a numbered ``[Llevel] title`` list for the prompt."""
    if not headings:
        return "(no headings detected)"
    return "\n".join(f"{i}. [L{level}] {title}" for i, (title, level, _line) in enumerate(headings, 1))


def _align_outline(outline: DocumentOutline, algo_headings: list[tuple[str, int, int]]) -> DocumentOutline:
    """Align an LLM outline onto the detected headings (title + level authoritative).

    Returns a ``DocumentOutline`` with exactly one ``OutlineEntry`` per detected
    heading, in document order: the heading's verbatim title and Markdown level,
    plus the LLM's ``description``/``summary`` where an LLM entry matched that
    heading by title (tolerant). Unmatched headings keep ``description``/``summary``
    as ``None``; LLM entries that match no heading are dropped. Document-level
    ``description``/``summary`` are preserved unchanged.
    """
    entries = list(outline.sections)
    used: list[bool] = [False] * len(entries)
    aligned: list[OutlineEntry] = []
    for ah_title, ah_level, _line_start in algo_headings:
        match_idx: int | None = None
        for j, entry in enumerate(entries):
            if used[j]:
                continue
            if _titles_match(ah_title, entry.title):
                match_idx = j
                used[j] = True
                break
        if match_idx is not None:
            entry = entries[match_idx]
            aligned.append(
                OutlineEntry(title=ah_title, level=ah_level, description=entry.description, summary=entry.summary)
            )
        else:
            aligned.append(OutlineEntry(title=ah_title, level=ah_level, description=None, summary=None))
    return outline.model_copy(update={"sections": aligned})


class SectionBranch(BaseModel):
    """A branch or group of sections to summarize in one LLM call."""

    branch_title: str
    sections: list[FlatSection]


def _split_into_section_branches(
    raw: str,
    algo_headings: list[tuple[str, int, int]],
    max_tokens_per_branch: int = 5000,
    max_sections_per_branch: int = 10,
) -> list[SectionBranch]:
    """Group sliced sections into hierarchical branches (Level 1 + nested L2-L6)."""
    if not algo_headings:
        return []

    sections = slice_sections(raw, algo_headings)
    heading_sections = [s for s in sections if s.level > 0]
    if not heading_sections:
        return []

    has_l1 = any(s.level == 1 for s in heading_sections)
    branches: list[SectionBranch] = []

    if has_l1:
        current_branch_title = heading_sections[0].title
        current_sections: list[FlatSection] = []
        current_tokens = 0

        for s in heading_sections:
            is_new_l1 = (s.level == 1) and len(current_sections) > 0
            is_oversized = (
                len(current_sections) >= max_sections_per_branch
                or current_tokens + s.token_count > max_tokens_per_branch
            )
            if is_new_l1 or (is_oversized and len(current_sections) >= 3):
                branches.append(SectionBranch(branch_title=current_branch_title, sections=current_sections))
                current_branch_title = s.title if s.level == 1 else f"{current_branch_title} (cont.)"
                current_sections = []
                current_tokens = 0

            current_sections.append(s)
            current_tokens += s.token_count

        if current_sections:
            branches.append(SectionBranch(branch_title=current_branch_title, sections=current_sections))
    else:
        current_sections = []
        current_tokens = 0
        branch_idx = 1
        for s in heading_sections:
            if current_sections and (
                len(current_sections) >= max_sections_per_branch
                or current_tokens + s.token_count > max_tokens_per_branch
            ):
                branches.append(
                    SectionBranch(
                        branch_title=f"Section Group {branch_idx}: {current_sections[0].title}",
                        sections=current_sections,
                    )
                )
                branch_idx += 1
                current_sections = []
                current_tokens = 0

            current_sections.append(s)
            current_tokens += s.token_count

        if current_sections:
            branches.append(
                SectionBranch(
                    branch_title=f"Section Group {branch_idx}: {current_sections[0].title}",
                    sections=current_sections,
                )
            )

    return branches


def _build_branch_prompt(
    *,
    filename: str,
    branch_title: str,
    branch_headings: list[tuple[str, int, int]],
    branch_raw: str,
    config: OutlineConfig,
) -> tuple[str, str]:
    system = f"""
        You enrich the table of contents for a document section branch that an AI agent reads
        to decide which section or table to open. The Markdown headings for this branch have
        ALREADY been detected and are listed in the user message (numbered, with their
        Markdown level as ``[Llevel]``). Return EXACTLY one ``sections`` entry per
        listed heading, in the same order, using each heading's verbatim title and its
        listed level, plus a description (and, for substantial sections, a summary).

        For every listed heading, return (in the same order as the list):
        - `title`: the heading text EXACTLY as listed (verbatim).
        - `level`: the level listed for that heading.
        - `description`: ONE plain-text sentence, at most {config.max_description_words}
          words, naming the CONCRETE subject matter under that heading — the specific
          entities, products, metrics, line items, time periods, or scope found in the body text.
          - If a heading is a Table (e.g. "Table 1", "Table 2", "Table FFO-1"), inspect the table
            headers and sample rows to state what metrics and categories the table covers
            (e.g., "Monthly receipts from individual income tax, corporation tax, and customs.").
          - NEVER restate or paraphrase the title alone (e.g. "Table 1" -> "Describes Table 1" is invalid).
          - If the heading is a structural divider with no real body text, set `description` to null.
        - `summary`: ONLY for substantial sections (more than roughly {config.summary_min_tokens} tokens):
          2-3 plain-text sentences, at most {config.max_summary_words} words. Leave null otherwise.

        HARD RULES:
        - Return exactly one ``sections`` entry per listed heading in exact order.
        - Never include a section's body text in your answer.
    """
    headings_block = _render_headings_block(branch_headings)
    user = f"""
        Document: {filename}
        Branch / Major Section: {branch_title}

        --- headings in this branch (return one section per heading, in this order) ---
        {headings_block}
        --- branch content ---
        {branch_raw}
        --- end branch content ---
    """
    return system, user


def _call_branch_llm(
    *,
    llm_id: str,
    filename: str,
    branch_title: str,
    branch_headings: list[tuple[str, int, int]],
    branch_raw: str,
    config: OutlineConfig,
    max_tokens: int | None,
) -> BranchOutline:
    """One branch-enrichment LLM call: BAML first (robust JSON parsing), LangChain fallback."""
    headings_block = _render_headings_block(branch_headings)
    try:
        from genai_tk.extra.structured.baml_util import create_baml_options

        from genai_graph.baml_client import b

        baml_options = create_baml_options(llm_id) or {}
        baml_result = b.ExtractBranchOutline(
            filename=filename,
            branch_title=branch_title,
            headings=headings_block,
            raw=branch_raw,
            max_description_words=config.max_description_words,
            max_summary_words=config.max_summary_words,
            summary_min_tokens=config.summary_min_tokens,
            baml_options=baml_options,
        )
        return _baml_result_to_model(BranchOutline, baml_result)  # type: ignore[return-value]
    except Exception as exc:  # noqa: BLE001
        logger.debug("BAML branch extraction unavailable ({}); using LangChain structured output", exc)

    from genai_tk.core.factories.llm_factory import get_llm

    system, user = _build_branch_prompt(
        filename=filename,
        branch_title=branch_title,
        branch_headings=branch_headings,
        branch_raw=branch_raw,
        config=config,
    )
    llm_kwargs = {"max_tokens": max_tokens} if max_tokens is not None else {}
    structured_llm = get_llm(llm_id, **llm_kwargs).with_structured_output(BranchOutline)
    out = structured_llm.invoke([("system", system), ("user", user)])
    if isinstance(out, BranchOutline):
        return out
    return BranchOutline.model_validate(out)


def _call_branch_llm_with_retry(
    *,
    llm_id: str,
    filename: str,
    branch_title: str,
    branch_headings: list[tuple[str, int, int]],
    branch_raw: str,
    config: OutlineConfig,
    warnings: list[str],
) -> BranchOutline | None:
    max_tokens = config.llm_max_tokens
    context = f"{filename} branch [{branch_title}]"
    for attempt in range(2):
        started = time.monotonic()
        try:
            res = _call_branch_llm(
                llm_id=llm_id,
                filename=filename,
                branch_title=branch_title,
                branch_headings=branch_headings,
                branch_raw=branch_raw,
                config=config,
                max_tokens=max_tokens,
            )
            if attempt > 0:
                logger.info("{}: branch retry succeeded ({:.1f}s)", context, time.monotonic() - started)
            return res
        except Exception as exc:  # noqa: BLE001
            if attempt == 0 and _is_length_limit_error(exc):
                max_tokens = max(max_tokens or 0, config.retry_max_tokens)
                msg = f"{context}: completion token limit reached; retrying with max_tokens={max_tokens}."
                warnings.append(msg)
                logger.warning(msg)
                continue
            msg = f"LLM call failed for {context}: {exc}"
            warnings.append(msg)
            logger.error(msg)
            return None
    return None


def _synthesize_document_summary(
    *,
    llm_id: str,
    filename: str,
    preamble_text: str,
    section_entries: list[OutlineEntry],
    config: OutlineConfig,
) -> tuple[str, str]:
    from genai_tk.core.factories.llm_factory import get_llm

    top_level_desc = "\n".join(f"- {e.title}: {e.description}" for e in section_entries[:15] if e.description)
    preamble_excerpt = preamble_text[:1500]
    try:
        from genai_tk.extra.structured.baml_util import create_baml_options

        from genai_graph.baml_client import b

        baml_options = create_baml_options(llm_id) or {}
        baml_result = b.SynthesizeDocumentSummary(
            filename=filename,
            preamble_text=preamble_excerpt,
            top_level_descriptions=top_level_desc,
            max_description_words=config.max_description_words,
            max_summary_words=config.max_summary_words,
            baml_options=baml_options,
        )
        res = _baml_result_to_model(DocumentSummarySynthesis, baml_result)  # type: ignore[assignment]
        return _clean_text(res.document_description, config.max_description_chars), _clean_text(
            res.document_summary, config.max_summary_chars
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("BAML summary synthesis unavailable ({}); using LangChain structured output", exc)

    system = f"""
        You generate the top-level document description and summary for a document library.
        - `document_description`: exactly ONE sentence, at most {config.max_description_words} words, stating what the document is and its scope.
        - `document_summary`: 2-4 sentences, at most {config.max_summary_words} words, abstracting the main topics, entities, and periods covered.
    """
    user = f"""
        Document: {filename}

        Preamble excerpt:
        {preamble_excerpt}

        Major Sections / Tables:
        {top_level_desc}
    """
    try:
        structured_llm = get_llm(llm_id).with_structured_output(DocumentSummarySynthesis)
        res = structured_llm.invoke([("system", system), ("user", user)])
        if not isinstance(res, DocumentSummarySynthesis):
            res = DocumentSummarySynthesis.model_validate(res)
        return _clean_text(res.document_description, config.max_description_chars), _clean_text(
            res.document_summary, config.max_summary_chars
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Document summary synthesis failed for {}: {}", filename, exc)
    return f"Document: {filename}", ""


def _summarize_branch_one(
    branch: SectionBranch,
    filename: str,
    config: OutlineConfig,
    llm_id: str,
    warnings: list[str],
) -> list[OutlineEntry]:
    branch_headings = [(s.title, s.level, s.line_start) for s in branch.sections]
    branch_text_parts = []
    for s in branch.sections:
        branch_text_parts.append(f"### {s.title}\n{_clean_markdown_for_prompt(s.text)}")
    branch_raw = "\n\n".join(branch_text_parts)

    try:
        res = _call_branch_llm_with_retry(
            llm_id=llm_id,
            filename=filename,
            branch_title=branch.branch_title,
            branch_headings=branch_headings,
            branch_raw=branch_raw,
            config=config,
            warnings=warnings,
        )
        if res is not None and res.sections:
            entry_map = {e.title: e for e in res.sections}
            out: list[OutlineEntry] = []
            for h_title, h_level, _ in branch_headings:
                if h_title in entry_map:
                    matched = entry_map[h_title]
                    out.append(
                        OutlineEntry(
                            title=h_title,
                            level=h_level,
                            description=_clean_text(matched.description, config.max_description_chars)
                            if matched.description
                            else None,
                            summary=_clean_text(matched.summary, config.max_summary_chars) if matched.summary else None,
                        )
                    )
                else:
                    out.append(OutlineEntry(title=h_title, level=h_level, description=None, summary=None))
            return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("{}: branch '{}' summarization failed: {}", filename, branch.branch_title, exc)

    return [OutlineEntry(title=s.title, level=s.level, description=None, summary=None) for s in branch.sections]


def _summarize_branches_parallel(
    raw: str,
    algo_headings: list[tuple[str, int, int]],
    filename: str,
    config: OutlineConfig,
    warnings: list[str],
    preamble_toc: DocumentTocPreamble | None = None,
) -> OutlineResult:
    llm_id = _resolve_llm_id(config)
    branches = _split_into_section_branches(raw, algo_headings)

    # If only 1 small branch and <= 4 headings (or 0 headings as in plain prose):
    if len(branches) <= 1 and len(algo_headings) <= 4:
        cleaned = _clean_markdown_for_prompt(raw)
        outline = _call_llm_with_retry(
            llm_id=llm_id,
            filename=filename,
            raw=cleaned,
            config=config,
            warnings=warnings,
            headings=algo_headings,
        )
        if outline is None:
            return OutlineResult(outline=None, degraded=True, reason="llm_call_failed", llm_calls=1)
        aligned = _align_outline(outline, algo_headings)
        return OutlineResult(outline=aligned, llm_calls=1)

    if not branches:
        entries = [OutlineEntry(title=title, level=level) for title, level, _ in algo_headings]
        return OutlineResult(
            outline=DocumentOutline(
                document_description=f"Document: {filename}",
                document_summary="",
                sections=entries,
            ),
            llm_calls=0,
        )

    workers = min(len(branches), max(1, config.workers))
    total_llm_calls = 0

    def _run_branch(b: SectionBranch) -> list[OutlineEntry]:
        return _summarize_branch_one(b, filename, config, llm_id, warnings)

    if workers <= 1:
        branch_results = [_run_branch(b) for b in branches]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            branch_results = list(pool.map(_run_branch, branches))

    total_llm_calls += len(branches)
    all_entries: list[OutlineEntry] = []
    for br in branch_results:
        all_entries.extend(br)

    doc_desc = None
    doc_sum = None
    if preamble_toc:
        doc_desc = preamble_toc.document_description
        doc_sum = preamble_toc.document_summary

    if not doc_desc:
        sections = slice_sections(raw, algo_headings)
        preamble_text = sections[0].text if sections and sections[0].level == 0 else ""
        doc_desc, doc_sum = _synthesize_document_summary(
            llm_id=llm_id,
            filename=filename,
            preamble_text=preamble_text,
            section_entries=all_entries,
            config=config,
        )
        total_llm_calls += 1

    doc_outline = DocumentOutline(
        document_description=doc_desc or f"Document: {filename}",
        document_summary=doc_sum or "",
        sections=all_entries,
    )
    cleaned_outline = _clean_outline(doc_outline, config)
    aligned_outline = _align_outline(cleaned_outline, algo_headings)
    return OutlineResult(outline=aligned_outline, llm_calls=total_llm_calls)


def _build_prompt(*, filename: str, raw: str, config: OutlineConfig) -> tuple[str, str]:
    """Build the (system, user) prompt for heading-anchored outline enrichment."""
    system = f"""
        You enrich the table of contents for a document library that an AI agent reads
        to decide which section to open. The document's Markdown headings have ALREADY
        been detected for you and are listed in the user message (numbered, with their
        Markdown level as ``[Llevel]``). Return EXACTLY one ``sections`` entry per
        listed heading, in the same order, using each heading's verbatim title and its
        listed level, plus a description (and, for substantial sections, a summary)
        that you write from the document content under that heading.

        For every listed heading, return (in the same order as the list):
        - `title`: the heading text EXACTLY as listed (verbatim).
        - `level`: the level listed for that heading.
        - `description`: ONE plain-text sentence, at most {config.max_description_words}
          words, naming the CONCRETE subject matter under that heading — the specific
          entities, products, metrics, line items, years, or scope found in the body text.
        - `summary`: ONLY for substantial sections (more than roughly
          {config.summary_min_tokens} tokens, or {config.summary_min_tokens * 4} words):
          2-3 plain-text sentences, at most {config.max_summary_words} words. Leave null otherwise.

        HARD RULES:
        - Return exactly one ``sections`` entry per listed heading in exact order.
        - Never include a section's body text in your answer.
        - Also return `document_description` (one sentence, at most {config.max_description_words} words)
          and `document_summary` (2-4 sentences, at most {config.max_summary_words} words).
    """
    user = """
        Document: {filename}

        --- headings detected in this document (return one section per heading, in this order) ---
        {headings}
        --- full document ---
        {raw}
        --- end document ---
    """
    return system, user


def _call_llm(
    *,
    llm_id: str,
    filename: str,
    raw: str,
    config: OutlineConfig,
    max_tokens: int | None,
    headings: list[tuple[str, int, int]] | None = None,
) -> DocumentOutline:
    """The LLM call boundary — isolated so tests can substitute a fake implementation."""
    from genai_tk.core.factories.llm_factory import get_llm

    target_headings = headings if headings is not None else detect_headings(raw)
    system, _ = _build_prompt(filename=filename, raw=raw, config=config)
    headings_block = _render_headings_block(target_headings)
    cleaned_doc = _clean_markdown_for_prompt(raw)
    try:
        from genai_tk.extra.structured.baml_util import create_baml_options

        from genai_graph.baml_client import b

        baml_options = create_baml_options(llm_id) or {}
        baml_result = b.ExtractOutline(
            filename=filename,
            headings=headings_block,
            raw=cleaned_doc,
            max_description_words=config.max_description_words,
            max_summary_words=config.max_summary_words,
            baml_options=baml_options,
        )
        return _baml_result_to_model(DocumentOutline, baml_result)  # type: ignore[return-value]
    except Exception as exc:  # noqa: BLE001
        logger.debug("BAML outline extraction unavailable ({}); using LangChain structured output", exc)

    user = f"""Document: {filename}

--- headings detected in this document (return one section per heading, in this order) ---
{headings_block}
--- full document ---
{cleaned_doc}
--- end document ---"""
    llm_kwargs = {"max_tokens": max_tokens} if max_tokens is not None else {}
    structured_llm = get_llm(llm_id, **llm_kwargs).with_structured_output(DocumentOutline)
    result = structured_llm.invoke([("system", system), ("user", user)])
    if isinstance(result, DocumentOutline):
        return result
    return DocumentOutline.model_validate(result)


def _call_llm_with_retry(
    *,
    llm_id: str,
    filename: str,
    raw: str,
    config: OutlineConfig,
    warnings: list[str],
    headings: list[tuple[str, int, int]] | None = None,
) -> DocumentOutline | None:
    """Call the LLM, retrying once with a larger completion budget on a length-limit failure."""
    max_tokens = config.llm_max_tokens
    context = f"{filename} outline"
    for attempt in range(2):
        started = time.monotonic()
        try:
            outline = _call_llm(
                llm_id=llm_id,
                filename=filename,
                raw=raw,
                config=config,
                max_tokens=max_tokens,
                headings=headings,
            )
            if attempt > 0:
                logger.info("{}: outline retry succeeded ({:.1f}s)", context, time.monotonic() - started)
            return _clean_outline(outline, config)
        except Exception as exc:  # noqa: BLE001
            if attempt == 0 and _is_length_limit_error(exc):
                max_tokens = max(max_tokens or 0, config.retry_max_tokens)
                msg = f"{context}: hit the completion token limit; retrying with max_tokens={max_tokens}."
                warnings.append(msg)
                logger.warning(msg)
                continue
            msg = f"LLM call failed for {context}: {exc}"
            warnings.append(msg)
            logger.error(msg)
            return None
    return None


def _load_cached(cache_path: Path) -> OutlineResult | None:
    """Load a cached outline result, or None if absent/stale-unreadable."""
    if not cache_path.exists():
        return None
    try:
        return OutlineResult.model_validate_json(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Stale/invalid outline cache {} ({}); re-extracting", cache_path, exc)
        return None


def _write_cached(cache_path: Path, result: OutlineResult) -> None:
    """Persist an outline result so later merges never re-call the LLM."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    except OSError as exc:  # noqa: BLE001
        logger.warning("Could not write outline cache {}: {}", cache_path, exc)


def extract_outline(
    md_text: str,
    markdown_hash: str,
    filename: str,
    config: OutlineConfig,
    *,
    warnings: list[str],
) -> OutlineResult:
    """Extract a document's outline (TOC + summaries), cache-addressed by *markdown_hash*.

    Idempotent: a fresh cache hit returns the stored result without an LLM call.
    Supports multi-tier structure discovery:
    - ``algo``: fast deterministic heading parsing (markdown-it + domain heuristics)
    - ``toc_preamble``: extracts printed TOC from preamble using LLM and anchors to body lines
    - ``llm_full``: full document outline extraction
    - ``auto``: chooses preamble TOC when a printed TOC exists in preamble, native markdown
      when rich Markdown headings are present, and heuristic fallback otherwise.

    Args:
        md_text: Full Markdown document text.
        markdown_hash: Content hash of the Markdown rendering (cache key + identity).
        filename: Document filename, for prompt context and log messages.
        config: Outline policy and LLM settings.
        warnings: List to append human-readable warnings to.

    Returns:
        `OutlineResult`; ``.outline`` is None when degraded.
    """
    llm_id = _resolve_llm_id(config)
    cache_path = _cache_path(config, llm_id, markdown_hash)
    if cache_path is not None:
        cached = _load_cached(cache_path)
        if cached is not None:
            return cached.model_copy(update={"llm_calls": 0})

    strategy = config.structure_strategy
    algo_headings: list[tuple[str, int, int]] = []
    preamble_toc_used = False
    toc_obj: DocumentTocPreamble | None = None

    # 1. Determine structure according to configured strategy
    if strategy == "algo":
        algo_headings = detect_headings(md_text)
    elif strategy == "toc_preamble":
        anchored, toc_obj = extract_toc_from_preamble(md_text, filename, config, warnings=warnings)
        algo_headings = anchored or detect_headings(md_text)
        preamble_toc_used = bool(anchored)
    elif strategy == "llm_full":
        algo_headings = detect_headings(md_text)
    elif strategy == "auto":
        raw_headings = detect_headings(md_text)
        toc_text, _s, _e = _extract_toc_excerpt(md_text)
        # If a printed TOC exists and there are few native markdown headings (<10), extract via preamble
        if toc_text is not None and len([h for h in raw_headings if h[1] > 0]) < 10:
            anchored, toc_obj = extract_toc_from_preamble(md_text, filename, config, warnings=warnings)
            if len(anchored) >= 3:
                algo_headings = anchored
                preamble_toc_used = True
            else:
                algo_headings = raw_headings
        else:
            algo_headings = raw_headings
    else:
        algo_headings = detect_headings(md_text)

    # 2. If summaries are disabled, return pure structure with zero additional LLM summary calls
    if not config.generate_summaries:
        entries = [OutlineEntry(title=title, level=level) for title, level, _ in algo_headings]
        doc_outline = DocumentOutline(
            document_description=f"Document with {len(entries)} section(s)",
            document_summary="",
            sections=entries,
        )
        result = OutlineResult(outline=doc_outline, llm_calls=(1 if preamble_toc_used else 0))
        if cache_path is not None:
            _write_cached(cache_path, result)
        return result

    # 3. Summaries requested: check context window safety
    cleaned = _clean_markdown_for_prompt(md_text)
    doc_tokens = count_tokens(cleaned)
    context_window = _context_window_for(llm_id)
    if context_window and doc_tokens > context_window * config.context_safety_ratio:
        msg = (
            f"{filename}: ~{doc_tokens} tokens over {config.context_safety_ratio:.0%} of "
            f"{llm_id}'s {context_window}-token context window; degrading to algorithmic parsing (no summaries)."
        )
        warnings.append(msg)
        logger.warning(msg)
        result = OutlineResult(
            outline=None,
            degraded=True,
            reason="context_window_overflow",
            llm_calls=(1 if preamble_toc_used else 0),
        )
        if cache_path is not None:
            _write_cached(cache_path, result)
        return result

    # Summaries requested: use hierarchical parallel branch summarization
    result = _summarize_branches_parallel(
        raw=md_text,
        algo_headings=algo_headings,
        filename=filename,
        config=config,
        warnings=warnings,
        preamble_toc=toc_obj,
    )
    if preamble_toc_used:
        result = result.model_copy(update={"llm_calls": result.llm_calls + 1})

    if cache_path is not None:
        _write_cached(cache_path, result)
    return result
