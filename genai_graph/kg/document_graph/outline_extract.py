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
from pathlib import Path

from genai_tk.utils.tokens import count_tokens
from loguru import logger
from pydantic import BaseModel, Field

from genai_graph.kg.document_graph.summarize import _clean_text, _is_length_limit_error
from genai_graph.kg.document_graph.tree_parser import _TOC_HEADER_RE, detect_headings

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


class DocumentTocPreamble(BaseModel):
    """Structured-output schema for TOC extraction from document preamble."""

    document_title: str | None = Field(default=None, description="Title of the document if identified")
    entries: list[TocPreambleEntry] = Field(default_factory=list, description="Ordered list of TOC entries")


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


def _clean_markdown_for_prompt(raw: str) -> str:
    """Drop ``Page N`` artifacts and condense bulky tables/numeric runs for the LLM input."""
    # 1. Drop page markers
    text = _PAGE_MARKER_RE.sub("", raw)

    # 2. Condense Markdown pipe tables: keep header + 2 rows, replace rest with placeholder
    def _condense_table(match: re.Match) -> str:
        table_lines = match.group(0).strip().splitlines()
        if len(table_lines) <= 4:
            return match.group(0)
        kept = table_lines[:3]
        omitted = len(table_lines) - 3
        return "\n" + "\n".join(kept) + f"\n| ... ({omitted} table rows omitted for outline extraction) |\n"

    table_re = re.compile(r"(?:^[ \t]*\|[^\n]+\|[ \t]*\n){4,}", re.MULTILINE)
    text = table_re.sub(_condense_table, text)

    # 3. Condense runs of 5+ numeric/currency lines (OCR plain-text tabular listings)
    def _condense_numbers(match: re.Match) -> str:
        num_lines = match.group(0).strip().splitlines()
        if len(num_lines) <= 4:
            return match.group(0)
        omitted = len(num_lines)
        return f"\n[... {omitted} numeric data lines omitted ...]\n"

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


def _call_toc_preamble_llm(
    *, llm_id: str, filename: str, toc_text: str, max_tokens: int | None = None
) -> DocumentTocPreamble:
    """Call LLM with structured output to extract table of contents from preamble text."""
    from genai_tk.core.factories.llm_factory import get_llm
    from genai_tk.core.prompts import def_prompt

    system = """
        You extract the structured Table of Contents from the preamble or beginning of a document.
        Return all sections/items/parts/tables in the EXACT order they appear in the Table of Contents.
        For each entry:
        - `title`: the exact heading/section text as listed in the Table of Contents.
        - `level`: hierarchical depth: 1 (top-level Part/Chapter/Major Section), 2 (Item/Section/Sub-chapter), 3 (Table/Chart/Note/Subsection).
        - `page`: reported page number or identifier if given, else null.

        Do not invent sections not present in the Table of Contents.
    """
    user = """
        Document: {filename}

        --- Table of Contents excerpt ---
        {toc_text}
        --- end excerpt ---
    """
    prompt = def_prompt(system=system, user=user)
    llm_kwargs = {"max_tokens": max_tokens} if max_tokens is not None else {}
    structured_llm = get_llm(llm_id, **llm_kwargs).with_structured_output(DocumentTocPreamble)
    result = (prompt | structured_llm).invoke({"filename": filename, "toc_text": toc_text})
    assert isinstance(result, DocumentTocPreamble)
    return result


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


def _build_prompt(*, filename: str, raw: str, config: OutlineConfig) -> tuple[str, str]:
    """Build the (system, user) prompt for heading-anchored outline enrichment.

    The Markdown headings have already been detected algorithmically (reliable)
    and are listed for the model in the user message. The model returns ONE
    section entry per listed heading, reusing each heading's verbatim title and
    listed level, plus a one-sentence description and (for substantial sections)
    a short summary based on the document content under that heading.

    The user message references the document through ``{filename}``/``{raw}``/``{headings}``
    template variables (filled at invoke time in :func:`_call_llm`) rather than
    baking the source text into the template string, so braces in the Markdown
    (e.g. LaTeX superscripts ``^{(1)}``) are not interpreted as prompt-template
    variables.
    """
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
          entities, products, metrics, line items, years, or scope found in the body
          text. No Markdown, no headings, no bullets, no line breaks.
          - NEVER restate or paraphrase the title. If the title already names the
            subject, your sentence must add facts not inferable from the title alone.
          - If the heading is a structural divider with no real body text (e.g. "PART I",
            "FORM 10-K", "INDEX", or a repeated company-name header above a statement),
            set `description` to null — a restatement adds nothing for routing.
          - Bad: title "PART I" -> "Begins Part I of the annual report."
            Good: title "Data Center Products" -> "Lists server CPUs (Genoa), GPUs
            (Instinct MI300), DPUs and adaptive SoCs with their target workloads."
          - Bad: title "Non-custom products" -> "Describes revenue recognition for
            non-custom products."
            Good: title "Non-custom products" -> "Off-the-shelf CPUs/GPUs recognized
            as revenue on delivery and transfer of control (ASC 606)."
        - `summary`: ONLY for substantial sections (more than roughly
          {config.summary_min_tokens} tokens, or {config.summary_min_tokens * 4} words):
          2-3 plain-text sentences, at most {config.max_summary_words} words. Leave null
          otherwise.

        HARD RULES:
        - Return exactly one ``sections`` entry per listed heading. Do not omit, merge,
          reorder, rename, or invent headings.
        - Never include a section's body text in your answer (token cost). Output only
          the title, level, description and (when warranted) summary for each section,
          plus the two document-level fields.
        - Also return `document_description` (one sentence, at most
          {config.max_description_words} words) and `document_summary` (2-4 sentences,
          at most {config.max_summary_words} words) describing the whole document.
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
    """The LLM call boundary — isolated so tests can substitute a fake implementation.

    The detected headings block is computed from ``headings`` or ``raw`` (the LLM input text)
    and passed as the ``{headings}`` template variable; this keeps the call
    signature stable for fakes while still giving the model the heading list.
    """
    from genai_tk.core.factories.llm_factory import get_llm
    from genai_tk.core.prompts import def_prompt

    target_headings = headings if headings is not None else detect_headings(raw)
    system, user = _build_prompt(filename=filename, raw=raw, config=config)
    headings_block = _render_headings_block(target_headings)
    prompt = def_prompt(system=system, user=user)
    llm_kwargs = {"max_tokens": max_tokens} if max_tokens is not None else {}
    structured_llm = get_llm(llm_id, **llm_kwargs).with_structured_output(DocumentOutline)
    # Fill {filename}/{raw}/{headings} as template variables (not a literal template
    # string) so braces in the source Markdown (e.g. LaTeX superscripts `^{(1)}`)
    # are not parsed as variables.
    result = (prompt | structured_llm).invoke({"filename": filename, "raw": raw, "headings": headings_block})
    assert isinstance(result, DocumentOutline)
    return result


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
                msg = (
                    f"{context}: hit the completion token limit (likely a reasoning model spending its budget on "
                    f"hidden reasoning tokens, not the input context window); retrying with max_tokens={max_tokens}."
                )
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
            # No LLM call was made this invocation; the stored llm_calls reflects the
            # original extraction and is reset so callers' totals count fresh calls only.
            return cached.model_copy(update={"llm_calls": 0})

    strategy = config.structure_strategy
    algo_headings: list[tuple[str, int, int]] = []
    preamble_toc_used = False

    # 1. Determine structure according to configured strategy
    if strategy == "algo":
        algo_headings = detect_headings(md_text)
    elif strategy == "toc_preamble":
        anchored, _ = extract_toc_from_preamble(md_text, filename, config, warnings=warnings)
        algo_headings = anchored or detect_headings(md_text)
        preamble_toc_used = bool(anchored)
    elif strategy == "llm_full":
        algo_headings = detect_headings(md_text)
    elif strategy == "auto":
        raw_headings = detect_headings(md_text)
        toc_text, _s, _e = _extract_toc_excerpt(md_text)
        # If a printed TOC exists and there are few native markdown headings (<10), extract via preamble
        if toc_text is not None and len([h for h in raw_headings if h[1] > 0]) < 10:
            anchored, _ = extract_toc_from_preamble(md_text, filename, config, warnings=warnings)
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

    outline = _call_llm_with_retry(
        llm_id=llm_id,
        filename=filename,
        raw=cleaned,
        config=config,
        warnings=warnings,
        headings=algo_headings,
    )
    if outline is None:
        result = OutlineResult(
            outline=None,
            degraded=True,
            reason="llm_call_failed",
            llm_calls=(1 if preamble_toc_used else 0),
        )
    else:
        # Anchor the LLM's entries onto the detected headings: the
        # heading's verbatim title and level are authoritative (the LLM only
        # supplies description/summary), so the cached outline matches the
        # structure the downstream merge slices on (one entry per heading).
        aligned = _align_outline(outline, algo_headings)
        result = OutlineResult(outline=aligned, llm_calls=(2 if preamble_toc_used else 1))
    if cache_path is not None:
        _write_cached(cache_path, result)
    return result
