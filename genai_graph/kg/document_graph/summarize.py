"""LLM-based summarization for the Document Graph.

Gives every `MarkdownSection` a one-sentence `description` (and substantial ones
a short `summary`), plus a `description`/`summary` for the owning `Document`, so
an agent can navigate a folder or document via `get_folder_toc`/`get_document_toc`
and open only the sections it actually needs.

Two fields, two jobs
--------------------
- `description` — one plain-text sentence, always present. This is the *routing*
  signal an agent scans to pick a section; it must stay short or the table of
  contents stops fitting in a prompt.
- `summary` — a short paragraph, only for sections at or above
  `summary_min_tokens`. This is the *triage* signal ("is it worth opening?").

Call strategy
-------------
The document is sent once, annotated with `[[section_id]]` markers before each
heading, and the model returns one entry per requested id. Sending the annotated
document instead of the full text *plus* a copy of every section's text (the
obvious construction) roughly halves input tokens and removes any ambiguity about
which text belongs to which id. Only when the *projected* output would exceed
`output_budget_tokens` is the id list split across several calls.

Brevity is enforced three times over, because models routinely ignore a stated
word limit: in the prompt, in the response-schema field descriptions, and finally
by `_clean_text()`, which strips Markdown and hard-truncates at a sentence
boundary.
"""

from __future__ import annotations

import re
import time

from genai_tk.utils.tokens import count_tokens
from loguru import logger
from pydantic import BaseModel, Field

from genai_graph.kg.backend import KgBackend
from genai_graph.kg.parallel import SharedKuzuParallel
from genai_graph.kg.query.document_graph_tools import (
    apply_document_summary,
    apply_section_summaries,
    get_document,
    get_document_toc,
    get_section_content,
    list_documents,
    render_toc_outline,
)

_DEFAULT_LLM_TAG = "default"

# A Markdown table row (possibly the `---` header separator row).
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")

# Leading Markdown noise to strip from a generated description/summary: heading
# hashes, list bullets, blockquote markers, and emphasis characters.
_MD_PREFIX_RE = re.compile(r"^\s*(?:#{1,6}\s*|[-*+]\s+|>\s*|\d+[.)]\s+)")
_MD_EMPHASIS_RE = re.compile(r"[*_`]{1,3}")
# "Page 12" conversion artifacts that leak in from PDF/Office → Markdown.
_PAGE_MARKER_RE = re.compile(r"(?im)^\s*#*\s*page\s+\d+\s*$")


class SummarizationConfig(BaseModel):
    """Policy and LLM settings for Document Graph summarization."""

    llm: str | None = Field(default=None, description="LLM id (name@provider) or tag; None uses kg_build.llms.default")
    max_level: int = Field(default=6, description="Deepest heading level that gets a description")
    summary_min_tokens: int = Field(
        default=800, description="Sections at or above this token count also get a paragraph `summary`"
    )
    max_description_words: int = Field(default=20, description="Target length of a section/document description")
    max_summary_words: int = Field(default=60, description="Target length of a section/document summary")
    max_description_chars: int = Field(default=180, description="Hard cap applied to a description after cleaning")
    max_summary_chars: int = Field(default=500, description="Hard cap applied to a summary after cleaning")
    output_budget_tokens: int = Field(
        default=4000, description="Split into multiple LLM calls once projected output exceeds this"
    )
    context_safety_ratio: float = Field(
        default=0.7,
        description="Warn when the document's token count exceeds this fraction of the model's context window",
    )
    table_sample_rows: int = Field(default=5, description="Data rows kept from a Markdown table when truncating")
    truncate_chars_per_section: int = Field(
        default=4000, description="Max characters of a single section's text kept in the LLM input"
    )
    llm_max_tokens: int | None = Field(
        default=None,
        description=(
            "Explicit max output tokens for the summarization call. Reasoning models can spend their whole "
            "completion budget on hidden reasoning tokens, leaving none for the answer ('length limit reached' "
            "errors) — set this (e.g. 32000) if that happens with your model. None uses the provider default."
        ),
    )
    retry_max_tokens: int = Field(
        default=32_000,
        description="max_tokens used for the one automatic retry after a 'length limit reached' failure",
    )


class SectionPlan(BaseModel):
    """One section selected for description, and whether it also warrants a summary."""

    section_id: str
    needs_summary: bool = False


class SectionEntry(BaseModel):
    """One section's generated description (and optional summary)."""

    section_id: str = Field(..., description="The section id exactly as given in the request")
    description: str = Field(
        ...,
        description="ONE plain-text sentence, at most 20 words, saying what this section contains. "
        "No Markdown, no bullet points, no headings, no line breaks.",
    )
    summary: str | None = Field(
        default=None,
        description="Only for sections marked 'summary needed': 2-3 plain-text sentences, at most 60 words.",
    )


class DocumentIndex(BaseModel):
    """Structured-output schema for one summarization LLM call."""

    document_description: str = Field(
        ..., description="ONE plain-text sentence, at most 20 words, saying what the whole document is."
    )
    document_summary: str = Field(
        ..., description="2-4 plain-text sentences, at most 60 words, abstracting the whole document."
    )
    sections: list[SectionEntry] = Field(..., description="Exactly one entry per requested section id")


class SummarizeDocumentResult(BaseModel):
    """Outcome of summarizing one document."""

    document_id: str
    sections_described: int = 0
    sections_summarized: int = 0
    already_summarized: bool = False
    llm_calls: int = 0
    warnings: list[str] = Field(default_factory=list)


class SummarizeGraphResult(BaseModel):
    """Outcome of summarizing every document in a graph (or folder subtree)."""

    documents_processed: int = 0
    documents_skipped: int = 0
    documents_failed: int = 0
    total_llm_calls: int = 0
    warnings: list[str] = Field(default_factory=list)


def select_sections(toc_rows: list[dict], *, max_level: int, summary_min_tokens: int) -> list[SectionPlan]:
    """Pick the sections to describe, flagging the substantial ones for a fuller summary.

    Every heading section within `max_level` gets a description — a table of contents
    with holes in it is not navigable. The synthetic level-0 root section is skipped:
    it is a container for preamble text, not a heading an agent would navigate to.
    """
    plans: list[SectionPlan] = []
    for row in toc_rows:
        level = int(row["level"])
        if level == 0 or level > max_level:
            continue
        token_count = int(row.get("token_count") or 0)
        plans.append(SectionPlan(section_id=row["section_id"], needs_summary=token_count >= summary_min_tokens))
    return plans


def _clean_text(text: str, max_chars: int) -> str:
    """Strip Markdown noise from generated text and hard-truncate at a sentence boundary.

    The backstop for models that ignore the stated word limit or answer in Markdown
    despite being told not to.
    """
    cleaned = _PAGE_MARKER_RE.sub("", text or "")
    lines = [_MD_PREFIX_RE.sub("", line).strip() for line in cleaned.splitlines()]
    flat = " ".join(line for line in lines if line)
    flat = _MD_EMPHASIS_RE.sub("", flat)
    flat = re.sub(r"\s+", " ", flat).strip()
    if len(flat) <= max_chars:
        return flat

    truncated = flat[:max_chars]
    for boundary in (". ", "; ", ", "):
        cut = truncated.rfind(boundary)
        if cut > max_chars // 2:
            return truncated[: cut + 1].strip()
    cut = truncated.rfind(" ")
    return (truncated[:cut] if cut > 0 else truncated).strip() + "…"


def _truncate_section_text(text: str, max_chars: int, table_sample_rows: int) -> str:
    """Shorten *text* for an LLM prompt: keep the heading and intro prose in full, but
    for a Markdown table keep only its header row plus the first `table_sample_rows`
    data rows (with a note of how many were omitted) — enough for the LLM to describe
    what the table covers without paying for every row.
    """
    lines = text.splitlines()
    out: list[str] = []
    total_chars = 0
    in_table = False
    header_done = False
    data_rows_kept = 0
    omitted = 0

    def flush_omitted() -> None:
        nonlocal omitted
        if omitted:
            out.append(f"_(... {omitted} more row(s) omitted ...)_")
            omitted = 0

    for line in lines:
        stripped = line.strip()
        if _TABLE_ROW_RE.match(stripped):
            if not in_table:
                in_table, header_done, data_rows_kept = True, False, 0
            if not header_done:
                out.append(line)
                total_chars += len(line)
                if _TABLE_SEPARATOR_RE.match(stripped) and set(stripped.replace("|", "")) <= set(" :-"):
                    header_done = True
                continue
            if data_rows_kept < table_sample_rows:
                out.append(line)
                total_chars += len(line)
                data_rows_kept += 1
            else:
                omitted += 1
            continue

        if in_table:
            flush_omitted()
            in_table = False

        out.append(line)
        total_chars += len(line)
        if total_chars > max_chars:
            out.append("_(... truncated ...)_")
            break

    flush_omitted()
    return "\n".join(out)


def _build_annotated_document(section_rows: list[dict], config: SummarizationConfig) -> str:
    """Rebuild the document with an `[[section_id]]` marker before each section.

    Sections partition the document without overlap, so concatenating them in
    `sequence` order reproduces it. Emitting the markers inline lets one prompt carry
    both the full context and an unambiguous id → text mapping, instead of sending the
    document and then a second copy of every section's text.
    """
    parts = []
    for row in sorted(section_rows, key=lambda r: r["sequence"]):
        text = _truncate_section_text(row["text"], config.truncate_chars_per_section, config.table_sample_rows)
        parts.append(f"[[{row['section_id']}]]\n{text}")
    return "\n\n".join(parts)


def _plan_batches(plans: list[SectionPlan], config: SummarizationConfig) -> list[list[SectionPlan]]:
    """Group section plans into batches whose *projected* output stays under
    `output_budget_tokens`. Most documents fit in a single batch (one LLM call).
    """
    if not plans:
        return []
    tokens_per_word = 1.4  # provider-agnostic rule of thumb
    batches: list[list[SectionPlan]] = []
    current: list[SectionPlan] = []
    current_tokens = 0.0
    for plan in plans:
        words = config.max_description_words + (config.max_summary_words if plan.needs_summary else 0)
        cost = max(words * tokens_per_word, 1.0)
        if current and current_tokens + cost > config.output_budget_tokens:
            batches.append(current)
            current, current_tokens = [], 0.0
        current.append(plan)
        current_tokens += cost
    if current:
        batches.append(current)
    return batches


def _resolve_llm_id(config: SummarizationConfig) -> str:
    if config.llm:
        return config.llm
    from genai_tk.config_mgmt.config_mngr import global_config

    return global_config().get_str("kg_build.llms.default", default=_DEFAULT_LLM_TAG) or _DEFAULT_LLM_TAG


def _context_window_for(llm_id: str) -> int | None:
    from genai_tk.core.factories.llm_factory import get_llm_info

    try:
        return get_llm_info(llm_id).effective_context_window
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not resolve context window for {}: {}", llm_id, exc)
        return None


def _is_length_limit_error(exc: Exception) -> bool:
    """True when *exc* is the OpenAI SDK's `LengthFinishReasonError`.

    Raised when the model hits `finish_reason="length"` before producing a parseable
    structured-output answer — typically a reasoning model spending its entire
    completion budget on hidden reasoning tokens, leaving none for the answer itself.
    This is an *output*-token problem, unrelated to the input context window.
    """
    try:
        from openai import LengthFinishReasonError

        if isinstance(exc, LengthFinishReasonError):
            return True
    except ImportError:
        pass
    return "length limit was reached" in str(exc)


def _build_prompt(
    *,
    filename: str,
    toc_outline: str,
    annotated_document: str,
    plans: list[SectionPlan],
    config: SummarizationConfig,
) -> tuple[str, str]:
    system = f"""
        You write the table of contents for a document library that an AI agent reads to
        decide which section to open. The agent sees ONLY your descriptions, never the text,
        so each one must say concretely what its section CONTAINS.

        Rules for every `description`:
        - Exactly ONE sentence, at most {config.max_description_words} words.
        - Plain text only: no Markdown, no headings, no bullet points, no line breaks.
        - Do not restate the section title, and do not open with "This section" or "Describes".
        - Name the concrete subject matter: obligations, parties, systems, figures, dates.

        Rules for `summary` (ONLY for sections marked "summary needed"; otherwise leave it null):
        - 2-3 plain-text sentences, at most {config.max_summary_words} words.

        If a section is mostly a table, describe what the table covers, its columns and how
        many rows it has. Never reproduce table rows.

        Good description: "Supplier liability caps, indemnities and the exclusions applying to
        indirect damages."
        Bad description: "## 8 Liability\\n\\nThis section describes liability. - Cap: ..."
    """
    requested = "\n".join(f"- {plan.section_id}{' (summary needed)' if plan.needs_summary else ''}" for plan in plans)
    user = f"""
        Document: {filename}

        Outline:
        {toc_outline}

        Full document below, annotated with [[section_id]] markers. The text following each
        marker, up to the next marker, is that section's own content.
        ---
        {annotated_document}
        ---

        Return exactly one entry per requested section id, using the id verbatim:
        {requested}

        Also return `document_description` (one sentence) and `document_summary` for the
        whole document.
    """
    return system, user


def _call_llm(
    *,
    llm_id: str,
    filename: str,
    toc_outline: str,
    annotated_document: str,
    plans: list[SectionPlan],
    config: SummarizationConfig,
    max_tokens: int | None = None,
) -> DocumentIndex:
    """The LLM call boundary — isolated so tests can substitute a fake implementation."""
    from genai_tk.core.factories.llm_factory import get_llm

    system, user = _build_prompt(
        filename=filename,
        toc_outline=toc_outline,
        annotated_document=annotated_document,
        plans=plans,
        config=config,
    )
    llm_kwargs = {"max_tokens": max_tokens} if max_tokens is not None else {}
    structured_llm = get_llm(llm_id, **llm_kwargs).with_structured_output(DocumentIndex)
    result = structured_llm.invoke([("system", system), ("user", user)])
    assert isinstance(result, DocumentIndex)
    return result


def _call_llm_with_retry(
    *,
    llm_id: str,
    filename: str,
    toc_outline: str,
    annotated_document: str,
    plans: list[SectionPlan],
    config: SummarizationConfig,
    max_tokens: int | None,
    context: str,
    warnings: list[str],
) -> DocumentIndex | None:
    """Call the LLM, retrying once with a larger completion budget on a length-limit failure.

    Any other exception is recorded as a warning and gives up immediately (no retry) —
    only the reasoning-budget-exhaustion failure is worth a second attempt. Every outcome
    (retry, final failure) is also logged immediately via loguru, not just returned in
    `warnings` — a multi-minute run must not go silent while this is happening.
    """
    for attempt in range(2):
        started = time.monotonic()
        try:
            llm_result = _call_llm(
                llm_id=llm_id,
                filename=filename,
                toc_outline=toc_outline,
                annotated_document=annotated_document,
                plans=plans,
                config=config,
                max_tokens=max_tokens,
            )
            if attempt > 0:
                logger.info("{}: retry succeeded ({:.1f}s)", context, time.monotonic() - started)
            return llm_result
        except Exception as exc:  # noqa: BLE001
            if attempt == 0 and _is_length_limit_error(exc):
                max_tokens = max(max_tokens or 0, config.retry_max_tokens)
                msg = (
                    f"{context}: hit the completion token limit (likely a reasoning model spending its budget "
                    f"on hidden reasoning tokens, not the input context window); retrying with max_tokens={max_tokens}."
                )
                warnings.append(msg)
                logger.warning(msg)
                continue
            msg = f"LLM call failed for {context}: {exc}"
            warnings.append(msg)
            logger.error(msg)
            return None
    return None


def summarize_document(
    backend: KgBackend,
    document_id: str,
    config: SummarizationConfig | None = None,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> SummarizeDocumentResult:
    """Summarize one document's sections (and its whole-document abstract).

    Args:
        backend: Connected `KgBackend`.
        document_id: Content hash (full or prefix), `markdown_hash`, filename, or path.
        config: Summarization policy and LLM settings. Defaults applied if omitted.
        force: Re-summarize even if the document already has a `summary`.
        dry_run: Compute the plan (selection, batching, warnings) without calling the
            LLM or writing to the graph.

    Returns:
        `SummarizeDocumentResult` with counts and any warnings.
    """
    config = config or SummarizationConfig()
    result = SummarizeDocumentResult(document_id=document_id)

    doc = get_document(backend, document_id)
    if doc is None:
        result.warnings.append(f"Document not found: {document_id}")
        return result

    markdown_hash = doc["markdown_hash"]
    toc_rows = get_document_toc(backend, markdown_hash)
    if not toc_rows:
        result.warnings.append(f"No sections found for document: {document_id}")
        return result

    plans = select_sections(toc_rows, max_level=config.max_level, summary_min_tokens=config.summary_min_tokens)
    if not plans:
        result.warnings.append(f"No describable sections in: {document_id}")
        return result

    if not force and doc.get("description") and all(r.get("description") for r in toc_rows if int(r["level"]) > 0):  # type: ignore[union-attr]
        result.already_summarized = True
        return result

    needs_summary = sum(1 for p in plans if p.needs_summary)
    logger.info("{}: {} section(s) to describe ({} also get a summary)", doc["filename"], len(plans), needs_summary)

    llm_id = _resolve_llm_id(config)
    section_rows = get_section_content(backend, [r["section_id"] for r in toc_rows])  # type: ignore[arg-type]
    annotated = _build_annotated_document(section_rows, config)  # type: ignore[arg-type]
    doc_tokens = count_tokens(annotated)
    context_window = _context_window_for(llm_id)
    if context_window and doc_tokens > context_window * config.context_safety_ratio:
        msg = (
            f"Document has ~{doc_tokens} tokens, over {config.context_safety_ratio:.0%} of "
            f"{llm_id}'s {context_window}-token context window; consider a larger-context model or "
            "a lower truncate_chars_per_section."
        )
        result.warnings.append(msg)
        logger.warning(msg)

    toc_outline = render_toc_outline(toc_rows)  # type: ignore[arg-type]
    batches = _plan_batches(plans, config)

    if dry_run:
        result.sections_described = len(plans)
        result.sections_summarized = needs_summary
        result.llm_calls = len(batches)
        return result

    section_updates: list[dict] = []
    document_description: str | None = None
    document_summary: str | None = None

    for batch_num, batch in enumerate(batches, start=1):
        logger.info(
            "{}: calling LLM for batch {}/{} ({} section(s))...",
            doc["filename"],
            batch_num,
            len(batches),
            len(batch),
        )
        started = time.monotonic()
        llm_result = _call_llm_with_retry(
            llm_id=llm_id,
            filename=doc["filename"],
            toc_outline=toc_outline,
            annotated_document=annotated,
            plans=batch,
            config=config,
            max_tokens=config.llm_max_tokens,
            context=f"{doc['filename']} batch {batch_num}/{len(batches)} ({len(batch)} section(s))",
            warnings=result.warnings,
        )
        if llm_result is None:
            continue
        result.llm_calls += 1
        logger.info(
            "{}: batch {}/{} done ({:.1f}s)", doc["filename"], batch_num, len(batches), time.monotonic() - started
        )
        if document_description is None and llm_result.document_description:
            document_description = _clean_text(llm_result.document_description, config.max_description_chars)
        if document_summary is None and llm_result.document_summary:
            document_summary = _clean_text(llm_result.document_summary, config.max_summary_chars)

        wanted = {p.section_id: p for p in batch}
        returned_ids = {s.section_id for s in llm_result.sections}
        missing = set(wanted) - returned_ids
        if missing:
            result.warnings.append(f"LLM omitted {len(missing)} section description(s): {sorted(missing)}")
        for entry in llm_result.sections:
            plan = wanted.get(entry.section_id)
            if plan is None:
                continue
            summary = (
                _clean_text(entry.summary, config.max_summary_chars) if plan.needs_summary and entry.summary else None
            )
            section_updates.append(
                {
                    "section_id": entry.section_id,
                    "description": _clean_text(entry.description, config.max_description_chars),
                    "summary": summary,
                    "summary_source": "llm",
                }
            )

    result.sections_described = len(section_updates)
    result.sections_summarized = sum(1 for u in section_updates if u["summary"])

    apply_section_summaries(backend, section_updates)
    apply_document_summary(backend, markdown_hash, description=document_description, summary=document_summary)

    logger.info(
        "{}: done ({} described, {} summarized, {} LLM call(s))",
        doc["filename"],
        result.sections_described,
        result.sections_summarized,
        result.llm_calls,
    )
    return result


def _summarize_hashes(
    parallel: SharedKuzuParallel,
    target_hashes: list[str],
    hash_to_filename: dict[str, str],
    config: SummarizationConfig,
    *,
    force: bool,
    dry_run: bool,
    result: SummarizeGraphResult | None = None,
) -> SummarizeGraphResult:
    """Fan `summarize_document` out over `target_hashes` on the shared-DB worker pool.

    Each worker borrows a backend from *parallel* and summarizes one document, so
    the slow LLM calls overlap while the fast graph writes stay on disjoint rows
    (every `markdown_hash` is summarized exactly once). Per-document failures are
    returned by `SharedKuzuParallel.map` as `Exception` values and folded into
    `documents_failed`/`warnings`, never aborting the whole run.
    """
    result = result or SummarizeGraphResult()
    if not target_hashes:
        logger.info("Summarize done: nothing to do")
        return result

    def _worker(backend: KgBackend, markdown_hash: str) -> SummarizeDocumentResult:
        return summarize_document(backend, markdown_hash, config, force=force, dry_run=dry_run)

    outcomes = parallel.map(_worker, target_hashes)
    for markdown_hash, outcome in zip(target_hashes, outcomes, strict=True):
        filename = hash_to_filename.get(markdown_hash, markdown_hash)
        if isinstance(outcome, Exception):
            result.documents_failed += 1
            result.warnings.append(f"Failed to summarize {filename}: {outcome}")
            logger.error("Failed to summarize {}: {}", filename, outcome)
            continue
        if outcome.already_summarized:
            result.documents_skipped += 1
            logger.info("{}: already summarized, skipping", filename)
        else:
            result.documents_processed += 1
        result.total_llm_calls += outcome.llm_calls
        result.warnings.extend(f"{filename}: {w}" for w in outcome.warnings)

    logger.info(
        "Summarize done: {} processed, {} skipped, {} failed, {} LLM call(s) total",
        result.documents_processed,
        result.documents_skipped,
        result.documents_failed,
        result.total_llm_calls,
    )
    return result


def summarize_graph(
    db_path: str,
    config: SummarizationConfig | None = None,
    *,
    folder_id: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    workers: int = 1,
) -> SummarizeGraphResult:
    """Summarize every ingested document (optionally scoped to one folder's subtree).

    Documents are summarized concurrently across `workers` threads that share one
    Ladybug ``Database`` (each with its own ``Connection``). Ladybug only allows a
    single read-write ``Database`` per file in a process, so the fan-out shares it
    rather than opening one per worker; concurrent writes require
    ``enable_multi_writes=True`` (set by ``SharedKuzuParallel``) and must touch
    disjoint rows, which is why each ``markdown_hash`` is summarized exactly once.

    Args:
        db_path: Path to the (already built) Ladybug Document Graph database.
        config: Summarization policy and LLM settings. Defaults applied if omitted.
        folder_id: When given, only summarize documents under this folder's subtree.
        force: Re-summarize documents that already have a `summary`.
        dry_run: Plan only — no LLM calls, no writes.
        workers: Number of documents summarized in parallel (>= 1).

    Returns:
        `SummarizeGraphResult` aggregating per-document outcomes.
    """
    config = config or SummarizationConfig()
    with SharedKuzuParallel(db_path, max_workers=workers) as parallel:
        docs = list_documents(parallel.primary, folder_id=folder_id)
        # Dedupe by markdown_hash: documents sharing content share the same
        # MarkdownSection/Document rows, so summarizing them concurrently would
        # write the same rows from two transactions and conflict.
        hash_to_filename: dict[str, str] = {}
        target_hashes: list[str] = []
        for doc in docs:
            markdown_hash = doc["markdown_hash"]
            if markdown_hash in hash_to_filename:
                continue
            hash_to_filename[markdown_hash] = doc["filename"]
            target_hashes.append(markdown_hash)
        scope = f" under folder {folder_id}" if folder_id else ""
        logger.info("Summarizing {} document(s){}...", len(target_hashes), scope)
        return _summarize_hashes(parallel, target_hashes, hash_to_filename, config, force=force, dry_run=dry_run)


def summarize_documents(
    db_path: str,
    document_ids: list[str],
    config: SummarizationConfig | None = None,
    *,
    force: bool = False,
    dry_run: bool = False,
    workers: int = 1,
) -> SummarizeGraphResult:
    """Summarize an explicit list of documents (by hash, prefix, filename, or path).

    Each reference is resolved to its `markdown_hash` on the shared-DB primary
    connection, references are de-duplicated by `markdown_hash` (same content = same
    rows; summarizing it twice concurrently would conflict), and the resolved
    documents are summarized concurrently across `workers` threads. Unresolved
    references are recorded as failures with a warning rather than raising.

    Args:
        db_path: Path to the (already built) Ladybug Document Graph database.
        document_ids: Document references (content/markdown hash or prefix, filename, path).
        config: Summarization policy and LLM settings. Defaults applied if omitted.
        force: Re-summarize documents that already have a `summary`.
        dry_run: Plan only — no LLM calls, no writes.
        workers: Number of documents summarized in parallel (>= 1).

    Returns:
        `SummarizeGraphResult` aggregating per-document outcomes.
    """
    config = config or SummarizationConfig()
    result = SummarizeGraphResult()
    with SharedKuzuParallel(db_path, max_workers=workers) as parallel:
        hash_to_filename: dict[str, str] = {}
        target_hashes: list[str] = []
        for ref in document_ids:
            doc = get_document(parallel.primary, ref)
            if doc is None:
                result.warnings.append(f"Document not found: {ref}")
                result.documents_failed += 1
                continue
            markdown_hash = doc["markdown_hash"]
            if markdown_hash in hash_to_filename:
                continue
            hash_to_filename[markdown_hash] = doc["filename"]
            target_hashes.append(markdown_hash)
        logger.info("Summarizing {} document(s)...", len(target_hashes))
        return _summarize_hashes(
            parallel, target_hashes, hash_to_filename, config, force=force, dry_run=dry_run, result=result
        )
