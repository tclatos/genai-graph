"""Prefect flow + workflow-engine step for summarizing a Document Graph.

Wraps `genai_graph.kg.document_graph.summarize.summarize_graph` so it can be
referenced by dotted path from a genai-tk workflow YAML (`run:` / `uses:`),
exactly like `document_graph_flow` or `kg_create_step`.

Concurrency model
-----------------
Documents are summarized concurrently across ``workers`` threads that share a
single Ladybug ``Database`` (each thread its own ``Connection``) — see
``genai_graph.kg.parallel.SharedKuzuParallel``. Ladybug is embedded: only one
read-write ``Database`` may exist per file in a process, but multiple
``Connection``s from that one ``Database`` issue concurrent read and write
transactions safely as long as they touch disjoint rows, which per-document
summarization does (each ``content_hash`` is summarized exactly once). This
stays a single Prefect task with internal thread parallelism, not a fan-out of
per-document Prefect tasks (those would each open their own ``Database`` and
race). The Ladybug REST API server is only needed for multi-*process* access,
which is not the case here.
"""

from __future__ import annotations

from typing import Any

from genai_tk.workflow.registry import workflow
from prefect import flow, task


@task(log_prints=False)
def _summarize_graph_task(
    db_path: str,
    *,
    folder_id: str | None,
    llm: str | None,
    max_level: int,
    summary_min_tokens: int,
    llm_max_tokens: int | None,
    force: bool,
    dry_run: bool,
    workers: int,
) -> dict[str, Any]:
    from genai_graph.kg.document_graph.summarize import SummarizationConfig, summarize_graph

    config = SummarizationConfig(
        llm=llm, max_level=max_level, summary_min_tokens=summary_min_tokens, llm_max_tokens=llm_max_tokens
    )
    result = summarize_graph(db_path, config, folder_id=folder_id, force=force, dry_run=dry_run, workers=workers)
    return result.model_dump()


@flow(name="document_graph_summarize")
def summarize_document_graph_flow(
    db_path: str,
    *,
    folder_id: str | None = None,
    llm: str | None = None,
    max_level: int = 6,
    summary_min_tokens: int = 800,
    llm_max_tokens: int | None = None,
    force: bool = False,
    dry_run: bool = False,
    workers: int = 4,
) -> dict[str, Any]:
    """Summarize every ingested document's sections (and whole-document abstract).

    Args:
        db_path: Path to the (already built) Ladybug Document Graph database.
        folder_id: When given, only summarize documents under this folder's subtree.
        llm: LLM id (`name@provider`) or config tag. Uses `kg_build.llms.default` when omitted.
        max_level: Deepest heading level that gets a description.
        summary_min_tokens: Sections at or above this token count also get a paragraph summary.
        llm_max_tokens: Explicit max output tokens for the LLM call. Raise this if you see
            'length limit reached' errors — a reasoning model spent its whole completion
            budget on hidden reasoning tokens, not the input context window.
        force: Re-summarize documents that already have a `summary`.
        dry_run: Compute the plan (selection, batching, warnings) without calling the
            LLM or writing to the graph.
        workers: Number of documents summarized in parallel (shared-DB thread pool).

    Returns:
        Dict with `db_path`, `documents_processed`, `documents_skipped`,
        `documents_failed`, `total_llm_calls`, `warnings`.
    """
    result = _summarize_graph_task(
        db_path,
        folder_id=folder_id,
        llm=llm,
        max_level=max_level,
        summary_min_tokens=summary_min_tokens,
        llm_max_tokens=llm_max_tokens,
        force=force,
        dry_run=dry_run,
        workers=workers,
    )
    return {"db_path": db_path, **result}


@workflow(name="document_graph_summarize", description="Summarize sections and documents in a Document Graph")
def document_graph_summarize_step(
    *,
    db_path: str,
    folder_id: str | None = None,
    llm: str | None = None,
    max_level: int = 6,
    summary_min_tokens: int = 800,
    llm_max_tokens: int | None = None,
    force: bool = False,
    dry_run: bool = False,
    workers: int = 4,
) -> dict[str, Any]:
    """Workflow-engine wrapper around `summarize_document_graph_flow` (see its docstring)."""
    return summarize_document_graph_flow(
        db_path=db_path,
        folder_id=folder_id,
        llm=llm,
        max_level=max_level,
        summary_min_tokens=summary_min_tokens,
        llm_max_tokens=llm_max_tokens,
        force=force,
        dry_run=dry_run,
        workers=workers,
    )
