"""Prefect flow + workflow-engine step for building a Document Graph.

Thin wrappers over `genai_graph.kg.document_graph.build.build_document_graph`
(the single build baseline shared with `cli docgraph build` and the bench) so
they can be referenced by dotted path from a genai-tk workflow YAML (`run:` /
`uses:`), exactly like `markdownize_flow` or `kg_create_step`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from genai_tk.workflow.registry import workflow
from prefect import flow

if TYPE_CHECKING:
    from collections.abc import Callable


@flow(name="document_graph")
def document_graph_flow(
    sources: list[str],
    db_path: str,
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    recursive: bool = True,
    force_stage: str | None = None,
    delete_first: bool = False,
    llm: str | None = None,
    structure_strategy: str = "auto",
    generate_summaries: bool = True,
    llm_max_tokens: int | None = None,
    summary_min_tokens: int = 800,
    outline_cache_dir: str | None = None,
    workers: int = 4,
    context_safety_ratio: float = 0.9,
    embeddings_id: str | None = None,
    fts: bool = True,
    chunk_size_tokens: int = 1500,
    embed_workers: int | None = None,
) -> dict[str, Any]:
    """Build (or update) a Document Graph at *db_path*.

    Thin wrapper over `genai_graph.kg.document_graph.build.build_document_graph`
    (see it for the pipeline and full parameter docs).

    Args:
        sources: Directories, files, or `.zip` archives to ingest.
        db_path: Path to the (shared) Ladybug database file.
        include: Glob patterns to include (default `["*.md"]`).
        exclude: Glob patterns to exclude.
        recursive: Recurse into sub-directories.
        force_stage: One of `graph`/`all` (see `genai_tk.workflow.force`).
            `graph` (and above) rebuilds sections for documents already in the
            graph (handles heading/line-number drift on file edits).
        delete_first: Drop the Section tables before ingesting (full reset of the
            document graph; the shared Document table is preserved). Implies
            `force_stage="graph"` — sections are rebuilt for every document.
        llm: LLM id (``name@provider``) or config tag (e.g. ``default``/``flash``)
            enabling the LLM build path: a flash model discovers each document's
            structure and summarizes its sections in one call. None (default) keeps
            the fast algorithmic-only path. See ``kg_build.llms.*`` config tags.
        structure_strategy: Decomposition strategy: 'auto' | 'algo' | 'toc_preamble' | 'llm_full'.
        generate_summaries: Whether to generate LLM section descriptions and summaries.
        llm_max_tokens: Explicit max output tokens for the outline call; raise for
            reasoning models that exhaust their completion budget.
        summary_min_tokens: Prompt guidance for what counts as a "substantial"
            section worth a fuller summary.
        outline_cache_dir: Directory for the content-addressed outline JSON cache.
            Defaults to ``<db_path stem>_outlines``.
        workers: Parallelism for the outline pre-pass (and LLM calls).
        context_safety_ratio: Degrade a document to algorithmic parsing (no LLM
            call, no summaries) when its token count exceeds this fraction of the
            model's context window.
        embeddings_id: Embeddings model for SectionChunk vectors (None disables).
        fts: Create the native BM25/FTS index over sections.
        chunk_size_tokens: Target chunk size for long sections.
        embed_workers: Parallel workers for per-document chunk-embedding batches
            during ingest; defaults to *workers*.

    Returns:
        Dict with `db_path`, `documents_processed`, `documents_skipped`,
        `documents_failed`, `sections_created`, `sections_summarized`,
        `relationships_created`, `files_degraded`, `warnings`.
    """
    from genai_tk.workflow.force import ForceStage, stage_active

    from genai_graph.kg.document_graph.build import build_document_graph

    # Dropping the Section tables leaves the Document nodes behind, so sections must be
    # rebuilt for them — otherwise the hash-based skip check makes the reset a no-op.
    force = delete_first or stage_active(force_stage, ForceStage.graph)
    stats = build_document_graph(
        sources=sources,
        db_path=db_path,
        include=include,
        exclude=exclude,
        recursive=recursive,
        delete_first=delete_first,
        force=force,
        llm=llm,
        llm_max_tokens=llm_max_tokens,
        structure_strategy=structure_strategy,
        generate_summaries=generate_summaries,
        summary_min_tokens=summary_min_tokens,
        outline_cache_dir=outline_cache_dir,
        workers=workers,
        context_safety_ratio=context_safety_ratio,
        embeddings_id=embeddings_id,
        fts=fts,
        chunk_size_tokens=chunk_size_tokens,
        embed_workers=embed_workers,
    )
    return {
        "db_path": db_path,
        "documents_processed": stats["documents_processed"],
        "documents_skipped": stats["documents_skipped"],
        "documents_failed": stats["documents_failed"],
        "sections_created": stats["sections_created"],
        "sections_summarized": stats["sections_summarized"],
        "relationships_created": stats["relationships_created"],
        "files_degraded": stats["files_degraded"],
        "warnings": stats["warnings"],
    }


@workflow(name="document_graph_build", description="Build a Document Graph from a corpus")
def document_graph_build_step(
    *,
    sources: list[str],
    db_path: str,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    recursive: bool = True,
    force_stage: str | None = None,
    delete_first: bool = False,
    llm: str | None = None,
    llm_max_tokens: int | None = None,
    summary_min_tokens: int = 800,
    outline_cache_dir: str | None = None,
    workers: int = 4,
    context_safety_ratio: float = 0.9,
) -> dict[str, Any]:
    """Workflow-engine wrapper around `document_graph_flow` (see its docstring)."""
    return document_graph_flow(
        sources=sources,
        db_path=db_path,
        include=include,
        exclude=exclude,
        recursive=recursive,
        force_stage=force_stage,
        delete_first=delete_first,
        llm=llm,
        llm_max_tokens=llm_max_tokens,
        summary_min_tokens=summary_min_tokens,
        outline_cache_dir=outline_cache_dir,
        workers=workers,
        context_safety_ratio=context_safety_ratio,
    )


def make_source_already_ingested(db_path: str) -> "Callable[[str], bool]":
    """Return a callback usable as ``markdownize_flow(already_processed=...)``.

    The returned callable takes a source file's content hash and reports whether
    a Document derived from it is already in the graph at *db_path* — letting the
    markdownize step skip re-converting files whose output is already stored,
    without genai-tk depending on genai-graph.
    """
    from genai_graph.kg.backend import KuzuBackend

    backend = KuzuBackend()
    backend.connect(db_path)

    def _already(source_hash: str) -> bool:
        try:
            df = backend.execute_get_as_df(
                "MATCH (d:Document {content_hash: $h}) RETURN d.content_hash AS h LIMIT 1",
                {"h": source_hash},
                union=False,
            )
        except Exception as exc:  # noqa: BLE001
            if "does not exist" in str(exc):
                return False
            raise
        return not df.empty

    return _already
