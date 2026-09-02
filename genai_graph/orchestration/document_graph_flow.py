"""Prefect flow + workflow-engine step for building a Document Graph.

Wraps `genai_graph.kg.document_graph.ingest.ingest_document_graph` so it can be
referenced by dotted path from a genai-tk workflow YAML (`run:` /
`uses:`), exactly like `markdownize_flow` or `kg_create_step`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from genai_tk.workflow.registry import workflow
from loguru import logger
from prefect import flow

if TYPE_CHECKING:
    from collections.abc import Callable


def _resolve_build_llm(llm: str | None) -> str | None:
    """Resolve a `--llm` value to a concrete LLM id, or None for the algo path.

    A value with `@` is a literal id (``name@provider``); any other non-empty
    value is treated as a config tag resolved via ``kg_build.llms.<tag>``
    (e.g. ``default``, ``flash``), falling back to ``kg_build.llms.default``.
    """
    if llm is None:
        return None
    if "@" in llm:
        return llm
    from genai_tk.config_mgmt.config_mngr import global_config

    cfg = global_config()
    resolved = cfg.get_str(f"kg_build.llms.{llm}", default=None)
    if resolved:
        return resolved
    return cfg.get_str("kg_build.llms.default", default=None)


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
) -> dict[str, Any]:
    """Build (or update) a Document Graph at *db_path*.

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

    Returns:
        Dict with `db_path`, `documents_processed`, `documents_skipped`,
        `documents_failed`, `sections_created`, `sections_summarized`,
        `relationships_created`, `files_degraded`, `warnings`.
    """
    from genai_tk.workflow.force import ForceStage, stage_active

    from genai_graph.kg.backend import KuzuBackend
    from genai_graph.kg.document_graph.ingest import drop_document_graph, ingest_document_graph
    from genai_graph.kg.document_graph.outline_extract import OutlineConfig
    from genai_graph.kg.factories.document_graph_factory import DocumentGraphFactory

    backend = KuzuBackend()
    backend.connect(db_path)
    try:
        if delete_first:
            logger.info("Dropping existing Document Graph tables at {}", db_path)
            drop_document_graph(backend)

        resolved_llm = _resolve_build_llm(llm)
        outline_config: OutlineConfig | None = None
        if resolved_llm is not None or structure_strategy != "algo":
            cache_root = outline_cache_dir or str(Path(db_path).with_suffix("")) + "_outlines"
            outline_config = OutlineConfig(
                llm=resolved_llm,
                structure_strategy=structure_strategy,
                generate_summaries=generate_summaries,
                llm_max_tokens=llm_max_tokens,
                summary_min_tokens=summary_min_tokens,
                cache_root=cache_root,
                context_safety_ratio=context_safety_ratio,
            )

        factory = DocumentGraphFactory(
            sources=sources,
            include=include or ["*.md"],
            exclude=exclude or [],
            recursive=recursive,
            outline_config=outline_config,
        )

        # The pre-pass warms the content-addressed outline cache in parallel (no DB),
        # so the subsequent ingest reads each outline from disk without an LLM call.
        files_degraded = 0
        outline_warnings: list[str] = []
        if outline_config is not None:
            stats = factory.extract_outlines(workers=workers)
            files_degraded = stats.degraded_count
            outline_warnings = list(stats.warnings)

        # Dropping the Section tables leaves the Document nodes behind, so sections must be
        # rebuilt for them — otherwise the hash-based skip check makes the reset a no-op.
        force = delete_first or stage_active(force_stage, ForceStage.graph)
        result = ingest_document_graph(backend, factory, force=force)

        return {
            "db_path": db_path,
            "documents_processed": result.documents_processed,
            "documents_skipped": result.documents_skipped,
            "documents_failed": result.documents_failed,
            "sections_created": result.sections_created,
            "sections_summarized": result.sections_summarized,
            "relationships_created": result.relationships_created,
            "files_degraded": files_degraded,
            "warnings": [*outline_warnings, *result.warnings],
        }
    finally:
        backend.close()


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
