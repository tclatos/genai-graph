"""Single-baseline builder for the Document Graph.

`build_document_graph` is the one implementation of the build pipeline
(parallel outline pre-pass → parse → parallel chunk/embed → single-writer
merge). The CLI (`cli docgraph build`), the workflow engine
(`orchestration.document_graph_flow`, `docgraph_build_step`) and the benchmark
(`bench.build_graph`) are all thin wrappers over it, so speed and robustness
improvements land everywhere at once.

`warm_outline_cache` is the DB-free half of the pipeline: it only warms the
content-addressed outline cache, so callers with per-document parallelism
(e.g. Prefect tasks) can run it fan-out before the single-writer merge.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from loguru import logger

from genai_graph.kg.document_graph.outline_extract import OutlineConfig


def resolve_build_llm(llm: str | None) -> str | None:
    """Resolve a `--llm` value to a concrete LLM id, or None for the algo path.

    A value with `@` is a literal id (``name@provider``); any other non-empty
    value is treated as a config tag resolved via ``kg_build.llms.<tag>``
    (e.g. ``default``, ``flash``), falling back to ``kg_build.llms.default``.
    """
    if llm is None:
        return None
    if "@" in llm:
        return llm
    try:
        from genai_tk.config_mgmt.config_mngr import global_config

        cfg = global_config()
        resolved = cfg.get_str(f"kg_build.llms.{llm}", default=None)
        if resolved:
            return resolved
        return cfg.get_str("kg_build.llms.default", default=None)
    except Exception:
        return llm


def _build_outline_config(
    *,
    llm: str | None,
    llm_max_tokens: int | None,
    structure_strategy: str,
    generate_summaries: bool,
    workers: int,
    summary_min_tokens: int,
    context_safety_ratio: float,
    cache_root: str,
) -> OutlineConfig | None:
    """Build the outline policy, or None for the algorithmic-only path."""
    resolved_llm = resolve_build_llm(llm)
    if resolved_llm is None and structure_strategy == "algo":
        return None
    return OutlineConfig(
        llm=resolved_llm,
        structure_strategy=structure_strategy,
        generate_summaries=generate_summaries,
        llm_max_tokens=llm_max_tokens,
        workers=workers,
        summary_min_tokens=summary_min_tokens,
        context_safety_ratio=context_safety_ratio,
        cache_root=cache_root,
    )


def warm_outline_cache(
    sources: list[str],
    *,
    db_path: str | Path,
    outline_cache_dir: str | None = None,
    llm: str | None = None,
    llm_max_tokens: int | None = None,
    structure_strategy: str = "auto",
    generate_summaries: bool = True,
    workers: int = 4,
    summary_min_tokens: int = 800,
    context_safety_ratio: float = 0.9,
) -> dict[str, Any]:
    """Extract outlines for the given sources without touching the database.

    Idempotent: results land in the content-addressed outline cache, so a later
    ``build_document_graph(outline_pre_pass=False)`` reads them from disk with
    no LLM calls. No DB access and no shared state, so this is safe to run per
    document from parallel tasks (per-document fault isolation). A document
    whose LLM extraction ultimately fails here is simply missing from the cache
    and degrades to algorithmic parsing at merge time.

    Args:
        sources: Directories, files, or `.zip` archives to outline.
        db_path: Ladybug database path (only used to derive the cache root).
        outline_cache_dir: Directory for the outline JSON cache; defaults to
            ``<db_path stem>_outlines``.
        llm: Build LLM id (``name@provider`` or tag); None keeps the algorithmic path.
        llm_max_tokens: Explicit max output tokens for the outline call.
        structure_strategy: Outline strategy: 'auto' | 'algo' | 'toc_preamble' | 'llm_full'.
        generate_summaries: Generate LLM section descriptions & summaries.
        workers: Parallel LLM workers for cross-document pre-pass and per-document
            branch summarization.
        summary_min_tokens: Prompt heuristic for 'substantial' sections.
        context_safety_ratio: Degrade a document to algorithmic parsing when its
            token count exceeds this fraction of the model's context window.

    Returns:
        `OutlineStats` dict for the warmed sources, or a ``status: skipped``
        dict when there is no outline cache to warm (algorithmic-only build).
    """
    cache_root = outline_cache_dir or str(Path(db_path).with_suffix("")) + "_outlines"
    outline_config = _build_outline_config(
        llm=llm,
        llm_max_tokens=llm_max_tokens,
        structure_strategy=structure_strategy,
        generate_summaries=generate_summaries,
        workers=workers,
        summary_min_tokens=summary_min_tokens,
        context_safety_ratio=context_safety_ratio,
        cache_root=cache_root,
    )
    if outline_config is None:
        return {"status": "skipped", "reason": "algorithmic-only build (no outline cache)"}

    from genai_graph.kg.factories.document_graph_factory import DocumentGraphFactory

    factory = DocumentGraphFactory(sources=sources, recursive=True, outline_config=outline_config)
    return factory.extract_outlines(workers=workers).model_dump()


def build_document_graph(
    sources: list[str],
    db_path: str | Path,
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    recursive: bool = True,
    delete_first: bool = False,
    force: bool = False,
    llm: str | None = None,
    llm_max_tokens: int | None = None,
    structure_strategy: str = "auto",
    generate_summaries: bool = True,
    summary_min_tokens: int = 800,
    context_safety_ratio: float = 0.9,
    outline_cache_dir: str | None = None,
    workers: int = 4,
    embed_workers: int | None = None,
    outline_pre_pass: bool = True,
    embeddings_id: str | None = None,
    fts: bool = True,
    chunk_size_tokens: int = 1500,
) -> dict[str, Any]:
    """Build (or update) a Document Graph at *db_path* from Markdown *sources*.

    The outline pre-pass warms the content-addressed outline cache in parallel
    across documents (no DB), so the ingest reads each outline from disk without
    an LLM call. Chunk embeddings are computed per document in parallel on a
    single asyncio loop; the database is only written afterwards by one
    single-writer merge (Ladybug allows one read-write `Database` per file).

    Args:
        sources: Directories, files, or `.zip` archives to ingest.
        db_path: Path to the (shared) Ladybug database file.
        include: Glob patterns to include (default `["*.md"]`).
        exclude: Glob patterns to exclude.
        recursive: Recurse into sub-directories.
        delete_first: Drop the Document Graph section tables before ingesting
            (full reset; shared Folder/Document tables are preserved).
        force: Rebuild sections for documents already present in the graph.
        llm: Build LLM id (``name@provider`` or tag); None keeps the algorithmic path.
        llm_max_tokens: Explicit max output tokens for the outline call; raise
            for reasoning models that exhaust their completion budget.
        structure_strategy: Outline strategy: 'auto' | 'algo' | 'toc_preamble' | 'llm_full'.
        generate_summaries: Generate LLM section descriptions & summaries.
        summary_min_tokens: Prompt heuristic for 'substantial' sections.
        context_safety_ratio: Degrade a document to algorithmic parsing when its
            token count exceeds this fraction of the model's context window.
        outline_cache_dir: Directory for the content-addressed outline JSON cache;
            defaults to ``<db_path stem>_outlines``.
        workers: Parallel LLM workers for the outline pre-pass and per-document
            branch summarization.
        embed_workers: Parallel workers for per-document chunk-embedding batches
            during ingest; defaults to *workers*. 1 keeps embedding serial.
        outline_pre_pass: Warm the outline cache in parallel before ingesting
            (recommended; set False when the cache was already warmed by
            `warm_outline_cache`, e.g. per-document Prefect tasks).
        embeddings_id: Embeddings model for SectionChunk vectors (None disables).
        fts: Create the native BM25/FTS index over sections.
        chunk_size_tokens: Target chunk size for long sections.

    Returns:
        Ingest statistics dict with counts, `files_degraded`, per-stage
        ``timings`` (seconds), `db_path` and combined warnings.
    """
    from genai_graph.kg.backend import KuzuBackend
    from genai_graph.kg.document_graph.ingest import drop_document_graph, ingest_document_graph
    from genai_graph.kg.document_graph.retrieval import RetrievalConfig
    from genai_graph.kg.factories.document_graph_factory import DocumentGraphFactory

    db_p = Path(db_path)
    db_p.parent.mkdir(parents=True, exist_ok=True)
    cache_root = outline_cache_dir or str(db_p.with_suffix("")) + "_outlines"

    outline_config = _build_outline_config(
        llm=llm,
        llm_max_tokens=llm_max_tokens,
        structure_strategy=structure_strategy,
        generate_summaries=generate_summaries,
        workers=workers,
        summary_min_tokens=summary_min_tokens,
        context_safety_ratio=context_safety_ratio,
        cache_root=cache_root,
    )
    resolved_llm = outline_config.llm if outline_config else None

    factory = DocumentGraphFactory(
        sources=sources,
        include=include or ["*.md"],
        exclude=exclude or [],
        recursive=recursive,
        outline_config=outline_config,
    )

    timings: dict[str, float] = {}
    backend = KuzuBackend()
    backend.connect(str(db_p))
    try:
        if delete_first:
            logger.info("Dropping existing Document Graph tables at {}", db_p)
            drop_document_graph(backend)

        files_degraded = 0
        outline_warnings: list[str] = []
        if outline_config is not None and outline_pre_pass:
            t0 = time.monotonic()
            stats = factory.extract_outlines(workers=workers)
            files_degraded = stats.degraded_count
            outline_warnings = list(stats.warnings)
            timings["outline_pre_pass_s"] = round(time.monotonic() - t0, 3)
            logger.info(
                "Outline pre-pass: {} file(s), {} degraded, {} LLM call(s) in {:.1f}s",
                stats.total_files,
                stats.degraded_count,
                stats.llm_calls,
                timings["outline_pre_pass_s"],
            )

        logger.info(
            "Ingesting Document Graph from {} into {} (llm={}, embeddings={}, fts={})",
            sources,
            db_p,
            resolved_llm,
            embeddings_id,
            fts,
        )
        t1 = time.monotonic()
        result = ingest_document_graph(
            backend,
            factory,
            force=force,
            retrieval_config=RetrievalConfig(
                embeddings_id=embeddings_id,
                chunk_size_tokens=chunk_size_tokens,
                fts=fts,
            ),
            embed_workers=max(1, embed_workers if embed_workers is not None else workers),
        )
        timings["ingest_s"] = round(time.monotonic() - t1, 3)
    finally:
        backend.close()

    stats = result.model_dump()
    stats["db_path"] = str(db_p)
    stats["files_degraded"] = files_degraded
    stats["timings"] = timings
    stats["warnings"] = [*outline_warnings, *result.warnings]
    logger.success("Document Graph build complete: {}", stats)
    return stats
