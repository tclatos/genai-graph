"""Build the Document Graph for benchmark documents.

Pipeline:
1. OCR or convert target document PDF/Office files to Markdown text, writing to
   saved_markdown_dir (persistent backup/mirror) using configured markdownize profiles.
2. Stage that Markdown into project-relative markdown_dir (e.g. data/markdown_multi/).
3. Ingest into Ladybug (Kuzu) Document Graph database (Folder -> Document -> Section).
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from loguru import logger

from genai_graph.kg.document_graph.outline_extract import OutlineConfig

MD_FILENAME_SUFFIX = "_pdf.md"


def _resolve_build_llm(llm: str | None) -> str | None:
    """Resolve build LLM to a model ID or None."""
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


def _convert_pdf(pdf_path: Path, markdownize_profile: str = "medium") -> str:
    """Return the Markdown text for *pdf_path* via the configured markdownize profile."""
    from genai_tk.extra.markdownize.factory import ConverterFactory
    from genai_tk.workflow.markdownize.config import get_markdownize_profile

    try:
        prof = get_markdownize_profile(markdownize_profile)
        converter_name = prof.select_route(pdf_path)
    except Exception as exc:
        logger.warning(
            "Failed to resolve markdownize profile '{}': {}; defaulting to mistral_ocr.",
            markdownize_profile,
            exc,
        )
        converter_name = "mistral_ocr"

    max_retries = 3 if converter_name in ("mistral_ocr", "mistral", "lighton_ocr") else 1
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Converting {} with '{}' (attempt {}/{})",
                pdf_path.name,
                converter_name,
                attempt,
                max_retries,
            )
            conv = ConverterFactory.create(converter_name)
            res = conv.convert(pdf_path)
            if res and res.content:
                return res.content
        except Exception as exc:
            logger.warning("Conversion attempt {} failed for {}: {}", attempt, pdf_path.name, exc)
            if attempt < max_retries:
                time.sleep(2**attempt)
            else:
                logger.warning("Primary converter '{}' exhausted. Trying fallback 'anydoc'...", converter_name)
                try:
                    res = ConverterFactory.create("anydoc").convert(pdf_path)
                    if res and res.content:
                        return res.content
                except Exception as fb_exc:
                    logger.warning("Fallback 'anydoc' failed: {}. Trying 'markitdown'...", fb_exc)
                    try:
                        res = ConverterFactory.create("markitdown").convert(pdf_path)
                        if res and res.content:
                            return res.content
                    except Exception as m_exc:
                        raise RuntimeError(f"All conversion strategies failed for {pdf_path}: {m_exc}") from m_exc
    raise RuntimeError(f"No Markdown generated for {pdf_path}")


def markdownize_target(
    doc_name: str,
    *,
    force: bool = False,
    pdfs_dir: Path | None = None,
    saved_markdown_dir: Path | None = None,
    onedrive_markdown_dir: Path | None = None,
    markdownize_profile: str = "medium",
) -> Path:
    """Ensure document PDF is converted to Markdown and saved in saved_markdown_dir."""
    target_saved_dir = saved_markdown_dir or onedrive_markdown_dir or (Path.cwd() / "data" / "saved_markdown")
    target_saved_dir.mkdir(parents=True, exist_ok=True)
    out_md = target_saved_dir / f"{doc_name}{MD_FILENAME_SUFFIX}"

    if out_md.exists() and not force and out_md.stat().st_size > 0:
        logger.debug("Markdown already exists in saved dir: {}", out_md)
        return out_md

    pdf_root = pdfs_dir or (Path.cwd() / "data" / "pdfs")
    pdf_path = pdf_root / f"{doc_name}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found for doc {doc_name!r} at {pdf_path}. Run fetch step first.")

    content = _convert_pdf(pdf_path, markdownize_profile=markdownize_profile)
    out_md.write_text(content, encoding="utf-8")
    logger.success("Wrote Markdown ({} bytes) -> {}", len(content), out_md)
    return out_md


def copy_markdown_to_project(
    source_md: Path,
    markdown_dir: Path | None = None,
) -> Path:
    """Copy a Markdown file into the project-relative markdown directory."""
    dest_dir = markdown_dir or (Path.cwd() / "data" / "markdown_multi")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / source_md.name
    shutil.copy2(source_md, dest_file)
    logger.debug("Copied Markdown -> {}", dest_file)
    return dest_file


def _build_outline_config(
    *,
    llm: str | None,
    structure_strategy: str,
    generate_summaries: bool,
    workers: int,
    summary_min_tokens: int,
    context_safety_ratio: float,
    kg_db: Path,
) -> OutlineConfig | None:
    """Build the outline policy for *kg_db*, or None for the algorithmic-only path."""
    resolved_llm = _resolve_build_llm(llm)
    if resolved_llm is None and structure_strategy == "algo":
        return None
    return OutlineConfig(
        llm=resolved_llm,
        structure_strategy=structure_strategy,
        generate_summaries=generate_summaries,
        workers=workers,
        summary_min_tokens=summary_min_tokens,
        context_safety_ratio=context_safety_ratio,
        cache_root=str(kg_db.with_suffix("")) + "_outlines",
    )


def _resolve_sources(md_dir: Path, doc_names: list[str] | None) -> list[str]:
    """Map selected benchmark doc names to staged markdown files (whole dir when None)."""
    if not doc_names:
        return [str(md_dir)]
    sources = []
    for name in doc_names:
        doc_md = md_dir / f"{name}{MD_FILENAME_SUFFIX}"
        if doc_md.exists():
            sources.append(str(doc_md))
    missing = len(doc_names) - len(sources)
    if missing:
        logger.warning("{} selected doc(s) have no staged markdown in {}; skipping them", missing, md_dir)
    if not sources:
        raise FileNotFoundError(f"No staged markdown found for {doc_names} in {md_dir}")
    return sources


def warm_outline_cache(
    doc_names: list[str] | None = None,
    *,
    markdown_dir: Path | None = None,
    kg_db: Path | None = None,
    llm: str | None = None,
    structure_strategy: str = "auto",
    generate_summaries: bool = True,
    workers: int = 4,
    summary_min_tokens: int = 800,
    context_safety_ratio: float = 0.9,
) -> dict[str, Any]:
    """Extract outlines for the selected documents without touching the database.

    Idempotent: results land in the content-addressed outline cache, so a later
    ``build_document_graph(outline_pre_pass=False)`` reads them from disk with no
    LLM calls. No DB access and no shared state, so this is safe to run per
    document from parallel Prefect tasks (per-document fault isolation). A
    document whose LLM extraction ultimately fails here is simply missing from
    the cache and degrades to algorithmic parsing at merge time.

    Args:
        doc_names: Benchmark doc names to warm (matched as ``<name>_pdf.md`` in
            *markdown_dir*); None warms every ``*.md`` in the directory.
        markdown_dir: Directory holding staged Markdown files.
        kg_db: Ladybug database path (only used to derive the cache root).
        llm: Build LLM id (``name@provider`` or tag); None keeps the algorithmic path.
        structure_strategy: Outline strategy: 'auto' | 'algo' | 'toc_preamble' | 'llm_full'.
        generate_summaries: Generate LLM section descriptions & summaries.
        workers: Parallel LLM workers for branch summarization within a document.
        summary_min_tokens: Prompt heuristic for 'substantial' sections.
        context_safety_ratio: Degrade a document to algorithmic parsing when its
            token count exceeds this fraction of the model's context window.

    Returns:
        `OutlineStats` dict for the warmed documents.
    """
    md_dir = markdown_dir or (Path.cwd() / "data" / "markdown_multi")
    db_p = kg_db or (Path.cwd() / "data" / "kg" / "bench.db")
    outline_config = _build_outline_config(
        llm=llm,
        structure_strategy=structure_strategy,
        generate_summaries=generate_summaries,
        workers=workers,
        summary_min_tokens=summary_min_tokens,
        context_safety_ratio=context_safety_ratio,
        kg_db=db_p,
    )
    if outline_config is None:
        return {"status": "skipped", "reason": "algorithmic-only build (no outline cache)"}

    from genai_graph.kg.factories.document_graph_factory import DocumentGraphFactory

    factory = DocumentGraphFactory(sources=_resolve_sources(md_dir, doc_names), recursive=True, outline_config=outline_config)
    return factory.extract_outlines(workers=workers).model_dump()


def build_document_graph(
    doc_names: list[str] | None = None,
    *,
    force: bool = False,
    llm: str | None = None,
    structure_strategy: str = "auto",
    generate_summaries: bool = True,
    workers: int = 4,
    summary_min_tokens: int = 800,
    context_safety_ratio: float = 0.9,
    markdown_dir: Path | None = None,
    kg_db: Path | None = None,
    embeddings_id: str | None = None,
    fts: bool = True,
    chunk_size_tokens: int = 1500,
    outline_pre_pass: bool = True,
    embed_workers: int | None = None,
) -> dict[str, Any]:
    """Ingest Markdown files into Ladybug Document Graph database.

    Args:
        doc_names: Restrict ingestion to these benchmark documents (matched as
            ``<name>_pdf.md`` inside *markdown_dir*); None ingests the directory.
        force: Rebuild sections for documents already present in the graph.
        llm: Build LLM id (``name@provider`` or tag); None keeps the algorithmic path.
        structure_strategy: Outline strategy: 'auto' | 'algo' | 'toc_preamble' | 'llm_full'.
        generate_summaries: Generate LLM section descriptions & summaries.
        workers: Parallel LLM/embedding workers (cross-document pre-pass and
            per-document branch calls).
        summary_min_tokens: Prompt heuristic for 'substantial' sections.
        context_safety_ratio: Degrade a document to algorithmic parsing when its
            token count exceeds this fraction of the model's context window.
        markdown_dir: Directory holding staged Markdown files.
        kg_db: Ladybug database path.
        embeddings_id: Embeddings model for SectionChunk vectors (None disables).
        fts: Create the native BM25/FTS index over sections.
        chunk_size_tokens: Target chunk size for long sections.
        outline_pre_pass: Warm the content-addressed outline cache in parallel
            across documents before ingesting (recommended; set False when the
            cache was already warmed by `warm_outline_cache`, e.g. per-document
            Prefect tasks).
        embed_workers: Parallel workers for per-document chunk-embedding batches
            during ingest; defaults to *workers*. 1 keeps embedding serial.

    Returns:
        Ingest statistics with per-stage ``timings`` (seconds) and warnings.
    """
    from genai_graph.kg.backend import KuzuBackend
    from genai_graph.kg.document_graph.ingest import ingest_document_graph
    from genai_graph.kg.document_graph.retrieval import RetrievalConfig
    from genai_graph.kg.factories.document_graph_factory import DocumentGraphFactory

    md_dir = markdown_dir or (Path.cwd() / "data" / "markdown_multi")
    db_p = kg_db or (Path.cwd() / "data" / "kg" / "bench.db")
    db_p.parent.mkdir(parents=True, exist_ok=True)

    outline_config = _build_outline_config(
        llm=llm,
        structure_strategy=structure_strategy,
        generate_summaries=generate_summaries,
        workers=workers,
        summary_min_tokens=summary_min_tokens,
        context_safety_ratio=context_safety_ratio,
        kg_db=db_p,
    )
    resolved_llm = outline_config.llm if outline_config else None

    factory = DocumentGraphFactory(sources=_resolve_sources(md_dir, doc_names), recursive=True, outline_config=outline_config)

    timings: dict[str, float] = {}
    backend = KuzuBackend()
    backend.connect(str(db_p))
    try:
        outline_warnings: list[str] = []
        if outline_config is not None and outline_pre_pass:
            t0 = time.monotonic()
            stats = factory.extract_outlines(workers=workers)
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
            md_dir,
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
    stats["timings"] = timings
    stats["warnings"] = [*outline_warnings, *result.warnings]
    logger.success("Document Graph build complete: {}", stats)
    return stats
