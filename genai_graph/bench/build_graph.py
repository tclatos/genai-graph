"""Bench plumbing for the Document Graph build.

The build pipeline itself lives in `genai_graph.kg.document_graph.build` (the
single baseline shared with `cli docgraph build` and the workflow engine); the
functions here only map benchmark doc names to staged Markdown files and fill
in bench-default paths:

1. OCR or convert target document PDF/Office files to Markdown text, writing to
   saved_markdown_dir (persistent backup/mirror) using configured markdownize profiles.
2. Stage that Markdown into project-relative markdown_dir (e.g. data/markdown_multi/).
3. Ingest into Ladybug (Kuzu) Document Graph database (Folder -> Document -> Section)
   via the shared core builder.
"""

from __future__ import annotations

import random
import shutil
import time
from pathlib import Path
from typing import Any

from loguru import logger

from genai_graph.kg.document_graph.build import build_document_graph as _build_document_graph_core
from genai_graph.kg.document_graph.build import warm_outline_cache as _warm_outline_cache_core

MD_FILENAME_SUFFIX = "_pdf.md"


def _convert_pdf(pdf_path: Path, markdownize_profile: str = "medium") -> str:
    """Return the Markdown text for *pdf_path* via the configured markdownize profile."""
    import asyncio
    import inspect

    from genai_tk.extra.markdownize.factory import ConverterFactory
    from genai_tk.workflow.markdownize.config import get_markdownize_profile

    def _sync_convert(conv_obj: Any, path: Path) -> str:
        ret = conv_obj.convert(path)
        if inspect.isawaitable(ret):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None and loop.is_running():
                # Running inside existing loop (e.g. nested task)
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    ret = pool.submit(asyncio.run, ret).result()
            else:
                ret = asyncio.run(ret)
        if hasattr(ret, "content"):
            return str(ret.content)
        return str(ret)

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
            content = _sync_convert(conv, pdf_path)
            if content and content.strip():
                return content
        except Exception as exc:
            logger.warning("Conversion attempt {} failed for {}: {}", attempt, pdf_path.name, exc)
            if attempt < max_retries:
                err = str(exc).lower()
                if "429" in err or "rate limit" in err:
                    time.sleep(min(60.0, 5 * 2**attempt) + random.uniform(0, 5))
                else:
                    time.sleep(2**attempt)
            else:
                logger.warning("Primary converter '{}' exhausted. Trying fallback 'anydoc'...", converter_name)
                try:
                    content = _sync_convert(ConverterFactory.create("anydoc"), pdf_path)
                    if content and content.strip():
                        return content
                except Exception as fb_exc:
                    logger.warning("Fallback 'anydoc' failed: {}. Trying 'markitdown'...", fb_exc)
                    try:
                        content = _sync_convert(ConverterFactory.create("markitdown"), pdf_path)
                        if content and content.strip():
                            return content
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

    clean_doc = doc_name[:-4] if doc_name.lower().endswith(".pdf") else doc_name
    pdf_root = pdfs_dir or (Path.cwd() / "data" / "pdfs")
    candidates = [
        pdf_root / f"{clean_doc}.pdf",
        pdf_root / doc_name,
        pdf_root / "documents" / f"{clean_doc}.pdf",
        pdf_root / "documents" / doc_name,
    ]
    # Files smaller than 1KB cannot be real documents (e.g. leftover error stubs
    # written as <doc>.pdf); skipping them lets the real copy in documents/ win.
    pdf_path = next(
        (p for p in candidates if p.exists() and p.is_file() and p.stat().st_size > 1000),
        None,
    )
    if pdf_path is None:
        matches = list(pdf_root.rglob(f"{clean_doc}.pdf")) or list(pdf_root.rglob(doc_name))
        if matches:
            pdf_path = matches[0]
        else:
            raise FileNotFoundError(
                f"PDF not found for doc {doc_name!r} at {pdf_root / f'{clean_doc}.pdf'}. Run fetch step first."
            )

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

    Thin bench wrapper over `genai_graph.kg.document_graph.build.warm_outline_cache`
    (see it for the full docstring): maps benchmark doc names to staged Markdown
    files and fills in bench-default paths. No DB access and no shared state, so
    this is safe to run per document from parallel Prefect tasks (per-document
    fault isolation). A document whose LLM extraction ultimately fails here is
    simply missing from the cache and degrades to algorithmic parsing at merge time.

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
    return _warm_outline_cache_core(
        _resolve_sources(md_dir, doc_names),
        db_path=db_p,
        llm=llm,
        structure_strategy=structure_strategy,
        generate_summaries=generate_summaries,
        workers=workers,
        summary_min_tokens=summary_min_tokens,
        context_safety_ratio=context_safety_ratio,
    )


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
    md_dir = markdown_dir or (Path.cwd() / "data" / "markdown_multi")
    db_p = kg_db or (Path.cwd() / "data" / "kg" / "bench.db")
    return _build_document_graph_core(
        _resolve_sources(md_dir, doc_names),
        db_p,
        force=force,
        llm=llm,
        structure_strategy=structure_strategy,
        generate_summaries=generate_summaries,
        workers=workers,
        summary_min_tokens=summary_min_tokens,
        context_safety_ratio=context_safety_ratio,
        embed_workers=embed_workers,
        outline_pre_pass=outline_pre_pass,
        embeddings_id=embeddings_id,
        fts=fts,
        chunk_size_tokens=chunk_size_tokens,
    )
