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


def build_document_graph(
    doc_name: str | None = None,
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
) -> dict[str, Any]:
    """Ingest Markdown files into Ladybug Document Graph database."""
    from genai_graph.kg.backend import KuzuBackend
    from genai_graph.kg.factories.document_graph_factory import DocumentGraphFactory

    md_dir = markdown_dir or (Path.cwd() / "data" / "markdown_multi")
    db_p = kg_db or (Path.cwd() / "data" / "kg" / "bench.db")
    db_p.parent.mkdir(parents=True, exist_ok=True)

    resolved_llm = _resolve_build_llm(llm)
    backend = KuzuBackend(db_path=str(db_p))

    factory = DocumentGraphFactory(
        data_root=md_dir,
        recursive=True,
        llm=resolved_llm,
        structure_strategy=structure_strategy,
        generate_summaries=generate_summaries,
        workers=workers,
        summary_min_tokens=summary_min_tokens,
        context_safety_ratio=context_safety_ratio,
        embeddings=embeddings_id,
        fts=fts,
        chunk_size_tokens=chunk_size_tokens,
    )

    if force:
        logger.info("Force rebuild requested: dropping existing graph in {}", db_p)
        try:
            backend.execute("MATCH (n) DETACH DELETE n;")
        except Exception as exc:
            logger.debug("Could not detach delete existing graph (may be empty): {}", exc)

    logger.info(
        "Ingesting Document Graph from {} into {} (llm={}, embeddings={}, fts={})",
        md_dir,
        db_p,
        resolved_llm,
        embeddings_id,
        fts,
    )
    schema = factory.create_schema()
    backend.apply_schema(schema)
    stats = factory.ingest(backend)
    logger.success("Document Graph build complete: {}", stats)
    return stats if isinstance(stats, dict) else {"status": "ok"}
