"""Factory that ingests a Markdown corpus as a Folder → Document → Section graph.

Each source file becomes a :class:`~genai_graph.kg.nodes.document.Document`
(keyed by content hash) inside a :class:`~genai_graph.kg.nodes.document.Folder`.
The document's Markdown heading hierarchy is extracted into a flat,
order-preserving list of
:class:`~genai_graph.kg.nodes.document_section.MarkdownSection` nodes (see
:mod:`genai_graph.kg.document_graph.tree_parser`).

`sources` accepts a mix of directories, individual files, and ``.zip`` archives;
each source becomes a Folder (see
:class:`~genai_graph.kg.document_graph.repository.SourceFolder`).

The actual graph is built by
:func:`genai_graph.kg.document_graph.ingest.ingest_document_graph`, which bypasses
the generic Pydantic-nesting extraction (the section hierarchy is
self-referential) and merges nodes/relationships directly via the same
Arrow/Ladybug primitives (`merge_nodes_batch`, `merge_relationships_batch`) used
elsewhere in genai-graph.
"""

from __future__ import annotations

import mimetypes
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr

from genai_graph.kg.document_graph.outline_extract import (
    OutlineConfig,
    OutlineResult,
    OutlineStats,
    extract_outline,
)
from genai_graph.kg.document_graph.outline_merge import merge_outline
from genai_graph.kg.document_graph.repository import FolderTree, SourceFolder
from genai_graph.kg.document_graph.tree_parser import _estimate_token_count, detect_headings, parse_markdown_tree
from genai_graph.kg.factories.base import KgFactory
from genai_graph.kg.nodes.document import (
    CONTAINS_DOC,
    HAS_SUBFOLDER,
    Document,
    DocumentNode,
    Folder,
    FolderNode,
)
from genai_graph.kg.nodes.document_section import (
    HAS_CHUNK,
    HAS_SECTION,
    HAS_SUBSECTION,
    MarkdownSection,
    SectionChunkNode,
    SectionNode,
)
from genai_graph.kg.schema.core import GraphSchema


class DocumentGraphBundle(BaseModel):
    """A fully parsed Markdown document: its folder ancestor chain, document and sections."""

    folders: list[Folder] = Field(..., description="Ancestor Folder chain, root-first, immediate-parent-last")
    document: Document
    sections: list[MarkdownSection] = Field(default_factory=list)


class DocumentGraphFactory(KgFactory):
    """Scans directories/files/zip archives and builds a Folder+Document+Section graph.

    Extracts the *heading hierarchy* of each Markdown file so an agent can
    navigate a document's table of contents and fetch only the sections it needs,
    and it stores the file structure (Folder → Document) explicitly.
    """

    TOP_CLASS: type[BaseModel] | None = None

    sources: list[str] = Field(..., description="Directories, files, or .zip archives to ingest")
    include: list[str] = Field(default_factory=lambda: ["*.md"], description="Glob patterns to include")
    exclude: list[str] = Field(default_factory=list, description="Glob patterns to exclude")
    recursive: bool = Field(default=True, description="Recurse into sub-directories")
    cache_dir: str | None = Field(default=None, description="Directory to extract .zip archives into")
    outline_config: OutlineConfig | None = Field(
        default=None,
        description="When set, build each document's sections from an LLM outline (structure + summaries) "
        "instead of the algorithmic heading parser; None keeps the fast algo path",
    )

    # Per-instance caches
    _files_cache: list[Path] | None = None
    _folder_tree_by_file: dict[str, FolderTree] | None = None
    _content_hash_by_file: dict[str, str] | None = None
    # Soft warnings from on-the-fly outline extraction in `_build_bundle` (cache misses
    # after, or without, the parallel pre-pass). The flow's pre-pass surfaces its own
    # warnings via `OutlineStats`; this list only fills in for the no-pre-pass edge case.
    _outline_warnings: list[str] = PrivateAttr(default_factory=list)

    # ------------------------------------------------------------------
    # KgFactory protocol
    # ------------------------------------------------------------------

    def build_schema(self) -> GraphSchema:
        return GraphSchema(
            root_model_class=None,
            nodes=[FolderNode, DocumentNode, SectionNode, SectionChunkNode],
            relations=[CONTAINS_DOC, HAS_SUBFOLDER, HAS_SECTION, HAS_SUBSECTION, HAS_CHUNK],
        )

    def get_keys(self) -> list[str]:
        """Return all discovered Markdown file paths as factory keys."""
        return [str(p) for p in self._get_files()]

    def get_struct_data_by_key(self, key: str) -> DocumentGraphBundle | None:
        """Return the parsed document-graph bundle for a file."""
        path = Path(key)
        if not path.exists():
            logger.warning("DocumentGraphFactory: file not found: {}", key)
            return None
        if self._folder_tree_by_file is None:
            self._get_files()
        assert self._folder_tree_by_file is not None and self._content_hash_by_file is not None
        tree = self._folder_tree_by_file.get(key)
        content_hash = self._content_hash_by_file.get(key)
        if tree is None or content_hash is None:
            # File not discovered through a source (e.g. a direct key) — build a standalone one-file tree.
            from genai_tk.utils.hashing import file_digest

            folder = SourceFolder.from_source(str(path.parent), cache_dir=self.cache_dir)
            resolved_files = folder.iter_files(single_file=path)
            content_hash = file_digest(path)
            tree = FolderTree(folder)
            tree.build(resolved_files, {str(path): content_hash})
        return self._build_bundle(path, tree, content_hash)

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def _get_files(self) -> list[Path]:
        if self._files_cache is not None:
            return self._files_cache

        from genai_tk.config_mgmt.file_patterns import resolve_config_path
        from genai_tk.utils.hashing import file_digest

        files: list[Path] = []
        folder_tree_by_file: dict[str, FolderTree] = {}
        content_hash_by_file: dict[str, str] = {}
        seen: set[str] = set()

        for raw_source in self.sources:
            resolved = Path(resolve_config_path(raw_source))
            if not resolved.exists():
                logger.warning("DocumentGraphFactory: source not found: {}", resolved)
                continue

            folder = SourceFolder.from_source(raw_source, cache_dir=self.cache_dir)
            single = resolved if (resolved.is_file() and resolved.suffix.lower() != ".zip") else None
            resolved_files = folder.iter_files(
                include=self.include, exclude=self.exclude, recursive=self.recursive, single_file=single
            )

            new_files = [rf for rf in resolved_files if str(rf.abs_path.resolve()) not in seen]
            seen.update(str(rf.abs_path.resolve()) for rf in new_files)
            if not new_files:
                continue

            content_hashes = {str(rf.abs_path): file_digest(rf.abs_path) for rf in new_files}
            tree = FolderTree(folder)
            tree.build(new_files, content_hashes)

            for rf in new_files:
                key = str(rf.abs_path)
                files.append(rf.abs_path)
                folder_tree_by_file[key] = tree
                content_hash_by_file[key] = content_hashes[key]

        self._files_cache = files
        self._folder_tree_by_file = folder_tree_by_file
        self._content_hash_by_file = content_hash_by_file
        logger.info("DocumentGraphFactory: discovered {} file(s) from {} source(s)", len(files), len(self.sources))
        return files

    # ------------------------------------------------------------------
    # Bundle construction
    # ------------------------------------------------------------------

    def _extract_outline_result(self, text: str, markdown_hash: str, filename: str) -> OutlineResult:
        """Read one document's cached outline, extracting it on a cache miss."""
        assert self.outline_config is not None
        return extract_outline(text, markdown_hash, filename, self.outline_config, warnings=self._outline_warnings)

    def extract_outlines(self, workers: int = 4) -> OutlineStats:
        """Pre-extract (and cache) outlines for every discovered file, in parallel.

        No database access: it only reads files and warms the content-addressed
        outline cache, so the subsequent ingest pass reads each outline from disk
        without an LLM call. Run it before :func:`~genai_graph.kg.document_graph.ingest.ingest_document_graph`
        when ``outline_config`` is set. A no-op (empty stats) when there is no
        ``outline_config`` or no files.
        """
        from genai_tk.utils.hashing import buffer_digest

        files = self._get_files()
        if self.outline_config is None or not files:
            return OutlineStats()

        def _one(path: Path) -> tuple[OutlineResult, list[str]]:
            text = path.read_text(encoding="utf-8", errors="replace")
            md_hash = buffer_digest(text.encode("utf-8"))
            local_warnings: list[str] = []
            result = extract_outline(text, md_hash, path.name, self.outline_config, warnings=local_warnings)
            return result, local_warnings

        if workers <= 1:
            pairs: list[tuple[OutlineResult, list[str]]] = [_one(p) for p in files]
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=workers) as ex:
                pairs = list(ex.map(_one, files))

        degraded = 0
        llm_calls = 0
        warnings: list[str] = []
        for result, file_warnings in pairs:
            if result.degraded:
                degraded += 1
            llm_calls += result.llm_calls
            warnings.extend(file_warnings)
        logger.info(
            "Outline pre-pass: {} file(s), {} degraded, {} LLM call(s)",
            len(files),
            degraded,
            llm_calls,
        )
        return OutlineStats(total_files=len(files), degraded_count=degraded, llm_calls=llm_calls, warnings=warnings)

    def _build_bundle(self, path: Path, tree: FolderTree, content_hash: str) -> DocumentGraphBundle:
        from genai_tk.utils.hashing import buffer_digest

        text = path.read_text(encoding="utf-8", errors="replace")

        try:
            stat = path.stat()
            file_size: int | None = stat.st_size
            modified_at: str | None = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not stat {}: {}", path, exc)
            file_size = None
            modified_at = None

        mime_type, _ = mimetypes.guess_type(str(path))

        markdown_hash = buffer_digest(text.encode("utf-8"))
        document_description: str | None = None
        document_summary: str | None = None
        if self.outline_config is not None:
            outline_result = self._extract_outline_result(text, markdown_hash, path.name)
            if outline_result.outline is not None:
                flat_sections = merge_outline(text, outline_result.outline, detect_headings(text))
                document_description = outline_result.outline.document_description
                document_summary = outline_result.outline.document_summary
            else:
                # Degraded (over context window or LLM failure): algorithmic structure, no summaries.
                flat_sections = parse_markdown_tree(text)
        else:
            flat_sections = parse_markdown_tree(text)

        sections = [
            MarkdownSection(
                section_id=f"{markdown_hash}::{idx}",
                markdown_hash=markdown_hash,
                parent_section_id=(f"{markdown_hash}::{fs.parent_index}" if fs.parent_index is not None else None),
                title=fs.title,
                level=fs.level,
                line_start=fs.line_start,
                line_end=fs.line_end,
                text=fs.text,
                token_count=fs.token_count,
                sequence=idx,
                description=fs.description,
                summary=fs.summary,
                summary_source=fs.summary_source,
            )
            for idx, fs in enumerate(flat_sections)
        ]

        chain = tree.chain_for(path)

        document = Document(
            content_hash=content_hash,
            markdown_hash=markdown_hash,
            filename=path.name,
            folder_id=chain[-1],
            relative_path=tree.source.relative_path_of(path),
            path=str(path),
            file_size=file_size,
            mime_type=mime_type,
            modified_at=modified_at,
            token_count=_estimate_token_count(text),
            section_count=len(sections),
            description=document_description,
            summary=document_summary,
        )

        return DocumentGraphBundle(
            folders=[tree.folders[fid] for fid in chain],
            document=document,
            sections=sections,
        )
