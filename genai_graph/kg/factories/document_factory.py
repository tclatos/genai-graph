"""Factory that ingests a directory of files as plain Document nodes.

Each file matching the include/exclude patterns becomes a
:class:`~genai_graph.kg.nodes.document.Document` node (keyed by content hash).
No chunking or embeddings are produced — for a navigable heading hierarchy use
:class:`~genai_graph.kg.factories.document_graph_factory.DocumentGraphFactory`
instead; embeddings will be reintroduced later.

Usage in a workflow YAML:

    graph:
      factory: genai_graph.kg.factories.document_factory.DocumentDirectoryFactory
      data_root: /path/to/documents
      include: ['*.md', '*.txt']

Extending
---------
To add LLM-based entity extraction on top of the document graph, create a
subclass and override ``build_schema()`` to add your domain-specific node types
and ``get_struct_data_by_key()`` to return extracted Pydantic models per
document. This mirrors the pattern used by ``JsonFileBackedFactory`` for
BAML-extracted data.
"""

from __future__ import annotations

import mimetypes
from datetime import UTC
from pathlib import Path
from typing import ClassVar

from loguru import logger
from pydantic import BaseModel, Field

from genai_graph.kg.factories.base import KgFactory
from genai_graph.kg.nodes.document import Document, DocumentNode
from genai_graph.kg.schema.core import GraphSchema


def _mtime_iso(mtime: float) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(mtime, tz=UTC).isoformat()


class DocumentDirectoryFactory(KgFactory):
    """Factory that scans a directory and builds a graph of Document nodes.

    Each file matching the include/exclude patterns becomes a Document node.
    The factory is intentionally minimal — it does NOT perform chunking or any
    LLM-based entity extraction. Subclass it and override ``build_schema()`` /
    ``get_struct_data_by_key()`` to layer structured extraction on top.
    """

    TOP_CLASS: type[BaseModel] | None = Document

    data_root: str = Field(..., description="Root directory to scan for documents")
    include: list[str] = Field(default_factory=lambda: ["*.md", "*.txt"], description="Glob patterns to include")
    exclude: list[str] = Field(default_factory=list, description="Glob patterns to exclude")
    recursive: bool = Field(default=True, description="Recurse into sub-directories")

    # Class-level file cache (reset between sessions by clear_cache())
    _files_cache: list[Path] | None = None
    _initialized_roots: ClassVar[set[str]] = set()

    # ------------------------------------------------------------------
    # KgFactory protocol
    # ------------------------------------------------------------------

    def build_schema(self) -> GraphSchema:
        return GraphSchema(
            root_model_class=Document,
            nodes=[DocumentNode],
            relations=[],
        )

    def get_struct_data_by_key(self, key: str) -> BaseModel | None:
        """Return the Document model for the given file path (key)."""
        path = Path(key)
        if not path.exists():
            logger.warning("File not found: {}", key)
            return None
        return self._build_document(path)

    def get_keys(self) -> list[str]:
        """Return all discovered file paths as factory keys."""
        return [str(p) for p in self._get_files()]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_files(self) -> list[Path]:
        """Discover files under data_root matching include/exclude patterns."""
        if self._files_cache is not None:
            return self._files_cache

        from genai_tk.config_mgmt.file_patterns import resolve_config_path, resolve_files

        resolved_root = resolve_config_path(self.data_root)
        root_path = Path(resolved_root)

        if not root_path.exists():
            logger.warning("DocumentDirectoryFactory: data_root not found: {}", root_path)
            self._files_cache = []
            return []

        user_patterns = self.include or ["*.md", "*.txt"]
        pathspecs: list[str] = []
        for pattern in user_patterns:
            if self.recursive:
                pathspecs.append(f"**/{pattern}")
                pathspecs.append(pattern)
            else:
                pathspecs.append(pattern)

        exclude_specs: list[str] = [f"!{p}" for p in (self.exclude or [])]

        all_specs = pathspecs + exclude_specs
        discovered = resolve_files(str(root_path), pathspecs=all_specs)
        self._files_cache = [Path(p) for p in discovered]
        logger.info("DocumentDirectoryFactory: discovered {} files under {}", len(self._files_cache), root_path)
        return self._files_cache

    def _build_document(self, path: Path) -> Document:
        """Build a Document model from file metadata."""
        try:
            stat = path.stat()
            file_size: int | None = stat.st_size
            modified_at: str | None = _mtime_iso(stat.st_mtime)
        except Exception as exc:
            logger.warning("Could not stat {}: {}", path, exc)
            file_size = None
            modified_at = None

        try:
            from genai_tk.utils.hashing import file_digest

            content_hash: str = file_digest(path)
        except Exception as exc:
            from genai_tk.utils.hashing import buffer_digest

            logger.warning("Could not hash {}: {}", path, exc)
            content_hash = buffer_digest(str(path).encode("utf-8"))

        mime_type, _ = mimetypes.guess_type(str(path))

        return Document(
            content_hash=content_hash,
            path=str(path),
            filename=path.name,
            file_size=file_size,
            mime_type=mime_type,
            modified_at=modified_at,
        )

    @classmethod
    def clear_cache(cls) -> None:
        """Clear class-level file caches (call between test runs or workflow steps)."""
        cls._initialized_roots.clear()

    def get_sample_queries(self) -> list[str]:
        return [
            "MATCH (d:Document) RETURN d.filename, d.file_size ORDER BY d.filename LIMIT 20",
            "MATCH (f:Folder)-[:CONTAINS]->(d:Document) RETURN f.name, count(d) AS docs ORDER BY docs DESC",
        ]
