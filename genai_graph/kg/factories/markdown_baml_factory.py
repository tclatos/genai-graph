"""Markdown-backed factory that extracts entities inline via a BAML function.

Unlike :class:`~genai_graph.kg.factories.json_factory.JsonFileBackedFactory` —
which reads *pre-extracted* JSON produced by a separate ``cli baml extract`` step
— this factory reads Markdown files directly and runs a BAML extraction function
on each, caching the resulting structured JSON as a build artifact so re-runs are
cheap.

Subclasses implement:
- ``build_schema()`` — returning a schema whose ``root_model_class`` is the model
  the BAML function returns (and typically calling ``get_document_schema_elements``
  to add the provenance ``Document`` node + ``MENTIONS`` relation).
- ``extract_from_markdown(md_text)`` — invoking the concrete BAML function and
  returning a populated root-model instance.

The provenance ``Document`` node is built from the Markdown file (via
``DocumentMixin``), so its ``content_hash`` matches the one produced by
:class:`~genai_graph.kg.factories.document_graph_factory.DocumentGraphFactory`
for the same file — the two factories therefore MERGE into a single Document node
carrying both the section tree and the extracted entities.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from loguru import logger
from pydantic import BaseModel

from genai_graph.kg.factories.base import KgFactory
from genai_graph.kg.factories.document_mixin import DocumentMixin
from genai_graph.kg.schema.core import GraphSchema


class MarkdownBamlFactory(DocumentMixin, KgFactory):
    """KG factory that extracts entities from Markdown files via a BAML function.

    Note: ``build_schema()`` must return a schema with ``root_model_class`` set,
    and subclasses must implement ``extract_from_markdown()``.
    """

    md_root: str
    json_cache_root: str | None = None
    include: list[str] | None = None
    exclude: list[str] | None = None
    recursive: bool = True

    _files_cache: list[Path] | None = None
    _initialized_roots: ClassVar[set[tuple[str, str]]] = set()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def model_post_init(self, _context: object) -> None:
        """Discover Markdown files under ``md_root``."""
        from genai_tk.config_mgmt.file_patterns import resolve_config_path, resolve_files

        schema = self.build_schema()
        if schema.root_model_class is None:
            raise ValueError(
                f"{self.__class__.__name__} requires build_schema() to return a schema with root_model_class set"
            )
        model_name = schema.root_model_class.__name__

        root_key = (self.md_root, model_name)
        if root_key in MarkdownBamlFactory._initialized_roots:
            self._files_cache = []
            return

        root_path = Path(resolve_config_path(self.md_root))
        if not root_path.exists():
            logger.warning("MarkdownBamlFactory: md_root not found: {}", root_path)
            self._files_cache = []
            return

        patterns = self.include or ["*.md"]
        pathspecs: list[str] = []
        for pattern in patterns:
            pathspecs.append(f"**/{pattern}" if self.recursive else pattern)
        pathspecs.extend(f"!{p}" for p in (self.exclude or []))

        self._files_cache = [Path(p) for p in resolve_files(str(root_path), pathspecs=pathspecs)]
        logger.debug("MarkdownBamlFactory: discovered {} markdown file(s) in {}", len(self._files_cache), root_path)
        MarkdownBamlFactory._initialized_roots.add(root_key)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the class-level file-discovery cache."""
        cls._initialized_roots.clear()

    def get_all_file_paths(self) -> list[Path]:
        """Return the discovered Markdown file paths (used for Document provenance)."""
        return self._files_cache or []

    def get_keys(self) -> list[str]:
        """Return the discovered Markdown file paths as factory keys."""
        return [str(p) for p in self.get_all_file_paths()]

    def get_json_cache_path(self, md_path: Path) -> Path | None:
        """Return the cached JSON path for a Markdown file, or None if caching is disabled."""
        schema = self.build_schema()
        model_cls = schema.root_model_class
        if model_cls is None:
            return None
        return self._cache_path(md_path, model_cls.__name__)

    # ------------------------------------------------------------------
    # Extraction (BAML) with JSON caching
    # ------------------------------------------------------------------

    def extract_from_markdown(self, md_text: str) -> BaseModel:
        """Run the concrete BAML extraction function on *md_text*.

        Subclasses must implement this to return a populated root-model instance.
        """
        raise NotImplementedError("Subclasses must implement extract_from_markdown()")

    def get_struct_data_by_key(self, key: str) -> BaseModel | None:
        """Return the extracted root model for a Markdown file (cached as JSON)."""
        md_path = Path(key)
        if not md_path.exists():
            logger.warning("MarkdownBamlFactory: file not found: {}", key)
            return None

        schema = self.build_schema()
        model_cls = schema.root_model_class
        assert model_cls is not None

        cache_path = self._cache_path(md_path, model_cls.__name__)
        if cache_path is not None and self._cache_fresh(cache_path, md_path):
            try:
                return model_cls.model_validate_json(cache_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Stale/invalid extraction cache {} ({}); re-extracting", cache_path, exc)

        md_text = md_path.read_text(encoding="utf-8", errors="replace")
        try:
            model = self.extract_from_markdown(md_text)
        except Exception as exc:
            logger.error("BAML extraction failed for {}: {}", md_path, exc)
            return None

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
        return model

    def _cache_path(self, md_path: Path, model_name: str) -> Path | None:
        if not self.json_cache_root:
            return None
        from genai_tk.config_mgmt.file_patterns import resolve_config_path

        root = Path(resolve_config_path(self.json_cache_root))
        return root / model_name / f"{md_path.stem}.json"

    @staticmethod
    def _cache_fresh(cache_path: Path, md_path: Path) -> bool:
        try:
            return cache_path.exists() and cache_path.stat().st_mtime >= md_path.stat().st_mtime
        except OSError:
            return False

    def build_schema(self) -> GraphSchema:
        """Build and return the graph schema (subclasses must implement)."""
        raise NotImplementedError("Subclasses must implement build_schema()")
