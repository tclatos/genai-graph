"""Mixin for factories that back their nodes by a source file.

Any factory that reads its data from files on disk can opt in to this mixin
to automatically produce :class:`~genai_graph.kg.nodes.document.Document`
nodes alongside the domain entities extracted from those files.

The mixin deliberately carries *zero* business logic at ingestion time — the
actual node/relationship creation is delegated to the dedicated Prefect task
``create_document_nodes_task``, which runs after the main ingestion pass.
This design makes it easy to add heavier file-processing steps (chunking,
summarization, embedding nodes, …) as additional Prefect tasks later without
touching the factory code.

Usage::

    class MyFactory(DocumentMixin, JsonFileBackedFactory, BaseModel):
        def build_schema(self) -> GraphSchema:
            nodes, relations = self.get_document_schema_elements(MyRootNode)
            return GraphSchema(root_model_class=MyRoot, nodes=[*nodes, ...], relations=[*relations, ...])
"""

from __future__ import annotations

import mimetypes
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from genai_graph.kg.nodes.document import Document
    from genai_graph.kg.schema.core import GraphNode, GraphRelation


class DocumentMixin:
    """Mixin that tags a factory as file-backed and provides document-node helpers.

    Consumers should call :meth:`get_document_schema_elements` inside
    ``build_schema()`` to obtain the ``DocumentNode`` and the ``CONTAINS``
    ``GraphRelation`` that should be registered in the schema.
    """

    def create_document_node(self, file_path: Path) -> Document:
        """Build a :class:`Document` from a file's on-disk metadata.

        Args:
            file_path: Path to the source file.

        Returns:
            Populated Document instance (not yet persisted to the graph).
        """
        from genai_tk.utils.hashing import file_digest

        from genai_graph.kg.nodes.document import Document

        try:
            stat = file_path.stat()
            file_size: int | None = stat.st_size
            modified_at: str | None = _mtime_iso(stat.st_mtime)
        except Exception as exc:
            logger.warning("Could not stat {}: {}", file_path, exc)
            file_size = None
            modified_at = None

        try:
            content_hash: str = file_digest(file_path)
        except Exception as exc:
            from genai_tk.utils.hashing import buffer_digest

            logger.warning("Could not hash {}: {}", file_path, exc)
            content_hash = buffer_digest(str(file_path).encode("utf-8"))

        mime_type, _ = mimetypes.guess_type(str(file_path))

        return Document(
            content_hash=content_hash,
            path=str(file_path),
            filename=file_path.name,
            file_size=file_size,
            mime_type=mime_type,
            modified_at=modified_at,
        )

    def get_document_schema_elements(
        self,
        root_node: GraphNode,
    ) -> tuple[list[GraphNode], list[GraphRelation]]:
        """Return the schema elements required to support Document nodes.

        Adds the canonical :data:`DocumentNode` to the node list and a
        ``MENTIONS`` :class:`GraphRelation` from ``Document`` to *root_node*.

        Args:
            root_node: The factory's root entity node (e.g. ``ReviewedOpportunityNode``).

        Returns:
            ``(nodes, relations)`` tuple to be merged into ``build_schema()`` output.
        """
        from genai_graph.kg.nodes.document import DocumentNode
        from genai_graph.kg.schema.core import GraphRelation

        mentions = GraphRelation(
            from_node=DocumentNode,
            to_node=root_node,
            name="MENTIONS",
            description="Source document that mentions this root entity",
        )
        return [DocumentNode], [mentions]


def _mtime_iso(mtime: float) -> str:
    """Convert a POSIX mtime float to an ISO-8601 UTC string."""
    from datetime import datetime

    return datetime.fromtimestamp(mtime, tz=UTC).isoformat()
