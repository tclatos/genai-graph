"""Generic graph node types for the genai-graph knowledge graph framework.

These types are domain-agnostic and can be used in any graph-building application.
Domain-specific node types (e.g. EKG-specific Opportunity, Customer, L3) live in
the downstream project that imports genai-graph.
"""

from genai_graph.kg.nodes.document import (
    CONTAINS_DOC,
    Document,
    DocumentNode,
    Folder,
    FolderNode,
)
from genai_graph.kg.nodes.document_section import (
    HAS_CHUNK,
    HAS_IMAGE,
    HAS_SECTION,
    HAS_SUBSECTION,
    Image,
    ImageNode,
    MarkdownSection,
    SectionChunk,
    SectionChunkNode,
    SectionNode,
)

__all__ = [
    "Document",
    "Folder",
    "DocumentNode",
    "FolderNode",
    "CONTAINS_DOC",
    "MarkdownSection",
    "SectionNode",
    "SectionChunk",
    "SectionChunkNode",
    "Image",
    "ImageNode",
    "HAS_SECTION",
    "HAS_SUBSECTION",
    "HAS_CHUNK",
    "HAS_IMAGE",
]
