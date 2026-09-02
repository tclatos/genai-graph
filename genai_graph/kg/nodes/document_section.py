"""Graph node model for Document Graph sections.

A ``MarkdownSection`` represents one heading-delimited section of a
:class:`~genai_graph.kg.nodes.document.Document`'s Markdown rendering. Sections
are stored as a flat node table (not a nested Pydantic structure) with an
explicit ``parent_section_id`` — the hierarchy is materialised entirely as
``HAS_SUBSECTION`` graph edges, which lets an agent navigate the tree with
ordinary Cypher traversals.

A section's ``text`` is its **own** content (heading line plus body up to the
next heading of any level), so the sections of a document partition its lines
without overlap and the original document can be reconstructed by concatenating
them in ``sequence`` order. Every document has at least one section (a synthetic
root section captures any preamble or a heading-less document).

These node/relation singletons are ingested directly via
:func:`genai_graph.kg.document_graph.ingest.ingest_document_graph`, which bypasses
the generic Pydantic-nesting extraction (``extract_graph_data``) — the section
hierarchy is a self-referential structure that doesn't map cleanly onto that
mechanism — and instead builds nodes/relationships explicitly, then merges them
with the same Arrow/Ladybug primitives (`merge_nodes_batch`,
`merge_relationships_batch`) used everywhere else in genai-graph.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from genai_graph.kg.nodes.document import DocumentNode
from genai_graph.kg.schema.core import GraphNode, GraphRelation


class MarkdownSection(BaseModel):
    """A single heading-delimited section of a document's Markdown rendering."""

    section_id: str = Field(..., description="Primary key: f'{markdown_hash}::{sequence}'")
    markdown_hash: str = Field(..., description="markdown_hash of the owning Document (foreign key)")
    parent_section_id: str | None = Field(
        default=None, description="section_id of the parent section, or None for a top-level (root) section"
    )
    title: str = Field(..., description="Heading text")
    level: int = Field(..., description="Heading level, 0 (synthetic root) or 1 (H1) to 6 (H6)")
    line_start: int = Field(..., description="1-indexed source line where this section's own text starts")
    line_end: int = Field(..., description="1-indexed source line where this section's own text ends (inclusive)")
    text: str = Field(..., description="Own Markdown text: heading line + body up to the next heading (any level)")
    token_count: int = Field(..., description="Approximate token count")
    sequence: int = Field(..., description="0-based position of this section within its document, in document order")
    description: str | None = Field(
        default=None, description="One-sentence routing description: what this section contains"
    )
    summary: str | None = Field(default=None, description="Short paragraph summary; only for substantial sections")
    summary_source: str | None = Field(
        default=None, description="How the description/summary was produced: 'llm', or None if not yet summarized"
    )


# ---------------------------------------------------------------------------
# GraphNode / GraphRelation singletons
# ---------------------------------------------------------------------------
#
# `field_paths` are set explicitly to a no-op sentinel because the section
# hierarchy is self-referential and is populated by the direct ingestion path
# in `genai_graph.kg.document_graph.ingest`, not by the generic
# Pydantic-nesting-based `extract_graph_data()` extraction.

SectionNode: GraphNode = GraphNode(
    node_class=MarkdownSection,
    name_from="title",
    key_from="section_id",
    description="A heading-delimited section of a document's Markdown rendering",
    explicitly_defined=True,
)

HAS_SECTION: GraphRelation = GraphRelation(
    name="HAS_SECTION",
    from_node=DocumentNode,
    to_node=SectionNode,
    description="Document has a top-level (root) Markdown section",
    field_paths=[("", "")],
)

HAS_SUBSECTION: GraphRelation = GraphRelation(
    name="HAS_SUBSECTION",
    from_node=SectionNode,
    to_node=SectionNode,
    description="Parent section contains a nested child section",
    field_paths=[("", "")],
)


# ---------------------------------------------------------------------------
# Section chunks — embedding-bearing sub-units of a section
# ---------------------------------------------------------------------------
#
# A ``SectionChunk`` is a contiguous slice of a ``MarkdownSection``'s ``text``.
# Chunks are the sole embedding-bearing node of the Document Graph: each chunk
# carries a ``chunk_embedding`` (a fixed-length ``FLOAT[N]`` column created and
# HNSW-indexed by the ingest path, not a Pydantic field) computed from a
# contextualized input — ``"{title} | {description}\n\n{chunk_text}"`` — so a
# semantic search over chunks resolves to the section that holds the evidence.
# Short sections yield a single chunk equal to the whole section; long sections
# are split on a token budget. The hierarchy stays ``Document → Section``; chunks
# are a flat, indexed leaf layer reached via ``HAS_CHUNK``.


class SectionChunk(BaseModel):
    """A contiguous text slice of a MarkdownSection, indexed for semantic search."""

    chunk_id: str = Field(..., description="Primary key: f'{section_id}::c{chunk_index}'")
    section_id: str = Field(..., description="section_id of the owning MarkdownSection (foreign key)")
    markdown_hash: str = Field(..., description="markdown_hash of the owning Document")
    chunk_index: int = Field(..., description="0-based position of this chunk within its section")
    chunk_text: str = Field(..., description="A contiguous slice of the section's body text")
    token_count: int = Field(..., description="Approximate token count of chunk_text")


SectionChunkNode: GraphNode = GraphNode(
    node_class=SectionChunk,
    name_from="chunk_id",
    key_from="chunk_id",
    description="An embedding-indexed text slice of a MarkdownSection",
    explicitly_defined=True,
)

HAS_CHUNK: GraphRelation = GraphRelation(
    name="HAS_CHUNK",
    from_node=SectionNode,
    to_node=SectionChunkNode,
    description="Section contains a chunk (embedding-bearing sub-unit)",
    field_paths=[("", "")],
)
