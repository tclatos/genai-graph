"""Generic source-structure node models for the Document Graph.

Folder
------
A logical base location (a local directory, a ``.zip`` archive, or — later —
a SharePoint site) from which source files are read. Storing a Folder lets
Documents carry a short ``relative_path`` instead of a full absolute path.

Document
--------
A source file from which graph data is extracted or ingested. Its primary key
is the **content hash** (``content_hash``) — two byte-identical files collapse to
one Document node. The Markdown rendering of the file is attached directly to the
Document (``markdown_hash`` plus its ``MarkdownSection`` children); there is no
separate MarkdownDocument node.

Relationships
-------------
- CONTAINS      : Folder → Document  (a folder holds source files)
- HAS_SUBFOLDER : Folder → Folder    (a folder holds a nested subfolder)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from genai_graph.kg.schema.core import GraphNode, GraphRelation

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class Folder(BaseModel):
    """A base location that source Documents are read from."""

    folder_id: str = Field(..., description="Primary key: content-hash (Merkle) identifier for the folder")
    parent_folder_id: str | None = Field(default=None, description="folder_id of the parent folder, or None if root")
    uri: str = Field(..., description="Base URI/path of the folder (directory, .zip, or remote site)")
    kind: Literal["directory", "zip", "file", "sharepoint"] = Field(
        default="directory", description="Folder backend kind"
    )
    name: str = Field(..., description="Human-friendly folder name")


class Document(BaseModel):
    """Source document node, keyed by content hash.

    Serves as the provenance anchor for everything derived from a file. The
    ``content_hash`` is the primary key, so identical files map to a single node
    regardless of where they live. ``markdown_hash`` is the hash of the file's
    Markdown rendering and is the foreign key that ``MarkdownSection`` nodes
    reference. ``path`` is informative metadata only; ``folder_id`` +
    ``relative_path`` locate the file within a Folder.
    """

    content_hash: str = Field(..., description="xxHash XXH3-64 digest of the file content (primary key)")
    markdown_hash: str | None = Field(default=None, description="xxHash digest of the Markdown rendering")
    filename: str = Field(..., description="Base filename without directory")
    folder_id: str | None = Field(default=None, description="Owning Folder.folder_id (foreign key)")
    relative_path: str | None = Field(default=None, description="Path relative to the folder base")
    path: str | None = Field(default=None, description="Absolute source path (informative only)")
    file_size: int | None = Field(default=None, description="File size in bytes")
    mime_type: str | None = Field(default=None, description="MIME type inferred from extension")
    modified_at: str | None = Field(default=None, description="Last-modified timestamp (ISO 8601)")
    token_count: int = Field(default=0, description="Approximate token count of the whole Markdown rendering")
    section_count: int = Field(default=0, description="Number of Markdown sections parsed from this document")
    language: str | None = Field(default="en", description="ISO 639-1 language code of the document (e.g. 'en', 'fr')")
    description: str | None = Field(default=None, description="One-sentence routing description of the document")
    summary: str | None = Field(default=None, description="LLM-generated document abstract (a short paragraph)")
    # Access control — basic; can be extended in domain-specific projects
    access_level: str = Field(default="public", description="Access level: public | restricted | confidential")
    allowed_roles: list[str] = Field(default_factory=list, description="Roles permitted to access this document")
    allowed_users: list[str] = Field(default_factory=list, description="Users permitted to access this document")


# ---------------------------------------------------------------------------
# GraphNode singletons — plug these into GraphSchema.nodes
# ---------------------------------------------------------------------------

FolderNode: GraphNode = GraphNode(
    node_class=Folder,
    name_from="name",
    key_from="folder_id",
    description="Base location that source documents are read from",
    explicitly_defined=True,
)

DocumentNode: GraphNode = GraphNode(
    node_class=Document,
    name_from="filename",
    key_from="content_hash",
    description="Source document from which graph data was extracted (keyed by content hash)",
    explicitly_defined=True,
)

# ---------------------------------------------------------------------------
# GraphRelation singletons — plug these into GraphSchema.relations
# ---------------------------------------------------------------------------

CONTAINS_DOC: GraphRelation = GraphRelation(
    name="CONTAINS",
    from_node=FolderNode,
    to_node=DocumentNode,
    description="Folder holds a source document",
)

HAS_SUBFOLDER: GraphRelation = GraphRelation(
    name="HAS_SUBFOLDER",
    from_node=FolderNode,
    to_node=FolderNode,
    description="Parent folder contains a nested child folder",
    field_paths=[("", "")],
)
