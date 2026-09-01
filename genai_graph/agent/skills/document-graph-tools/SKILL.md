---
name: document-graph-tools
description: Reference for the Document Graph navigation tools (get_folder_toc, get_document_toc, get_section_content, search_sections, list_documents) and the Ladybug schema they traverse. Use to look up exact tool arguments, return shapes, and the Folder/Document/MarkdownSection node model when navigating a document corpus.
---

# Document Graph Tools Reference

## Schema

```
Folder ──CONTAINS──▶ Document ──HAS_SECTION──▶ MarkdownSection ──HAS_SUBSECTION──▶ MarkdownSection ──…
```

- **Folder** — a source location (a directory, `.zip`, or site). Key `folder_id`.
- **Document** — a source file, keyed by `content_hash`; carries `filename`,
  `markdown_hash`, `token_count`, `section_count`, and a routing `description`
  plus `summary` (when summarized). The converted Markdown lives in its sections.
- **MarkdownSection** — one heading-delimited section, keyed by
  `section_id = "{markdown_hash}::{sequence}"`. Has `title`, `level`, `text`,
  `token_count`, `sequence`, `description`, `summary`, `parent_section_id`.

A section's `text` is its OWN content (heading + body up to the next heading), so
reading a section does not include its subsections — navigate to children from the
TOC when you need them.

## Tools

### `get_folder_toc(folder_id: str | None) -> str`  — start here
YAML list of the documents under a folder, each `id` (content hash), `name`,
`sections` count, and one-line `description`. Omit `folder_id` for the whole
corpus. Does not include sections — call `get_document_toc` for the doc you pick.

### `get_document_toc(document_id: str, include_summaries: bool = False, max_level: int | None = None) -> str`
YAML section tree for one document: each section `id`, `title`, `description`.
`document_id` accepts a content hash (full or prefix), filename, or path.
Use `max_level=2` for a long document's top-level outline.

### `get_section_content(section_ids: str, start_line: int | None = None, max_lines: int | None = None) -> str`
Raw Markdown text of one or more sections, comma-separated section ids
(`hash::sequence`). This is the only way to read actual body text.
Optional `start_line` (1-based) and `max_lines` allow slicing long financial tables/sections.

### `search_sections(keyword: str, limit: int = 20, folder_id: str | None = None) -> str`
Keyword search over section titles and body text. Returns matching
`section_id`, `title`, `line_start`, `markdown_hash`. Pass `folder_id` to scope.

### `list_documents() -> str`
Every ingested document with content hash, filename, section count, and
description. Equivalent to `get_folder_toc()` with no folder.

## Idempotence note

Tools tolerate a partially-ingested or older database: they omit fields the DB
does not have and return a clear "No ... found" string instead of an error.
