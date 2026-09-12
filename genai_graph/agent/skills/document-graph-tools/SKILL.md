---
name: document-graph-tools
description: Reference for the Document Graph navigation tools (get_folder_toc, get_document_toc, get_section_content, search_sections, list_documents) and the Ladybug schema they traverse. Use to look up exact tool arguments, return shapes, and the Folder/Document/MarkdownSection node model when navigating a document corpus.
---

# Document Graph Tools Reference

## Schema

```
Folder ──CONTAINS──▶ Document ──HAS_SECTION──▶ MarkdownSection ──HAS_SUBSECTION──▶ MarkdownSection ──…
                                                      │
                                                      ├──HAS_CHUNK──▶ SectionChunk (vector index)
                                                      └──HAS_IMAGE──▶ Image (charts, plots, figures)
```

- **Folder** — a source location (directory, zip, or dataset bucket). Key `folder_id`.
- **Document** — a source file, keyed by `content_hash` (xxHash of bytes); carries `filename`,
  `markdown_hash`, `token_count`, `section_count`, routing `description`, and `summary`.
- **MarkdownSection** — one heading-delimited section, keyed by `section_id = "{markdown_hash}::{sequence}"`.
  Carries `title`, `level`, `text` (raw markdown of the section excluding subsections), `token_count`,
  `sequence`, `description`, `summary`, and `parent_section_id`.
- **SectionChunk** — text chunk for long sections, keyed by `chunk_id = "{section_id}::{chunk_index}"`.
  Indexed by HNSW vector embeddings for semantic search.
- **Image** — extracted image/chart node, keyed by `image_id = "{section_id}::{image_hash}"`.
  Carries `name` (hash), `filename`, `path`, `description` (extracted caption/alt text), and `size`.

---

## Tools Reference

### `get_folder_toc(folder_id: str | None = None) -> str`
List documents in a folder as YAML:
- Returns each document's `id` (`content_hash`), `name` (filename), `sections` count, and one-line `description`.
- When `folder_id` is omitted, lists all documents in the corpus.
- Does not list individual sections — call `get_document_toc` on selected documents.

### `get_document_toc(document_id: str, include_summaries: bool = False, max_level: int | None = None) -> str`
Returns the hierarchical section outline of a single document as YAML:
- `document_id`: accepts content hash (full or prefix), filename, or path.
- `max_level`: restricts depth (e.g. `max_level=2` for top-level headers only).
- `include_summaries`: when `True`, includes synthesized section descriptions/summaries and table metrics.
- Output contains each section's `id` (`{markdown_hash}::{sequence}`), `title`, `level`, and `description`.

### `get_section_content(section_ids: str, start_line: int | None = None, max_lines: int | None = None) -> str`
Retrieves raw Markdown body text for specified section IDs:
- `section_ids`: comma-separated string of section IDs (e.g. `"568acd8b::3,568acd8b::4"`).
- `start_line` (optional, 1-based): starting line offset within the section markdown.
- `max_lines` (optional): maximum number of lines to return.
- Essential for paginating tall/wide financial and statistical tables without context overflow.

### `search_sections(query: str, limit: int = 20, folder_id: str | None = None, document_id: str | None = None, mode: str = "hybrid") -> str`
Performs ranked search across section titles, chunk embeddings, and markdown text:
- `query`: natural language phrase or keyword search query.
- `document_id`: restrict search to a single document (eliminates cross-document noise).
- `folder_id`: restrict search to a specific folder.
- `limit`: maximum number of matching sections to return (default 20).
- `mode`: `"hybrid"` (fuses vector similarity + BM25 keyword search via RRF), `"vector"`, `"bm25"`, or `"cypher"`.
- Returns matching sections ranked best-first with `section_id`, `title`, `score`, `level`, and matching snippet.

### `search_images(query: str = "", document_id: str | None = None, section_id: str | None = None, limit: int = 10) -> str`
Search for extracted images, charts, plots, and figures in the document graph:
- `query`: search query matching image caption/description, filename, or section heading (e.g. `'unemployment rate'`, `'Figure 1'`, `'bar chart'`, `'*'` for all).
- `document_id`: optional document ID (filename, content hash) to filter results.
- `section_id`: optional section ID to restrict to a specific section.
- `limit`: maximum number of image results to return (default: 10).
- Returns a YAML list of images with `image_id`, `name`, `filename`, `path`, `description` (caption), `section_title`, and `document_name`.

### `query_image(image: str, question: str, model: str | None = None) -> str`
Analyze an image using a Vision-Language Model (VLM) to answer visual questions:
- `image`: Image ID (e.g. `'doc1::0::7a8b9c0d'`), image hash/name, filename, or local file path.
- `question`: specific question about the image (e.g. `'What is the percentage shown for 2015 in Figure 1?'`).
- `model`: optional VLM model ID (defaults to `'glm_5.3_flash@openrouter'`).
- Returns the VLM's visual analysis and answer based on the actual image pixels.

### `list_documents() -> str`
Lists all documents in the corpus with their content hash, filename, section count, and routing description. Equivalent to calling `get_folder_toc()` with no folder.

---

## Idempotence & Schema Tolerance

Tools tolerate partially-ingested or older databases: they omit fields absent in the schema and return clean "No ... found" indicators rather than raising exceptions.
