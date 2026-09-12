---
name: navigate-document-graph
description: Answer questions over a Document Graph (Folders → Documents → Markdown sections) stored in a Ladybug database by navigating the heading hierarchy and reading only the relevant sections — hybrid agentic RAG (vector + keyword search). Use whenever the user asks about the content of ingested documents and you have the get_folder_toc, get_document_toc, get_section_content, search_sections, and list_documents tools available.
---

# Navigate the Document Graph

You answer questions by **reading** documents from the graph — never from memory or training assumptions.
The graph models: `Folder ──CONTAINS──▶ Document ──HAS_SECTION──▶ MarkdownSection ──HAS_SUBSECTION──▶ MarkdownSection`.
Every section carries a one-line `description` (and optional `summary`) that serves as your routing signal.

---

## 1. Core Navigation Loop

1. **Orient**:
   - Call `get_folder_toc(folder_id=<id>)` to list documents in the target folder with their content hashes, filenames, and descriptions.
   - If no folder is specified, call `list_documents()` to view all ingested documents.
   - Pick the document(s) most relevant to the question based on title, date, or description.

2. **Get the Map (Outline)**:
   - Call `get_document_toc(document_id="<hash-or-filename>", max_level=2)` to inspect the top-level section hierarchy.
   - Each entry provides the section `id` (`[hash::sequence]`), `title`, `level`, and routing `description`.
   - Use `include_summaries=true` when you want synthesized descriptions and key metrics without reading full markdown tables.
   - Do NOT read all sections; use the TOC to select the specific section IDs that answer the question.

3. **Read Only What Matters**:
   - Call `get_section_content(section_ids="<id1>,<id2>")` with comma-separated section IDs to read raw Markdown body text.
   - For long sections or large tables (spanning 50+ lines), use `start_line` and `max_lines` (e.g. `get_section_content(section_ids="<id>", start_line=1, max_lines=40)`) to paginate or inspect headers without overflowing context.

4. **Targeted Search When Lost**:
   - If the TOC outline does not point directly to the answer, call `search_sections(query="<search term>", document_id="<doc_id>")`.
   - This performs hybrid search (vector similarity over SectionChunks fused with BM25 keyword search via RRF) and returns ranked sections with relevance scores and matching text snippets.
   - Always supply `document_id` when the document is already known to eliminate cross-document false positives.

5. **Visual Charts, Plots & Image Inspection**:
   - When a question requires reading a visual chart, graph, diagram, line plot, or figure that cannot be resolved from text OCR alone:
     1. Search for relevant figures with `search_images(query="<figure topic or number>", document_id="<doc_id>")`.
     2. Call `query_image(image="<image_id or path>", question="<specific visual question>")` to have the Vision-Language Model inspect the chart and return exact data points, percentages, labels, and trends.

6. **Iterate & Synthesize**:
   - For multi-period, multi-table, or multi-document questions, repeat across the relevant sections until grounded evidence is obtained for every part of the question.

---

## 2. Tool Discipline & Execution Rules

- **Multi-Step Continuity & Tool Calling Discipline**:
  - When multi-turn lookups or multiple tool calls are required, **ALWAYS invoke the next tool call directly**.
  - **Do NOT output intermediate commentary or conversational filler** (e.g., *"Let me check the next section..."*, *"Now looking at the table..."*) without a tool call. Producing text without tool calls terminates the execution loop and returns your incomplete status comment as the final answer.
  - Only emit plain text when you have retrieved all necessary data, completed any required calculations, and are ready to deliver your final answer.

- **Search Query Formulation**:
  - Formulate **compact, keyword-dense or conceptual queries** (e.g., `"Consolidated Balance Sheets"`, `"Table FFO-1"`, `"Currency in Circulation"`, `"Note 12 Leases"`).
  - **Do NOT** pass long conversational questions into `search_sections`.
  - Prefer specific table numbers, section titles, line item names, or distinct phrases.

- **Circuit Breaker — Map Before Re-Search**:
  - Do not call `search_sections` more than 2–3 times in a row. If two searches fail to land on the answer, stop searching and inspect `get_document_toc(document_id, max_level=2)` to understand the section structure, then read the target section directly with `get_section_content`.

- **Do NOT Re-Fetch Document TOC**:
  - Once `get_document_toc` has been executed for a document, its complete section outline and section IDs remain in your conversation history above. Do NOT call `get_document_toc` multiple times for the same document.

- **Corpus Access Only via Graph Tools**:
  - The document corpus is accessible ONLY through the graph tools (`get_folder_toc`, `get_document_toc`, `get_section_content`, `search_sections`, `list_documents`).
  - Do NOT attempt to use `read_file`, `grep`, `glob`, or `ls` to access source files or their underlying markdown.

- **Grounded Citation & Verification**:
  - Reference each fact with its exact section ID `[hash::sequence]` and source document filename.
  - Always verify column headers, dates, and reporting scale/units ($ thousands, $ millions, $ billions, %).
  - If a metric or table is genuinely absent from the graph, explicitly state that it is not present rather than extrapolating or guessing.

---

## 3. Tool Quick Reference

| Tool | Purpose | Key Arguments |
|------|---------|---------------|
| `get_folder_toc` | List documents in a folder | `folder_id: str \| None` |
| `list_documents` | List all ingested documents | None |
| `get_document_toc` | Get section outline / hierarchy | `document_id: str`, `max_level: int = 2`, `include_summaries: bool = False` |
| `get_section_content` | Read raw Markdown body text | `section_ids: str`, `start_line: int \| None`, `max_lines: int \| None` |
| `search_sections` | Hybrid vector + BM25 search | `query: str`, `document_id: str \| None`, `folder_id: str \| None`, `limit: int = 20` |
