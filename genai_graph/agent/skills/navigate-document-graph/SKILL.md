---
name: navigate-document-graph
description: Answer questions over a Document Graph (Folders → Documents → Markdown sections) stored in a Ladybug database by navigating the heading hierarchy and reading only the relevant sections — hybrid agentic RAG (vector + keyword search). Use whenever the user asks about the content of ingested documents and you have the get_folder_toc, get_document_toc, get_section_content, search_sections, and list_documents tools available.
---

# Navigate the Document Graph

You answer by **reading** documents from the graph — never from memory. The graph
holds Folders → Documents → Markdown sections; every section has a one-line
`description` that is your routing signal.

## Core loop

1. **Orient.** Call `get_folder_toc(folder_id=<id>)` to list the documents in the
   target folder, each with a content hash id and a one-line description. If no
   folder was given, call `list_documents()` to see every ingested document.
   Read the descriptions and pick the document(s) most likely to answer.

2. **Get the map.** Call `get_document_toc(document_id=<hash-or-filename>)` to get
   that document's section tree as YAML: each section's `id`, `title`, and
   `description` (and `summary` if you pass `include_summaries=true`). Do NOT read
   every section — use the descriptions to pick the few that matter.

3. **Read only what matters.** Call `get_section_content(section_ids="<id1>,<id2>")`
   with the comma-separated section ids you selected. This returns the raw
   Markdown text of those sections only.

4. **Search when lost.** If the TOC descriptions do not point you to an answer,
   call `search_sections(query="<natural-language question>", folder_id=<id>)`.
   This runs a hybrid search (vector similarity over SectionChunks fused with
   BM25 keyword search) and returns ranked sections best-first with a relevance
   score and, when a chunk matched, a short snippet of the matching text. Use the
   returned section ids with `get_section_content`. Prefer one good search plus
   `get_document_toc` over many blind searches — see the map-first rule below.

5. **Iterate.** A single document rarely answers a complex question. Repeat
   across the relevant documents and sections, refining keywords, until you have
   grounded evidence for every part of the answer.

## Choosing tools

- `get_folder_toc` / `list_documents` → "which documents exist?"
- `get_document_toc` → "what sections does this document have?" (the map)
- `get_section_content` → "show me the actual text of these sections"
- `search_sections` → "where does the graph mention <query>?" (ranked hybrid vector + keyword)

## Rules

- **Cite your sources.** Reference each fact with its section id `[hash::sequence]`
  and name the source document filename.
- **No hallucination.** If a tool returns "No ... found" or a section does not
  contain the answer, say the information is not present rather than guessing.
- **Corpus only via the graph.** The document corpus is reachable ONLY through
  the graph tools (`get_folder_toc`, `get_document_toc`, `get_section_content`,
  `search_sections`). Do NOT use `read_file`, `grep`, `glob`, or `ls` to read the
  source documents or their markdown — those file tools, when present, are for
  reading skill and reference files only, never for reading the corpus.
- **Be economical.** Do not dump whole documents into your answer — synthesize.
  Read only the sections you need; the TOC descriptions exist to keep you from
  reading everything.
- **Large documents.** If a document has many sections, pass `max_level=2` to
  `get_document_toc` first for the top-level outline, then drill into the
  relevant subtree.
- **Map before you re-search.** Do not call `search_sections` more than three
  times in a row. If two searches have not landed on the answer, stop and call
  `get_document_toc` on the most relevant document to see its section map, then
  read the specific section with `get_section_content`. One grounded read beats
  another blind search.
- **Do not re-fetch the TOC.** Once you call `get_document_toc` for a document,
  its full section tree and section IDs remain in your conversation history above.
  Do NOT call `get_document_toc` again for the same document — refer to the
  earlier output to pick subsequent section IDs.

## Example: "What SLAs does the RFP require?"

1. `get_folder_toc(folder_id="folder_273e65da416b2e72")` → see the documents.
2. Spot an "Appendix 6. SLA" document → `get_document_toc(document_id="<its hash>")`.
3. Read its section descriptions → `get_section_content(section_ids="<sla section ids>")`.
4. Also `search_sections(query="availability", folder_id="folder_273e65da416b2e72")`
   to catch SLA clauses mentioned inside the main RFP body.
5. Answer, citing each SLA with its section id and the document it came from.
