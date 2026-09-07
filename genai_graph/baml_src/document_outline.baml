// Structured document outline and TOC extraction schemas & functions

class TocPreambleEntry {
  title string @description("Heading or section title as listed in the Table of Contents")
  level int @description("Hierarchical level: 1 (Part/Chapter/Major Section), 2 (Item/Section), 3 (Table/Note/Subsection)")
  page string? @description("Page number or identifier if given in the TOC, otherwise null")
}

class DocumentTocPreamble {
  document_title string? @description("Title of the document if identified")
  document_description string? @description("ONE plain-text sentence, at most 20 words, on the whole document")
  document_summary string? @description("2-4 plain-text sentences, at most 60 words, abstracting the document")
  entries TocPreambleEntry[] @description("Ordered list of TOC entries")
}

class OutlineEntry {
  title string @description("The heading text EXACTLY as it appears on its own line in the document")
  level int @description("Heading level, 1 (top) to 6")
  description string? @description("One plain-text sentence naming concrete subject matter")
  summary string? @description("2-3 plain-text sentences abstracting substantial sections")
}

class DocumentOutline {
  document_description string @description("ONE plain-text sentence, at most 20 words, on the whole document")
  document_summary string @description("2-4 plain-text sentences, at most 60 words, abstracting the document")
  sections OutlineEntry[] @description("Every section in document order; titles must appear verbatim")
}

class BranchOutline {
  sections OutlineEntry[] @description("Every requested section in this branch, in exact order")
}

class DocumentSummarySynthesis {
  document_description string @description("ONE plain-text sentence, at most 20 words, on the whole document")
  document_summary string @description("2-4 plain-text sentences, at most 60 words, abstracting the document")
}

function ExtractTocPreamble(filename: string, toc_text: string) -> DocumentTocPreamble {
  client CustomClient
  prompt #"
    You extract the structured Table of Contents from the preamble or beginning of a document.
    Return all sections/items/parts/tables in the EXACT order they appear in the Table of Contents.
    For each entry:
    - title: the exact heading/section text as listed in the Table of Contents.
    - level: hierarchical depth: 1 (top-level Part/Chapter/Major Section), 2 (Item/Section/Sub-chapter), 3 (Table/Chart/Note/Subsection).
    - page: reported page number or identifier if given, else null.

    Do not invent sections not present in the Table of Contents.

    Document: {{ filename }}

    --- Table of Contents excerpt ---
    {{ toc_text }}
    --- end excerpt ---

    {{ ctx.output_format }}
  "#
}

function ExtractOutline(
  filename: string,
  headings: string,
  raw: string,
  max_description_words: int,
  max_summary_words: int
) -> DocumentOutline {
  client CustomClient
  prompt #"
    You enrich the table of contents for a document library that an AI agent reads
    to decide which section to open. The document's Markdown headings have ALREADY
    been detected for you and are listed in the user message (numbered, with their
    Markdown level as `[Llevel]`). Return EXACTLY one `sections` entry per
    listed heading, in the same order, using each heading's verbatim title and its
    listed level, plus a description (and, for substantial sections, a summary)
    that you write from the document content under that heading.

    For every listed heading, return (in the same order as the list):
    - `title`: the heading text EXACTLY as listed (verbatim).
    - `level`: the level listed for that heading.
    - `description`: ONE plain-text sentence, at most {{ max_description_words }} words, naming concrete subject matter (entities, metrics, products, scope) found under the heading. null for structural dividers with no real body. Never restate the title.
    - `summary`: ONLY for substantial sections (detailed disclosures, major tables, substantive analyses, long narrative items): 2-3 plain-text sentences, at most {{ max_summary_words }} words. null for brief sections.

    HARD RULES:
    - Return exactly one `sections` entry per listed heading. Do not omit, merge, reorder, rename, or invent headings.
    - Never include a section's body text in your answer. Output only the title, level, description and (when warranted) summary for each section, plus the two document-level fields.
    - Also return `document_description` (one sentence, at most {{ max_description_words }} words) and `document_summary` (2-4 sentences, at most {{ max_summary_words }} words) describing the whole document.

    Document: {{ filename }}

    --- headings detected in this document (return one section per heading, in this order) ---
    {{ headings }}
    --- full document ---
    {{ raw }}
    --- end document ---

    {{ ctx.output_format }}
  "#
}

function ExtractBranchOutline(
  filename: string,
  branch_title: string,
  headings: string,
  raw: string,
  max_description_words: int,
  max_summary_words: int,
  summary_min_tokens: int
) -> BranchOutline {
  client CustomClient
  prompt #"
    You enrich the table of contents for a document section branch that an AI agent reads
    to decide which section or table to open. The Markdown headings for this branch have
    ALREADY been detected and are listed in the user message (numbered, with their
    Markdown level as `[Llevel]`). Return EXACTLY one `sections` entry per
    listed heading, in the same order, using each heading's verbatim title and its
    listed level, plus a description (and, for substantial sections, a summary).

    For every listed heading, return (in the same order as the list):
    - `title`: the heading text EXACTLY as listed (verbatim).
    - `level`: the level listed for that heading.
    - `description`: ONE plain-text sentence, at most {{ max_description_words }} words, naming the CONCRETE subject matter under that heading — the specific entities, products, metrics, line items, time periods, or scope found in the body text.
      - If a heading is a Table (e.g. "Table 1", "Table 2", "Table FFO-1"), inspect the table headers and sample rows to state what metrics and categories the table covers (e.g., "Monthly receipts from individual income tax, corporation tax, and customs.").
      - NEVER restate or paraphrase the title alone (e.g. "Table 1" -> "Describes Table 1" is invalid).
      - If the heading is a structural divider with no real body text, set `description` to null.
    - `summary`: ONLY for substantial sections (more than roughly {{ summary_min_tokens }} tokens): 2-3 plain-text sentences, at most {{ max_summary_words }} words. Leave null otherwise.

    HARD RULES:
    - Return exactly one `sections` entry per listed heading in exact order.
    - Never include a section's body text in your answer.

    Document: {{ filename }}
    Branch / Major Section: {{ branch_title }}

    --- headings in this branch (return one section per heading, in this order) ---
    {{ headings }}
    --- branch content ---
    {{ raw }}
    --- end branch content ---

    {{ ctx.output_format }}
  "#
}

function SynthesizeDocumentSummary(
  filename: string,
  preamble_text: string,
  top_level_descriptions: string,
  max_description_words: int,
  max_summary_words: int
) -> DocumentSummarySynthesis {
  client CustomClient
  prompt #"
    You generate the top-level document description and summary for a document library.
    - `document_description`: exactly ONE sentence, at most {{ max_description_words }} words, stating what the document is and its scope.
    - `document_summary`: 2-4 sentences, at most {{ max_summary_words }} words, abstracting the main topics, entities, and periods covered.

    Document: {{ filename }}

    Preamble excerpt:
    {{ preamble_text }}

    Major Sections / Tables:
    {{ top_level_descriptions }}

    {{ ctx.output_format }}
  "#
}
