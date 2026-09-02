# Document Graph

The **Document Graph** is genai-graph's generic representation of a corpus of source
documents: which folder they came from, their file metadata, and — for Markdown —
their heading hierarchy. It is the provenance layer that every entity-extraction
factory (BAML, CRM tables, Neo4j imports, …) attaches to, and it is also useful on
its own as a **vectorless, agentic RAG** substrate: an agent walks a document's table
of contents and reads section text directly, no embeddings required.

## Schema

```
Folder ──CONTAINS──▶ Document ──HAS_SECTION──▶ MarkdownSection ──HAS_SUBSECTION──▶ MarkdownSection ──HAS_SUBSECTION──▶ …
```

| Node | Key | Notes |
|------|-----|-------|
| `Folder` | `folder_id` | A directory, a `.zip` archive, or a single file's parent — the base location documents are read from. |
| `Document` | `content_hash` (xxHash of the raw file bytes) | Provenance anchor for everything derived from a file. Carries `filename`, `relative_path`, `path`, `mime_type`, plus `markdown_hash`, `token_count`, `section_count`, and (once summarized) a one-sentence `description` and a paragraph `summary`. |
| `MarkdownSection` | `section_id` = `{markdown_hash}::{sequence}` | One heading-delimited section (heading line + body up to the next heading of any level). Every document has at least one section — a synthetic level-0 root section captures a heading-less document or its preamble. Carries `token_count` and, once summarized, `description` (always) plus `summary` (substantial sections only) and `summary_source`. |

Sections form a flat table with an explicit `parent_section_id` — the hierarchy is
materialized entirely as `HAS_SUBSECTION` edges, so an agent (or a Cypher query)
walks the tree with ordinary graph traversals. A section's `text` is its own content
only (non-overlapping), so concatenating every section of a document in `sequence`
order reconstructs the original Markdown exactly.

**Identity and dedup:** `Document` and `MarkdownSection` are both keyed by content
hash, so re-ingesting unchanged files is a no-op MERGE. When an entity-extraction
factory (see below) also produces a `Document` node for the same file, it MERGEs
into the *same* node — a document's provenance and its extracted entities share one
graph node.

There are no `Chunk` nodes in the Document Graph — no chunking or embeddings are
produced. Chunking/embedding based RAG is a separate, unrelated path (see
[`DocumentDirectoryFactory`](#documentdirectoryfactory) below); the Document Graph is
for heading-based, vectorless navigation.

## Factories

| Factory | Produces | Use it when |
|---------|----------|--------------|
| `genai_graph.kg.factories.document_graph_factory.DocumentGraphFactory` | `Folder → Document → MarkdownSection` | You want the navigable heading hierarchy over a Markdown corpus. |
| `genai_graph.kg.factories.document_factory.DocumentDirectoryFactory` | Plain `Document` nodes | You just need file-level provenance (no sections), e.g. as a base class for a custom RAG pipeline. |
| `genai_graph.kg.factories.markdown_baml_factory.MarkdownBamlFactory` | Your entity nodes + a provenance `Document` node (`MENTIONS` relation) | You want to extract structured entities (Opportunity, Risk, Person, …) from Markdown via a BAML function, run inline (no separate `cli baml extract` step). |
| `genai_graph.kg.factories.json_factory.JsonFileBackedFactory` | Your entity nodes + a provenance `Document` node | Your entities are *already* extracted as JSON files (e.g. produced by a prior `cli baml extract` run), one directory per model name. |

### `DocumentGraphFactory`

```python
from genai_graph.kg.backend import KuzuBackend
from genai_graph.kg.document_graph.ingest import ingest_document_graph
from genai_graph.kg.factories.document_graph_factory import DocumentGraphFactory

backend = KuzuBackend()
backend.connect("./data/kg/tree.db")

factory = DocumentGraphFactory(sources=["./docs"], include=["*.md"])
result = ingest_document_graph(backend, factory)
```

`sources` accepts a mix of directories, individual files, and `.zip` archives — each
becomes a `Folder`. `ingest_document_graph` builds the schema, parses each file's
heading hierarchy, and MERGEs everything in one call; pass `force=True` to rebuild
sections for documents already in the graph (e.g. after a heading edit).

### `MarkdownBamlFactory` (inline BAML entity extraction)

Subclass it to extract structured entities directly from Markdown, without a
separate JSON-extraction step. Extracted results are cached as JSON build artifacts
(keyed by mtime) so re-runs are cheap:

```python
from genai_graph.kg.factories import MarkdownBamlFactory
from genai_graph.kg.schema import GraphSchema
from pydantic import BaseModel


class MyEntityGraph(MarkdownBamlFactory):
    def build_schema(self) -> GraphSchema:
        nodes, relations = self.get_document_schema_elements(MyEntityNode)
        return GraphSchema(root_model_class=MyEntity, nodes=[MyEntityNode, *nodes], relations=relations)

    def extract_from_markdown(self, md_text: str) -> BaseModel:
        from genai_tk.extra.structured.baml_processor import BamlStructuredProcessor

        processor = BamlStructuredProcessor(model_cls=MyEntity, function_name="ExtractMyEntity", kvstore_id="")
        return processor.analyze_document("doc", md_text)
```

`md_root` selects the Markdown directory to scan; `json_cache_root` (optional)
enables the JSON extraction cache. Because `MarkdownBamlFactory` mixes in
`DocumentMixin`, calling `get_document_schema_elements(root_node)` in `build_schema()`
adds the provenance `Document` node and a `MENTIONS` relation from `Document` to your
entity's root node — the same `Document` node the `DocumentGraphFactory` produces for
the same file, so both MERGE together.

## Building a KG from documents end to end

`docgraph_build_step` (`genai_graph.orchestration.workflow_steps.docgraph_build_step`)
ties markdownization, entity extraction, and the document graph together into **one**
database:

1. Optionally markdownize `sources` (PPT/PDF/… or pre-existing Markdown — already-
   Markdown files are copied through unchanged) via `genai_tk.workflow.markdownize.markdownize_flow`.
2. Run each configured entity `factory` (e.g. a `MarkdownBamlFactory` subclass) into
   a single KG named `kg_name`.
3. Optionally ingest the `Folder → Document → MarkdownSection` graph over the same
   Markdown into the *same* database.

This is what backs both `cli docgraph run` (ad-hoc sources) and a project's own named
workflow profiles consumed via `cli kg create <name>` — see
[docs/workflows.md](workflows.md) for the full workflow-engine reference and force-stage
semantics.

## Summarization

Every section carries a one-sentence `description` (the *routing* signal an agent
scans to pick a section) and, for substantial sections, a short-paragraph `summary`
(the *triage* signal — "is this worth opening?"), and the `Document` node gets its
own description and summary. There are two ways to populate them.

### During build: `cli docgraph build --llm`

Pass `--llm` to `docgraph build` and a (typically cheap, large-context "flash")
model discovers each document's structure **and** summarizes its sections in a
single call. It emits a content-free JSON outline — the table of contents plus a
description of every section (and a summary of the substantial ones) — *without*
re-emitting section text, so the output stays small on million-token documents.
The outline is cached by `markdown_hash` (and a policy/LLM hash, so re-runs are
free), and a deterministic merge anchors its titles back to the Markdown to build
the actual section nodes.

```bash
# Algorithmic build only — fast, no LLM, no descriptions
cli docgraph build ./docs --db ./data/kg/tree.db

# LLM-enhanced build: structure + descriptions + summaries in one call per doc
cli docgraph build ./docs --db ./data/kg/tree.db --llm flash
cli docgraph build ./docs --db ./data/kg/tree.db --llm gpt_4o_mini@edenai --workers 4
cli docgraph build ./docs --db ./data/kg/tree.db --llm flash --summary-min-tokens 400
cli docgraph build ./docs --db ./data/kg/tree.db --llm flash --llm-max-tokens 32000
```

`--llm` takes a literal `name@provider` id or a config tag resolved from
`kg_build.llms.<tag>` (e.g. `flash`, `default`). Two failure modes degrade
gracefully — the document is still ingested, just without descriptions/summaries,
and a warning is surfaced: a document exceeding `context_safety_ratio` (default
0.9) of the model's context window makes no LLM call at all; an LLM call failure
(or no outline title matching the Markdown) falls back to the algorithmic
structure. The result table reports `Sections summarized` and
`Files degraded to algo`.

### After build: the library / workflow step

`genai_graph.kg.document_graph.summarize` annotates an already-ingested graph —
use it when a graph was built algorithmically and you want to enrich it without
rebuilding sections:

```python
from genai_graph.kg.document_graph.summarize import SummarizationConfig, summarize_graph

# summarize_graph opens its own Ladybug Database and shares it across worker
# threads (each document's rows are disjoint, so writes don't conflict). Pass
# the db path and a worker count — not a backend.
result = summarize_graph(
    "./data/kg/tree.db",
    SummarizationConfig(summary_min_tokens=800),
    workers=4,
)
```

For an explicit document list (by hash, prefix, or filename), use
`summarize_documents(db_path, document_ids, config, *, force=False, dry_run=False,
workers=1)`; references sharing a `markdown_hash` are de-duplicated so the same
rows are never written from two concurrent transactions.

**Two fields, two jobs.** `description` is one plain-text sentence and is always
present — it is the *routing* signal an agent scans to choose a section, and it must
stay short or the table of contents stops fitting in a prompt. `summary` is a short
paragraph, generated only for the document itself and for sections at or above
`summary_min_tokens` — the *triage* signal ("is this worth opening?").

**Selection policy:** every heading section down to `max_level` gets a description —
a table of contents with holes in it is not navigable. The synthetic level-0 root
section is skipped. A Markdown table inside a section is reduced to its header row
plus a sample of data rows before being sent to the LLM, and the prompt asks for the
table's shape and coverage rather than its contents.

**Call strategy:** the document is sent once, annotated with `[[section_id]]` markers
before each section, and the model returns one entry per requested id. Sending the
annotated document rather than the full text *plus* a copy of every section's text
roughly halves input tokens and removes any ambiguity about which text belongs to
which id. Only when the *projected* output would exceed `output_budget_tokens` is the
id list split across several calls. The document's token count is checked against the
target LLM's context window and a warning is added (not raised) when it is tight —
this is an *input*-token check only.

**Brevity is enforced three times over**, because models routinely ignore a stated
word limit: in the prompt, in the response-schema field descriptions, and finally by
`_clean_text()`, which strips Markdown (headings, bullets, emphasis), drops `Page N`
conversion artifacts, flattens to a single line, and hard-truncates at a sentence
boundary (`max_description_chars` / `max_summary_chars`).

**Reasoning models and the output token budget:** a separate, unrelated failure
mode is a reasoning model spending its whole *completion* budget on hidden
reasoning tokens, leaving none for the actual answer — LangChain surfaces this as
`openai.LengthFinishReasonError` ("Could not parse response content as the length
limit was reached"). This has nothing to do with the input context window (a 1M
context model can still hit it). `summarize_document`/`summarize_graph` retry once
automatically with a larger `max_tokens` (`SummarizationConfig.retry_max_tokens`,
default 32000) on this specific failure; set `SummarizationConfig.llm_max_tokens`
(or `--llm-max-tokens`) to raise the budget from the first call instead of waiting
for the retry.

Summarization is resumable and idempotent: `summarize_document`/`summarize_graph`
skip a document that already has every section described unless `force=True`, and
`dry_run=True` computes the plan (selection, batching, warnings) without calling the
LLM or writing to the graph.

A run streams progress via loguru as it happens — per-batch "calling LLM"/"done
(Xs)" lines from each worker, retries and failures logged immediately, and a final
aggregate summary. With `workers > 1` documents complete in a nondeterministic
order, so the old sequential `[i/N]` per-document counter is gone, but no run goes
silent until the final summary table, even across many documents.

> The `cli docgraph summarize` command was removed. Use `cli docgraph build --llm`
> (above) to produce sections and summaries during ingest, or call
> `summarize_graph` / the `document_graph_summarize_step` workflow step to enrich
> an already-built graph.

## Navigating as an agent

The tools from `create_document_graph_tools(db_path)` are designed for **progressive
disclosure** — each step narrows the scope, so nothing ever dumps a whole corpus into
the context window:

1. `get_folder_toc()` — the orientation view. Documents only (id, name, section count,
   one-line description); **no sections**. A 16-document corpus is a few hundred tokens.
2. `get_document_toc(document_id)` — the chosen document's section tree, each node with
   `id`, `title` and `description`. `include_summaries=True` adds the paragraph
   summaries; `max_level` prunes depth on a very long document. Heading level and token
   count are deliberately not emitted — level is redundant with the tree's own nesting,
   and token count adds nothing an agent can act on once `description` already answers
   "is this worth opening?".
3. `get_section_content(section_ids)` — the actual Markdown, for the few sections the
   agent picked.

`search_sections(keyword)` is the keyword fallback when the descriptions don't surface
an obvious candidate.

```yaml
# get_folder_toc()
documents:
  - id: 7edf6684
    name: 3. Service description.md
    sections: 44
    description: Managed workplace and infrastructure services scope for Alko.

# get_document_toc("7edf6684")
document: 3. Service description.md
id: 7edf6684
description: Managed workplace and infrastructure services scope for Alko.
sections:
  - id: 7edf6684::12
    title: 4 Infrastructure services
    description: Server, network, database and backup operations scope.
    sections:
      - id: 7edf6684::60
        title: Database services
        description: Database monitoring, capacity, maintenance, backup and patch duties.
```

## CLI

`cli docgraph` is the generic, document-focused command group:

```bash
# Markdownize + build the document graph directly on a Ladybug DB
cli docgraph build ./docs --db ./data/kg/tree.db
cli docgraph build ./RFQ.zip --db ./data/kg/tree.db --profile fast

# Run a project-defined docgraph_build workflow (markdownize + entity factories + document graph)
cli docgraph run --workflow rainbow_extract -s "some_file.pptx"

# Navigate an ingested graph
cli docgraph list --db ./data/kg/tree.db
cli docgraph toc <filename-or-hash> --db ./data/kg/tree.db
cli docgraph toc <filename-or-hash> --db ./data/kg/tree.db --yaml   # section tree with descriptions
cli docgraph toc <filename-or-hash> --db ./data/kg/tree.db --yaml --summaries --max-level 2
cli docgraph toc <folder-hash-or-name> --db ./data/kg/tree.db --yaml  # --yaml also works on a folder ref
cli docgraph folder-toc --db ./data/kg/tree.db                     # orientation view: documents only
cli docgraph folder-toc --db ./data/kg/tree.db --sections          # inline each document's section tree
cli docgraph cat <filename-or-hash> --db ./data/kg/tree.db
cli docgraph search "keyword" --db ./data/kg/tree.db
cli docgraph tui --db ./data/kg/tree.db

# LLM-enhanced build: structure + descriptions + summaries in one call per document
cli docgraph build ./docs --db ./data/kg/tree.db --llm flash
```

`cli kg create <name>` runs the same workflow engine against a **predefined** set of
documents (a named workflow profile in a project's `config/workflows/*.yaml`), while
`cli docgraph run` targets **ad-hoc** sources passed with `-s`/`--source`. Both are
thin CLI layers over `resolve_workflow_invocation` + `execute_workflow`.

## Querying

`genai_graph.kg.query.document_graph_tools` provides read-only helpers used by the
CLI, an agent's tools, and the Textual TUI:

```python
from genai_graph.kg.query.document_graph_tools import (
    list_documents,
    get_document_toc,
    get_section_content,
    reconstruct_document,
    search_sections,
    document_toc_yaml,
    folder_toc_yaml,
)
```

Documents can be addressed by content hash (full or prefix), `markdown_hash`,
filename, or source path. `document_toc_yaml`/`folder_toc_yaml` render a document's
(or a folder subtree's) heading hierarchy as YAML, including summaries once
generated — the format an agent tool returns. `create_document_graph_tools(db_path)`
wraps these (plus `get_folder_toc`) as LangChain `BaseTool`s for an agent.

## Related docs

- [docs/workflows.md](workflows.md) — the workflow DSL, force stages, `cli docgraph`/`cli kg create` CLI reference
- [docs/graph-definition-guide.md](graph-definition-guide.md) — defining a `GraphSchema` from Pydantic models (entity extraction, not the document graph itself)
- [docs/graph-authoring-patterns.md](graph-authoring-patterns.md) — pattern catalog including document ingestion
- [docs/baml_extraction_guide.md](baml_extraction_guide.md) — BAML schema → entity graph factory patterns
