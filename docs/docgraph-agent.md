# Document Graph Deep Agent

The **Document Graph Deep Agent** is a LangChain `type: deep` agent (DeepAgents SDK)
that answers questions about an ingested document corpus by **navigating** the
Ladybug Document Graph — `Folder → Document → MarkdownSection` — with read-only
Cypher-backed tools. It is deliberately **vectorless**: no embeddings, no
chunking. The agent orients with a folder listing, gets a document's section map,
reads only the sections whose description matches the question, and iterates.

This document covers the architecture, the public API, the runtime skills, the
CLI, how a downstream project wires a use-case extraction, and the design
decisions behind the schema-tolerant tools.

## Why a deep agent over the Document Graph

A typical RAG pipeline chunks documents, embeds the chunks, and retrieves by
similarity. The Document Graph already exposes a navigable heading hierarchy with
per-section routing `description`s, so an agent can do the retrieval *itself*:
treat the table of contents as the map, and `get_section_content` as the fetch.
This keeps provenance exact (every answer cites a section id), avoids a separate
embedding/index pipeline, and works on a corpus the moment it is ingested.

A **deep** agent (planning + tools + skills via the DeepAgents SDK) is the right
shape for the complex-query case: it plans across many documents, reads several
sections, and compiles a structured report — exactly the RFQ extraction workload.

## Architecture

```
                ┌──────────────────────────────────────────────┐
                │            cli docgraph agent                 │
                │   (genai_graph/core/commands_docgraph.py)    │
                └──────────────────────┬───────────────────────┘
                                       │ loads
                                       ▼
   ┌─────────────────────────────────────────────────────────────┐
   │  genai_graph/agent/docgraph_agent.py                         │
   │  ┌───────────────────────┐   ┌───────────────────────────┐  │
   │  │ prepare_docgraph_     │   │ create_docgraph_agent     │  │
   │  │ profile               │   │  → LangChainHarness        │  │
   │  │  • system prompt      │   │  (extra_tools = nav tools) │  │
   │  │  • skill dirs         │   └─────────────┬─────────────┘  │
   │  │  • filesystem backend │                 │ compiles lazily │
   │  └──────────┬───────────┘                 ▼                │
   │             │           deepagents.create_deep_agent        │
   │             │  + SkillsMiddleware (progressive disclosure)  │
   └─────────────┼───────────────────────────────────────────────┘
                 │ calls
                 ▼
   ┌─────────────────────────────────────────────────────────┐
   │  genai_graph/kg/query/document_graph_tools.py           │
   │  get_folder_toc │ get_document_toc │ get_section_content │
   │  search_sections │ list_documents                       │
   │  (schema-tolerant: introspect columns + rels)           │
   └────────────────────────┬────────────────────────────────┘
                            │ Cypher (read-only)
                            ▼
                   Ladybug Document Graph DB
```

### Runtime injection, not profile wiring

The navigation tools depend on the target database path and (optionally) a
folder. Those are runtime concerns — a YAML profile can't know them. So the
profile (`config/agents/docgraph.yaml`) holds only *structure*: `type: deep`,
`llm`, `system_prompt`, `skill_directories`, `recursion_limit`, `tools: []`.
`create_docgraph_agent` injects the tools at runtime via the harness
`extra_tools`, and `prepare_docgraph_profile` sets the system prompt (scoped to
the folder), the skill directories, and the filesystem backend. `--db`,
`--folder`, and `--llm` overrides therefore work without editing the profile.

### Skills (progressive disclosure)

Two runtime skills ship **co-located** with the agent code, under
`genai_graph/agent/skills/`:

- `navigate-document-graph` — the navigation loop (orient → map → read → search →
  iterate) and the grounding/citation rules.
- `document-graph-tools` — exact tool arguments, return shapes, and the
  `Folder → Document → MarkdownSection` schema.

They are resolved from the **package location** (`Path(__file__).parent/"skills"`)
so they load whether the agent is launched from genai-graph or a downstream
project. DeepAgents' `SkillsMiddleware` reads them through a `FilesystemBackend`;
because the generic skills live in the `genai_graph` package while a downstream
project's skills live in its own tree, `prepare_docgraph_profile` roots the
backend at the **common ancestor** of all skill directories — that keeps
`virtual_mode=True` path-traversal checks satisfied across project boundaries.

## Public API

```python
from genai_graph.agent import (
    create_docgraph_agent,  # profile -> LangChainHarness (tools + skills + folder injected)
    prepare_docgraph_profile,  # mutate profile in place (system prompt, skills, backend)
    run_docgraph_agent,  # async: stream one turn, return assistant text
    resolve_db_path,  # db_path or graph_db.default
    create_document_graph_tools_from_config,
    build_docgraph_system_prompt,
    DEFAULT_LLM,  # "deepseek_v4flash@openrouter"
    DEFAULT_PROFILE,  # "docgraph"
)
```

`create_docgraph_agent(profile, *, llm=None, db_path=None, folder_id=None,
extra_skill_dirs=None) -> LangChainHarness`. The harness lazily compiles the
deep agent on first use. Stream a one-shot turn with `run_docgraph_agent`, or run
an interactive REPL with
`genai_tk.agents.harness.chat_repl.run_chat_repl(harness)`.

## CLI

```bash
# one-shot
cli docgraph agent "What IT services is Alko requesting?" --folder folder_273e65da416b2e72

# interactive multi-turn REPL (memory enabled)
cli docgraph agent --chat --folder folder_273e65da416b2e72

# overrides
cli docgraph agent "Summarize the SLAs" --llm deepseek_v4flash@openrouter --db ./data/kg/tree.db
cli docgraph agent "..." --profile docgraph --skill-dir ./extra-skills --recursion-limit 160
```

Flags: `--profile/-p` (default `docgraph`), `--llm/-m`, `--db`, `--folder`,
`--skill-dir` (repeatable), `--recursion-limit` (default 120), `--chat`,
`--trace`.

## Schema-tolerant tools

`document_graph_tools.py` is the foundation. Older Ladybug databases (e.g. one
ingested before `Folder.parent_folder_id`, `HAS_SUBFOLDER`, or the
`description`/`summary` columns existed) are common in the wild. The tools now:

- introspect each table's columns once via `CALL table_info('<table>')` and the
  relationship set via `CALL show_tables()` (cached per backend);
- build `RETURN` clauses from the columns that are **actually present**, so a
  missing `description`/`summary` is simply omitted rather than crashing;
- fall back to **flat** folder navigation when `HAS_SUBFOLDER` is absent;
- return `[]` / `None` (and a clear "No ... found" string at the tool layer) when
  a table is genuinely missing, and raise a plain-English `DocumentGraphError`
  when a table exists but is malformed.

This is what lets the same agent run against the full-schema default DB
(`${paths.kg_outputs}/rfq_pricing.lbdb`) and degrade gracefully on a partial
older DB — no binder tracebacks reach the LLM or the user.

## LLM

The default model is `deepseek_v4flash@openrouter` (OpenRouter slug
`deepseek/deepseek-v4-flash`: tool-calling, 1M-token context). It is registered
in `config/providers/llm.yaml` and referenced explicitly as
`deepseek_v4flash@openrouter` in the profiles — the **bare** `deepseek_v4flash`
id fuzzy-resolves to Azure, so always use the `@openrouter` form. The `--llm` flag
accepts any registered id or tag.

## Wiring a downstream project (e.g. rfq_pricing)

1. Add `deepseek_v4flash` to the project's `config/providers/llm.yaml`
   (openrouter slug `deepseek/deepseek-v4-flash`).
2. Add a unified `agents:` profile (`type: deep`) in `config/agents/<name>.yaml`
   with the use-case `system_prompt` and `skill_directories` pointing at the
   project's use-case skills (e.g. `${paths.project}/skills/custom`).
3. Write a use-case runtime skill (e.g. `skills/custom/rfq-extraction/SKILL.md`)
   describing the information categories to collect and a corpus-specific
   navigation strategy (which sections/keywords map to which category).
4. In a CLI command, call `create_docgraph_agent(profile, llm=..., db_path=...,
   folder_id=..., extra_skill_dirs=[...])` then `run_docgraph_agent`. The generic
   `genai_graph/agent/skills` are injected automatically; pass the project's
   use-case skills via `extra_skill_dirs` or list them in the profile.
5. Install the `harnessing` extra (`uv sync --extra harnessing`) — `deepagents`
   is required for `type: deep` agents.

### rfq_pricing example

`cli agent extract` loads the `rfq-extract` profile, scopes the agent to a
folder, and runs an extraction query. The agent navigates every document in the
folder, reads the relevant sections for each of 9 RFP categories (client
background, scope, budget, SLAs, timeline, evaluation, compliance,
collaboration, risks), and emits a structured Markdown report with
`[hash::sequence]` citations, written to `data/rfq_extraction/<folder_id>.md`.

## Design decisions & trade-offs

- **Deep, not react.** Multi-document extraction needs planning and iteration;
  a single-pass ReAct loop stops too early. The DeepAgents SDK's planning + skills
  progressive disclosure fits.
- **Runtime tool injection over profile tools.** The tools close over `db_path`;
  baking them into the YAML would block `--db` overrides. Injecting via
  `extra_tools` keeps the profile portable.
- **Co-located runtime skills.** They ship with the package and resolve from
  `__file__`, so downstream projects get them for free. The developer-facing
  skills stay under `skills/genai-graph/` (for editing the codebase) — the two
  audiences are separate.
- **Common-ancestor backend root.** The only way to load skills from two
  different project trees through one `FilesystemBackend(virtual_mode=True)`
  is to root it at their shared ancestor. Cheap to compute, works everywhere.
- **Vectorless.** The Document Graph's heading hierarchy + section descriptions
  are the retrieval index. Adding embeddings/Chunk nodes would duplicate
  provenance and break the "cite the section you read" guarantee.
- **Schema-tolerant tools.** Real deployments have mixed-schema DBs. Crashing on
  a missing column would make the agent unusable on older corpora; the hardening
  makes the same code path serve both.

## Testing

- `tests/integration_tests/test_document_graph_ingest.py` — covers the
  schema-tolerant behavior (empty DB → `[]`, dropped Section table → `[]`,
  full-schema navigation, folder hierarchy, hash-prefix resolution).
- Manual smoke: `cli docgraph agent "..." --folder folder_273e65da416b2e72`.
- Real extraction: `cli agent extract` (rfq_pricing) produces the 9-category
  report cited above.

```bash
uv run pytest tests/integration_tests/test_document_graph_ingest.py -q
uv run ruff check genai_graph/agent genai_graph/kg/query/document_graph_tools.py
```

## See also

- `kg-document-graph` skill — the Document Graph schema, factories, and tools.
- `kg-query` skill — the broader Cypher / text-to-Cypher story.
- `kg-docgraph-agent` skill — developer guide to this agent module.
- `genai-tk/agent-profiles` — the unified `agents:` profile format and `type: deep`.
