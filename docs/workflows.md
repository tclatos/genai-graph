# Workflows in genai-graph: Knowledge Graph Orchestration

The **Workflow Engine** (from genai-tk) orchestrates knowledge graph creation and
document pre-processing pipelines. Workflow definitions live in
`config/workflows/*.yaml` using the standard workflow DSL — the same DSL genai-tk
uses for RAG ingestion, anonymization, and BAML extraction.

genai-graph itself only ships **domain-agnostic** workflows (document conversion, a
generic single-factory KG build primitive, generic document graphs). Domain-specific
workflows (data paths, concrete graph factories) belong to the project that imports
genai-graph — see `ekg-atos/config/workflows/` for a worked example (rainbow/RFQ
presets, multi-factory KG builds). For the Document Graph schema and factories
themselves (as opposed to the workflow layer covered here), see
[docs/document-graph.md](document-graph.md).

> **Note:** genai-graph does not ship `config/workflows/*.yaml` as installed package
> data — it's dev-only config, discovered only when running `cli` from inside the
> genai-graph repo itself. A downstream project (ekg-atos, rfq_pricing) that depends
> on genai-graph as a package must define its own full workflow entries (`run:` /
> `defaults:` / `params:`) for `office2pdf_documents` / `markdownize_documents`, not
> just presets — see `ekg-atos/config/workflows/data_injection.yaml` for the pattern.

**Key benefits:**
- Single CLI entry point: `cli workflow run` for all pipelines
- `--dry-run` to see the full step plan before executing
- `--set KEY=VAL` to override any parameter inline
- Sub-workflow composition: a pipeline step's `run:` can reference another
  workflow name — it's inlined automatically (step IDs prefixed `{step_id}.`)
- A single ordered `--force <stage>` replaces ad-hoc `force` / `force_rebuild` /
  `--remarkdownize` booleans (see [Force stages](#force-stages) below)

---

## Quick Start

### List available workflows

```bash
uv run cli workflow list
# genai-graph itself (generic): document_nodes, document_graph,
#                                markdownize_documents, office2pdf_documents
# A downstream project (e.g. ekg-atos) additionally defines: one_rainbow,
#                                rainbow_add_crm, stratnav_subset, full_kg_pipeline, ...
```

### Run a workflow

```bash
# Dry-run: resolve the workflow, show the plan
uv run cli workflow run document_to_kg --dry-run

# Markdownize a directory, zip archive, or file — always one call, no separate
# unzip / office2pdf step (markdownize_flow does that internally)
uv run cli workflow run markdownize_documents --set sources=./RFQ.zip --set md_output_dir=./out/md

# Override values inline
uv run cli workflow run document_to_kg --set sources=./docs --set md_output_dir=./md --set graph='{...}'
```

### Or use `cli docgraph build` / `cli kg create` (high-level shorthands)

```bash
# Markdownizes ./RFQ.zip (or a plain folder) then ingests into the Document Graph
cli docgraph build ./RFQ.zip --db ./data/kg/tree.db --profile fast

# Re-run just the Markdown conversion (and everything downstream of it)
cli docgraph build ./RFQ.zip --db ./data/kg/tree.db --force md

# Build a KG profile (downstream-project-defined, e.g. ekg-atos's one_rainbow)
cli kg create one_rainbow

# Force rebuild of parquet import caches (and downstream graph/embed stages)
cli kg create one_rainbow --force parquet

# Full clean rebuild, dropping the destination database first
cli kg create one_rainbow --force all
```

---

## Force stages

A single ordered `--force <stage>` replaces the old collection of ad-hoc booleans
(`force`, `force_rebuild`, `--remarkdownize`). **Forcing a stage re-runs it and
everything downstream of it**, since downstream caches are derived from upstream
outputs. Defined in `genai_tk.workflow.force.ForceStage`:

| Stage     | Effect                                                              | Relevant commands |
|-----------|----------------------------------------------------------------------|--------------------|
| `unzip`   | Re-extract `.zip` archives even if already cached                    | `markdownize_documents`, `docgraph build` |
| `pdf`     | Re-run Office → PDF conversion                                       | `markdownize_documents`, `docgraph build` |
| `md`      | Re-run document → Markdown conversion                                | `markdownize_documents`, `docgraph build` |
| `parquet` | Rebuild JSON → parquet import caches                                 | `kg create`, `kg_build` |
| `graph`   | Re-ingest into the graph database (drops the destination store)      | `docgraph build`, `kg create`, `kg_build` |
| `embed`   | Recompute embeddings                                                 | `docgraph build`, `kg create` |
| `all`     | Force every stage, including dropping the destination store          | all of the above |

`--delete-first` remains a separate, explicit DB-lifecycle flag (independent of
`--force`) — use it when you want a clean database without forcing any upstream
cache to be recomputed.

```bash
# Old (removed): --set delete_first=true / --set force_rebuild=true
# New:
cli kg create one_rainbow --delete-first          # drop DB, reuse all upstream caches
cli kg create one_rainbow --force parquet          # rebuild import caches (implies graph rebuild)
cli kg create one_rainbow --force all              # full clean rebuild
uv run cli workflow run <kg-profile> --force graph # equivalent via the low-level command
```

---

## Architecture

### Config files (genai-graph itself)

| File | Purpose |
|------|---------|
| `config/workflows/generic_workflows.yaml` | `kg_build` (single-factory primitive, hidden), `document_nodes`, `document_graph` |
| `config/workflows/data_injection.yaml` | `office2pdf_documents`, `markdownize_documents`, `document_to_kg` (generic doc→KG pipeline) |

All are merged into the global config via `:merge:` in `app_conf.yaml`, so all
workflows are available from a single `cli workflow` command.

### The `kg_build_step` function

The generic `kg_build` workflow calls `genai_graph.orchestration.workflow_steps.kg_build_step`, which:

1. Accepts a single `graph` factory config (dict with a `factory` key) + `kg_name`
2. Registers it as a temporary profile in the `KgManager` singleton
3. Clears factory caches (prevents cross-contamination)
4. Runs the full `create_kg_flow()` Prefect pipeline
5. Returns `{config_name, total_processed, total_failed, warnings_count, db_path}`

For multi-factory graphs (several JSON sources merged into one KG), a downstream
project chains several `kg_build`-based pipeline steps with `after:` — each targeting
the same `kg_name` — so all factories MERGE into one database (see ekg-atos's
`graph_construction.yaml` — `rainbow_add_crm` combines three factories this way).

### The `docgraph_build_step` function

`genai_graph.orchestration.workflow_steps.docgraph_build_step` is the building block
for document-driven KGs — markdownize sources, run entity-extraction factories, and
build the [Document Graph](document-graph.md) into one database:

1. Optionally dispatches `sources` through the ingest route table into
   `md_output_dir` when `routes` is set (rule-selected workflows: markdownize,
   web-page fetching, …).
2. Runs each configured entity `factories` (e.g. a `MarkdownBamlFactory` subclass)
   into a single KG named `kg_name`.
3. Optionally ingests the `Folder → Document → MarkdownSection` graph over the same
   Markdown into the *same* database (`build_document_graph`, default `true`) —
   Document nodes MERGE by content hash with the ones the entity factories create.

A project wires this up as its own workflow (like `kg_build`, it isn't a YAML
workflow shipped by genai-graph itself):

```yaml
# a project's config/workflows/*.yaml
workflows:
  docgraph_build:
    run: genai_graph.orchestration.workflow_steps.docgraph_build_step
    hidden: true
    defaults:
      routes: default            # ingest route table (config/ingest_routes.yaml)
      build_document_graph: true
      delete_first: false
      export_html: true
    params:
      kg_name: {required: true}
      sources: {required: true}
      md_output_dir: {required: true}
      factories: {required: true}

  rainbow_extract:
    description: "Rainbow: PPT/PDF/Markdown -> opportunity entities + document graph"
    pipeline:
      - id: build
        run: docgraph_build
        with:
          kg_name: rainbow_extract
          sources: '${values.sources}'
          md_output_dir: '${paths.rainbow_md}'
          factories:
            - factory: myproject.schema.rainbow_review.ReviewedOpportunityGraph
              md_root: '${paths.rainbow_md}'
              json_cache_root: '${paths.rainbow_json}'
    defaults:
      sources: '${paths.rainbow_ppt}'
```

---

## Workflow Definitions

### Generic document → Markdown → KG pipeline

```yaml
workflows:
  document_to_kg:
    description: "End-to-end pipeline: documents (zip/dir/files) -> Markdown -> single-factory KG build"
    pipeline:
      - id: markdownize
        run: markdownize_documents
        with:
          sources: "${values.sources}"
          md_output_dir: "${values.md_output_dir}"
          cache_dir: "${values.cache_dir}"
          profile: "${values.profile}"
          force_stage: "${values.force_stage}"
      - id: create_kg
        run: kg_build
        after: [markdownize]
        with:
          graph: "${values.graph}"
          kg_name: "${values.kg_name}"
          force_stage: "${values.force_stage}"
    params:
      sources: {required: true}
      md_output_dir: {required: true}
      graph: {required: true}
```

`markdownize_documents` handles directories, `.zip` archives, and individual files —
raw Office/PDF/image documents *or* pre-existing Markdown (copied through
unchanged) — in one step. There is no separate Office→PDF step in the generic
pipeline; `markdownize_flow` converts Office → PDF → Markdown internally.

### Domain-specific multi-factory workflows (example: ekg-atos)

Each pipeline step targets the same `kg_name` via `kg_build`, so every factory's
nodes/relations MERGE into one database:

```yaml
# ekg-atos/config/workflows/graph_construction.yaml
workflows:
  rainbow_add_crm:
    description: "Rainbow reviews + architecture docs + CRM export"
    pipeline:
      - id: rainbow
        run: kg_build
        with:
          kg_name: rainbow_add_crm
          graph:
            factory: ekg_atos.schema.rainbow_review.ReviewedOpportunityGraph
            md_root: '${paths.rainbow_md}'
            json_cache_root: '${paths.rainbow_json}'
            include: ['*CNES*']
      - id: arch_doc
        run: kg_build
        after: [rainbow]
        with:
          kg_name: rainbow_add_crm
          delete_first: false
          graph:
            factory: ekg_atos.schema.architecture_doc.ArchitectureDocumentGraph
            data_root: '${paths.add_json}'
      - id: crm
        run: kg_build
        after: [arch_doc]
        with:
          kg_name: rainbow_add_crm
          delete_first: false
          graph:
            factory: ekg_atos.schema.crm_export.CrmExtractGraph
            files: ['${paths.ekg_data}/crm_export/report.xlsx']
```

### Sub-workflow composition

A pipeline step's `run:` can reference another workflow name directly — it is
expanded in place, with its steps prefixed `{step_id}.`:

```yaml
workflows:
  full_kg_pipeline:
    pipeline:
      - id: markdownize
        run: markdownize_documents
        with: {...}
      - id: create_kg
        run: rainbow_add_crm      # another workflow, expands to create_kg.build
        after: [markdownize]
```

```bash
cli workflow run full_kg_pipeline/rainbow --dry-run

   Id                  │ Invoke              │ Wait For
───────────────────────┼─────────────────────┼──────────────────
   markdownize.run      │ ...markdownize_flow │ -
   create_kg.build      │ ...kg_build_step    │ markdownize.run
```

---

## CLI Reference

### `cli workflow run` (low-level, full control)

```bash
uv run cli workflow run document_to_kg --dry-run
uv run cli workflow run document_to_kg
uv run cli workflow run document_to_kg --force md
uv run cli workflow run document_to_kg --set export_html=false
```

### `cli kg create` (high-level shorthand)

`cli kg create` wraps `cli workflow run <profile>` with convenient flags. CLI flag
values **always override** workflow YAML defaults.

```bash
cli kg create one_rainbow                       # build (or update) the KG
cli kg create one_rainbow --force parquet       # force rebuild of import caches
cli kg create one_rainbow --force all           # full clean rebuild
cli kg create one_rainbow --no-export-html      # skip HTML export (faster)
cli kg create one_rainbow --no-delete-first     # keep existing database (merge/upsert)
cli kg create one_rainbow --dry-run             # resolve the plan only
cli kg create one_rainbow --set export_html=false
cli kg create --all                             # run every kg_* profile
```

### `cli docgraph run` (ad-hoc sources, project-defined workflow)

Runs a project's `docgraph_build`-based workflow (e.g. `rainbow_extract`) against
either its configured default sources or ad-hoc ones passed with `-s`:

```bash
cli docgraph run --workflow rainbow_extract                         # configured default sources
cli docgraph run -w rainbow_extract -s ./some_file.pptx              # ad-hoc source override
cli docgraph run -w rainbow_extract -s ./ppt --force md              # re-run markdown conversion
cli docgraph run -w rainbow_extract --dry-run                        # resolve the plan only
cli docgraph run -w rainbow_extract --set export_html=false
```

`cli kg create <name>` and `cli docgraph run --workflow <name>` both resolve through
`resolve_workflow_invocation` + `execute_workflow` — use `kg create` for a named,
predefined document set, and `docgraph run` when you want to point at a file/folder
ad hoc.

### `cli docgraph build` (Document Graph only, no entity extraction)

Always markdownizes its sources first, then ingests into the graph database:

```bash
cli docgraph build ./docs --db ./data/kg/tree.db
cli docgraph build ./RFQ.zip --db ./data/kg/tree.db --routes default   # ingest route table
cli docgraph build ./RFQ.zip --db ./data/kg/tree.db --md-output-dir ./out/md --cache-dir ./out/.cache
cli docgraph build ./RFQ.zip --db ./data/kg/tree.db --force md      # re-run markdown conversion
cli docgraph build ./RFQ.zip --db ./data/kg/tree.db --force graph   # re-ingest, reuse markdown cache
cli docgraph build ./RFQ.zip --db ./data/kg/tree.db --delete-first  # drop Section tables first
```

---

## Adding a New KG Config

Add a workflow to your project's `config/workflows/*.yaml`:

```yaml
workflows:
  my_new_kg:
    description: "My new knowledge graph"
    run: genai_graph.orchestration.workflow_steps.kg_create_step
    defaults:
      config_name: my_new_kg
      delete_first: false
      export_html: true
```

Then run:

```bash
uv run cli workflow run my_new_kg --dry-run
uv run cli workflow run my_new_kg
# or:
cli kg create my_new_kg
```

---

## Troubleshooting

### Data directory not found

```
WARNING | json_factory.py - Data root directory not found: /path/to/data
```

The path in the graph config (`data_root`) doesn't exist. Check that the relevant
`paths.*` entry in your config points to the correct location and the data
subdirectory exists.

### Database already exists

```bash
cli kg create one_rainbow --delete-first
```

### Force rebuild of parquet cache

```bash
cli kg create one_rainbow --force parquet
```

### Vector index error during merge

`Cannot set property vec ... because it is used in one or more indexes` is a
Ladybug (Kuzu) limitation: HNSW vector-indexed properties cannot be updated in
place via `MERGE … SET`.

**This is handled automatically.** The flow calls `drop_vector_indexes_task` before
the ingestion phase and `create_vector_indexes_task` after. If you still see the
error, use `--delete-first` to start from a completely clean database:

```bash
cli kg create one_rainbow --delete-first
```

