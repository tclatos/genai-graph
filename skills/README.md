# GenAI Graph Skill Map

This folder organizes skills for agents working on the **genai-graph** library or on downstream projects that depend on it into a **4-tier architecture**:

- `skills/runtime/` — **Runtime skills**: solve user problems (Cypher queries, Document Graph agentic navigation, KG Explorer).
- `skills/development/` — **Development skills**: construct and extend knowledge graphs & benchmarks (schema definition, factories, ingestion, workflows, neo4j import, benchmark framework).
- `skills/governance/` — **Governance skills**: maintain consistency, schema health, and codebase orientation (`kg-repo-map`, `kg-schema-maintenance`).
- `skills/vendor/` — **Vendor skills**: imported external skills (`atos-slidev`).

## How to use these skills

1. **Start with `kg-repo-map`.** It maps a work area to the closest doc, the concrete code
   paths, and the `kg-*` skill to load next. Load it first whenever a task touches Knowledge
   Graph construction, the Document Graph, Cypher agents, Neo4j import, KG workflows,
   export/visualization, or the KG Explorer.
2. **Load one domain skill** based on the work area (see the table below). Skills point at
   the authoritative `docs/*.md` first, then the implementing `genai_graph/**` code, so read
   the doc before inferring behavior from filenames.
3. **Load the matching `genai-tk/*` skill alongside it.** genai-graph extends genai-tk's
   three domains (Core GenAI, Agents, Workflows). The "Complements" row in each skill — and
   the table below — tells you which `genai-tk` skill to load with each `kg-*` skill. The two
   bundles are designed to be loaded together (see `genai-tk/skills/README.md` for the
   genai-tk skill map).

## 1. Runtime Skills (`skills/runtime/`)

| Skill | Closest docs | Primary code/config | Complements (genai-tk) |
|---|---|---|---|
| `kg-docgraph-agent` | `docs/document-graph.md` | `genai_graph/agent/docgraph_agent.py` | `agent-profiles`, `browser-and-sandbox` |
| `kg-document-graph` | `docs/document-graph.md` | `genai_graph/kg/document_graph/` | `workflow-engine` |
| `kg-explorer` | `docs/kg_explorer.md` | `genai_graph/webapp/`, `genai_graph/main/streamlit.py` | `webapp`, `streamlit-workflow-runner` |
| `kg-query` | `docs/kg_explorer.md`, `docs/document-graph.md` | `genai_graph/kg/query/` | `agent-profiles`, `add-tool` |

## 2. Development Skills (`skills/development/`)

| Skill | Closest docs | Primary code/config | Complements (genai-tk) |
|---|---|---|---|
| `benchmark-framework` | `docs/benchmark_framework.md` | `genai_graph/bench/`, `genai_graph/core/commands_bench.py` | `evaluation-testing` |
| `kg-schema` | `docs/graph-definition-guide.md`, `docs/schema-compilation.md` | `genai_graph/kg/schema/` | `core-models` |
| `kg-factories` | `docs/graph-authoring-patterns.md`, `docs/graph_construction.md` | `genai_graph/kg/factories/` | `baml-structured-extraction` |
| `kg-ingest` | `docs/graph_construction.md`, `docs/cache_management.md`, `docs/workflows.md` | `genai_graph/kg/ingest/`, `genai_graph/kg/backend.py` | `core-models` |
| `kg-neo4j-import` | `docs/graph-authoring-patterns.md` (Pattern 3) | `genai_graph/neo4j_import/` | — |
| `kg-workflows` | `docs/workflows.md`, `docs/prefect_dag_pipeline.md` | `genai_graph/orchestration/` | `workflow-engine` |
| `kg-export` | `docs/graph-definition-guide.md`, `docs/kg_create_enhancements.md` | `genai_graph/kg/export/` | — |
| `kg-cli` | `docs/workflows.md` (CLI Reference), `docs/document-graph.md` | `genai_graph/core/commands_*.py` | `cli-and-scaffolding` |

## 3. Governance Skills (`skills/governance/`)

| Skill | Closest docs | Primary code/config | Complements (genai-tk) |
|---|---|---|---|
| `kg-repo-map` | `docs/graph-definition-guide.md` | `genai_graph/kg/`, `genai_graph/orchestration/` | `repo-map` |
| `kg-schema-maintenance` | `Agents_Skills.md`, `docs/schema-compilation.md` | `genai_graph/kg/schema/`, `genai_graph/kg/factories/` | — |

## 4. Vendor Skills (`skills/vendor/`)

| Skill | Description | Primary Origin |
|---|---|---|
| `atos-slidev` | Executive Slidev presentation decks | External Atos branding package |

All skill `name:` fields are `kg-`-prefixed (except shared utilities) so they never collide with `genai-tk/*` skills
when both directories are loaded together.

```yaml
# config/agents/langchain/kg_dev.yaml — unified `agents:` dict
agents:
  kg_dev:
    harness: langchain
    type: deep                 # deep (not react) so SkillsMiddleware injects skills
    llm: default
    skill_directories:
      - ${paths.project}/skills/genai-graph   # this bundle
      - ${paths.project}/skills/genai-tk      # the genai-tk bundle (if vendored locally)
```

```yaml
# or a DeerFlow profile (any mode honors skills)
agents:
  kg_dev:
    harness: deerflow
    mode: pro
    skills:
      - kg-repo-map
      - kg-schema
    skill_directories:
      - ${paths.project}/skills/genai-graph
```

If you only need a skill to guide development (not runtime), `type: react` is fine and you
do not need to wire `skill_directories` at all.

## Related

- `genai-graph/Agents.md` — coding conventions (Pydantic v2, modern type hints, no
  backward-compat code, Ladybug/Kuzu backend notes).
- `genai-graph/Agents_Skills.md` — project-specific procedures that `kg-schema-maintenance`
  generalizes.
- `genai-tk/skills/README.md` — the genai-tk skill map (the foundation skills these
  complement).
