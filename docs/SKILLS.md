# GenAI Graph Skills Guide

Skills are `SKILL.md` files that provide AI agents with **procedural knowledge on demand**.
Instead of bloating system prompts, agents dynamically load domain skills when performing Knowledge Graph construction, Cypher querying, Document Graph navigation, or benchmark evaluations.

This project implements the [skills.sh](https://www.skills.sh) open standard — fully compatible with GitHub Copilot, Cursor, Claude Code, and autonomous coding agents.

---

## Quick Reference

```bash
# List all discovered skills across tiers
cli skills list

# Validate all skills against schema & guidelines
cli skills validate --all

# Display detailed metadata and content for a skill
cli skills info kg-schema

# Scaffold a new skill in skills/custom/
cli skills create my-graph-skill
```

---

## 4-Tier Skills Architecture

Skills in `genai-graph` are organized across four tiers in `skills/`:

```
skills/
├── runtime/              # User-facing capabilities executed during agent runs
│   ├── kg-query/         # Direct Cypher, text-to-Cypher, vector search
│   ├── kg-explorer/      # Streamlit KG explorer UI & Cypher interface
│   ├── kg-document-graph/# Document Graph schema & hierarchical traversal
│   └── kg-docgraph-agent/# Autonomous deep planning agent with docgraph tools
│
├── development/          # Skills for constructing and extending graphs
│   ├── kg-schema/        # GraphNode/GraphRelation/GraphSchema definitions
│   ├── kg-factories/     # JsonFile, Table, and BAML graph factories
│   ├── kg-ingest/        # Batch ingestion, Ladybug backend, fingerprint cache
│   ├── kg-cli/           # CLI command reference (cli kg, cli docgraph, cli bench)
│   ├── kg-workflows/     # YAML orchestration and Prefect DAG execution
│   ├── kg-neo4j-import/  # Neo4j JSONL export to Ladybug migration
│   ├── kg-export/        # Interactive D3/HTML graph visualization & warnings
│   └── benchmark-framework/ # 6-stage multi-dataset evaluation pipeline (cli bench)
│
├── governance/           # Quality, schema maintenance, and codebase navigation
│   ├── kg-repo-map/      # Codebase orientation: map feature area -> docs -> code
│   └── kg-schema-maintenance/ # Step-by-step schema evolution, merges & audits
│
└── vendor/               # Imported third-party skills
    └── atos-slidev/      # Slidev executive and technical presentation decks
```

For the comprehensive skill index and how `genai-graph` skills complement `genai-tk` skills, see [skills/README.md](../skills/README.md).

---

## How AI Agents Use Skills

Agents configured with the `deep` type or equipped with `SkillsMiddleware` discover and load skills dynamically:

1. **Orientation**: The agent starts with `kg-repo-map` to determine which documentation and code paths are authoritative for the user's task.
2. **Progressive Disclosure**: When tasked with schema changes, the agent loads `kg-schema` and `kg-schema-maintenance`. When building pipelines, it loads `kg-workflows` and `kg-factories`.
3. **Paired Skills**: Many `kg-*` skills complement foundational `genai-tk` skills (e.g., `kg-workflows` pairs with `workflow-engine`, `kg-schema` pairs with `core-models`).

---

## Writing a New Graph Skill

Use `cli skills create <name>` to scaffold a new skill:

```markdown
---
name: kg-my-extension
description: Procedure for authoring custom graph extractors and loaders in genai-graph.
tags: [knowledge-graph, extraction, ladybug]
version: "1.0"
---

# Knowledge Graph Custom Extraction

## When to Use

Use when adding custom domain entity extraction logic to a genai-graph pipeline.

## Read First

- `docs/graph-definition-guide.md` — Core modeling and ingestion steps
- `docs/graph-authoring-patterns.md` — Common factory patterns
- `genai_graph/kg/schema/` — Schema definition classes

## Step-by-Step Procedure

1. Define the Pydantic v2 node model inheriting from `BaseModel`.
2. Wrap with `GraphNode` and define relation endpoints.
3. Register with `GraphSchema` and run ingestion.
```

---

## Related Documentation

- [skills/README.md](../skills/README.md) — Complete inventory and pairing matrix
- [docs/README.md](README.md) — Master documentation index
- [docs/graph-definition-guide.md](graph-definition-guide.md) — 5-minute graph definition guide
- [Agents_Skills.md](../Agents_Skills.md) — Procedural checklist for AI agents
