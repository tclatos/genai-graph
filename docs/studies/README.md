# Empirical Studies & Benchmark Research (genai-graph)

This directory contains research studies, comparative benchmark evaluations, and empirical investigation reports for **genai-graph**.

> **Policy**: Documents in this directory are strictly reserved for external surveys, comparative research, and architectural evaluations that have **not** been implemented in the codebase. Implemented features and architectural blueprints belong in [docs/design/](../design/README.md) or the core user guides.

---

## Index of Studies (Unimplemented Research & External Surveys)

| Study / Report | Focus | Summary |
|---|---|---|
| [lightrag_entity_extraction.md](lightrag_entity_extraction.md) | **LightRAG Dual-Level Entity Extraction** | Design proposal and benchmark evaluation for corpus-scale entity & relation layer over Document Graph |
| [wiki_agent_scaffolding_retrospective.md](wiki_agent_scaffolding_retrospective.md) | **Wiki Agent Scaffolding Retrospective** | Scaffolding retrospective, offline notes, and image query case study |
| [other_agentic_search_frameworks.md](other_agentic_search_frameworks.md) | **MMLongBench-Doc Agentic Survey** | Synthesis and comparative landscape of external agentic frameworks (DocAtlas, MDocAgent, VLD-RAG, DocLens, MAGE-RAG) on multimodal long documents |
| [codebase_audit_2026-09.md](codebase_audit_2026-09.md) | **Codebase Quality & Security Audit** | Static analysis review across `genai_graph/` covering security, schema validation, and maintainability |

---

## Promoted & Implemented Specifications

- **Rule-Selected Ingestion Routing**: Formerly an exploratory study, now fully implemented and documented in [docs/design/04_rule_selected_ingest_routing.md](../design/04_rule_selected_ingest_routing.md).
- **Access Control & Security Trimming**: Fully implemented and documented in [docs/design/03_access_control_and_security_trimming.md](../design/03_access_control_and_security_trimming.md) and [docs/access-control-security-trimming.md](../access-control-security-trimming.md).

---

## Related Documentation

- [docs/README.md](../README.md) — Master documentation index
- [docs/design/README.md](../design/README.md) — Architectural design curriculum
- [docs/benchmark_framework.md](../benchmark_framework.md) — 6-Stage benchmark execution engine (`cli bench`)
- [docs/benchmarks_financebench_officeqa.md](../benchmarks_financebench_officeqa.md) — FinanceBench & OfficeQA Pro empirical evaluation
- [docs/document-graph.md](../document-graph.md) — Hierarchical Document Graph & vectorless agentic RAG
