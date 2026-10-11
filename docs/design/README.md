# Architectural Design Documents (genai-graph)

This directory contains the foundational design specifications, architectural blueprints, and implementation guides for **genai-graph**.

The core architecture is presented in a didactic, numbered sequence designed for engineers, architects, and autonomous coding agents.

---

## 📚 Numbered Architecture Curriculum

| # | Design Document | Core Subject & Key Topics |
|---|---|---|
| **01** | [01_knowledge_graph_architecture.md](01_knowledge_graph_architecture.md) | **Knowledge Graph Foundation & Schema Compilation**: Ladybug embedded engine, Pydantic v2 declarative schema compilation, deterministic primary key hashing, batch MERGE mechanics, and buffer pool management. |
| **02** | [02_document_graph_decomposition.md](02_document_graph_decomposition.md) | **Hierarchical Document Graph & Agentic Navigation**: Structure-first outline decomposition (`Folder` -> `Document` -> `MarkdownSection` -> `SectionChunk`), vectorless agentic navigation, hybrid full-text/vector index, and dynamic query relaxation. |
| **03** | [03_access_control_and_security_trimming.md](03_access_control_and_security_trimming.md) | **Enterprise Access Control & Security Trimming**: Document-level authorization, principal propagation via harness, early-binding Cypher query filtering, and fail-closed security. |
| **04** | [04_rule_selected_ingest_routing.md](04_rule_selected_ingest_routing.md) | **Multi-Source Ingestion Routing & Route Tables**: Rule-selected YAML workflows, pathspec route tables, concurrent sub-workflows (PDF, Office, Web, Git), and single-writer fan-in into Ladybug. |
| **05** | [05_mcp_and_agent_navigation.md](05_mcp_and_agent_navigation.md) | **Model Context Protocol (MCP 2.0) & Graph Tools**: Document Graph MCP servers (stdio and SSE), navigation tool suite (`list_documents`, `get_document_outline`, `get_section_content`, `search_sections`, `query_cypher`), and IDE integration. |

---

## 🔬 Empirical Studies, Proposals & Research

For proposals, retrospective case studies, empirical benchmarks, diagnostic post-mortems, and research that have not yet been promoted to core design, consult [docs/studies/](../studies/README.md):
- [LightRAG Dual-Level Entity Extraction Proposal](../studies/lightrag_entity_extraction.md)
- [Wiki Agent Scaffolding Retrospective](../studies/wiki_agent_scaffolding_retrospective.md)
- [MMLongBench-Doc Agentic Frameworks Survey](../studies/other_agentic_search_frameworks.md)
- [Codebase Quality & Security Audit](../studies/codebase_audit_2026-09.md)
