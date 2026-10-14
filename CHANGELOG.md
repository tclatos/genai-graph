# Changelog (genai-graph)

All notable changes to genai-graph will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) — [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Knowledge Graph Core & Embedded Ladybug
- **Buffer Pool Scaling**: Increased default Ladybug buffer pool ceiling to 8GB, with graceful `buffer_pool_size: 0` fallback when unset to prevent out-of-memory crashes during bulk loading.
- **Batch MERGE Optimization**: Optimized bulk relationship insertion routines to eliminate accidental Cartesian cross-products and memory pool exhaustion.
- **Scoped Schema Introspection Caching**: Scoped table and relationship metadata introspection cache directly to backend instances to prevent stale cache contamination across concurrent test fixtures or multi-database sessions.
- **Provider Role Alignment**: Aligned `config/providers/providers.yaml` with the `genai-tk` Inversion of Control (IoC) architecture using role-based class fields (`chat_class`, `embeddings_class`, `decision_class`).

#### Hierarchical Document Graph (DocGraph)
- **Vectorless Agentic Navigation**: Implemented hierarchical document tree decomposition (`Folder` -> `Document` -> `MarkdownSection` -> `SectionChunk`), allowing autonomous agents to explore document outlines without consuming context window budget on full-text vectors.
- **Dynamic Query Relaxation**: Added automated relaxation of restrictive boolean search constraints and extension-agnostic document name resolution in `search_sections`.
- **Parquet Staging & Image Support**: Added intermediate Parquet staging pipeline for high-throughput batch loading, and added `Image` node type with base64 payload representation.
- **DocGraph Terminal TUI**: Enhanced interactive terminal TUI with mouse text selection support and clean Markdown table rendering.

#### Document Graph Tools
- **Declarative Tool Factory**: Added `create_docgraph_tools` factory in `genai_graph.agent` that resolves database configuration automatically and instantiates all 6 Document Graph navigation tools (`get_folder_toc`, `get_document_toc`, `get_section_content`, `search_sections`, `query_image`, `list_documents`) for direct reference in agent profile YAML.

#### Graph Commands & Cypher Query TUI
- **Top-Level `cli graph` Command Group**: Introduced `GraphCommands` (`cli graph`) providing `display` and `query` subcommands across `genai-graph` and downstream projects such as `rfq-pricing`.
- **Interactive Graph Display (`cli graph display`)**: Generates an interactive D3.js force-directed HTML graph visualization from a Cypher query with automatic browser opening, custom query overrides, and limit flags.
- **Interactive Cypher Query TUI (`cli graph query`)**: Textual terminal user interface for testing and exploring Cypher queries. Features live schema table introspection (nodes and relationships), curated generic and Document Graph example queries with one-click editor population, query execution with timing and row count metrics, scrollable tabular result display, and CSV export. Includes batch execution mode (`--batch -q "..."`).

#### Enterprise Access Control & Security Trimming
- **Early-Binding Security Trimming**: Integrated document-level authorization and tenant isolation directly into Ladybug Cypher execution (`WHERE d.tenant_id = $tenant_id AND ...`).
- **Parameterized Security Clauses**: Sanitized security parameter injection to strictly eliminate Cypher injection vulnerabilities.
- **Fail-Closed Security Invariant**: Enforced fail-closed authorization semantics ensuring unauthorized nodes are never retrieved or visible.
- **Principal Propagation**: Seamless flow of user identity, roles, and group claims from outer HTTP/CLI requests through the agent harness into graph query tools.

#### Orthogonal Ingestion Routing
- **Rule-Selected Ingestion Routing**: Connected gitignore-style pathspec route tables (`config/ingest_routes.yaml`) to parallel Prefect sub-workflows (PDF OCR, Docling Office conversion, Web page fetchers, Git repository ingestion).
- **Single-Writer Fan-In**: Isolated concurrent Markdown conversion into staging directories, resolving embedded Ladybug database write locks with a single atomic build pass.

#### Model Context Protocol (MCP 2.0)
- **MCP 2.0 Document Graph Server**: Exposed Document Graph navigation tools (`list_documents`, `get_document_outline`, `get_section_content`, `search_sections`, `query_cypher`) over MCP 2.0 with `stdio` and `sse` transports for external IDEs (VS Code, Cursor, Windsurf) and agents.

#### Benchmarking & Evaluation
- **OfficeQA Pro V2 Ingestion**: Added ingestion pipeline for the OfficeQA Pro V2 JSON corpus (249 documents) and implemented dual reward scoring (official Databricks exact-match evaluator + LLM judge rubric).

#### Documentation Architecture
- **Didactic Numbered Curriculum**: Reorganized architectural documentation into a 5-part numbered curriculum in `docs/design/` (`01_knowledge_graph_architecture.md`, `02_document_graph_decomposition.md`, `03_access_control_and_security_trimming.md`, `04_rule_selected_ingest_routing.md`, `05_mcp_and_agent_navigation.md`).
- **Promoted Implemented Studies**: Promoted rule-selected ingestion routing from `studies/` to core design, ensuring `docs/studies/` contains only unimplemented research and external surveys.
