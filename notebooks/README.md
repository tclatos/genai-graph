# GenAI Graph Interactive Notebooks

This directory contains interactive Jupyter notebooks demonstrating core capabilities of **genai-graph**, from basic in-memory schema definition to enterprise access control and full document processing pipelines.

---

## 🗺️ Recommended Learning Progression

| # | Notebook | Focus | Prerequisites & Environment |
|---|---|---|---|
| **01** | [01_define_graph_from_scratch.ipynb](01_define_graph_from_scratch.ipynb) | **First 5 Minutes with Knowledge Graphs**<br/>Define nodes & edges from Pydantic models, compile schema, ingest into in-memory Ladybug DB, run Cypher queries, render interactive D3 diagrams. | In-memory; no external services or API keys required. |
| **02** | [02_document_graph_full_pipeline.ipynb](02_document_graph_full_pipeline.ipynb) | **End-to-End Document Graph Pipeline**<br/>Document conversion (PDF/DOCX → Markdown), HTML table preservation, BAML/LLM outline enrichment, Ladybug DB indexing, and agent navigation tools. | `genai-tk` installed. Optional LLM API keys for BAML enrichment. |
| **03** | [03_access_control_security_trimming.ipynb](03_access_control_security_trimming.ipynb) | **Enterprise Access Control & Security Trimming**<br/>Hierarchical security propagation from Folders to Documents, principal evaluation, runtime user context, and early-binding Cypher filtering. | In-memory Ladybug DB; no external services required. |
| **Ref** | [cypher_examples.ipynb](cypher_examples.ipynb) | **Cypher Query Patterns & Idioms Reference**<br/>Practical reference for pattern matching, variable-length path traversal, aggregations, vector similarity queries (`CALL QUERY_VECTOR_INDEX`), and MERGE optimizations. | In-memory Ladybug DB. |
| **Alt** | [document_graph_demo.ipynb](document_graph_demo.ipynb) | **Lightweight Markdown-Only Demo**<br/>Zero-LLM demo of parsing markdown files into a Document Graph without external models. *(Superseded by #02 for production workflows)*. | Markdown files directory. |

---

## How to Run These Notebooks

From the repository root:

```bash
# Launch Jupyter Lab with uv environment
uv run jupyter lab notebooks/
```

Or open any notebook directly inside **VS Code** with the Python & Jupyter extensions active.

---

## Related Documentation

- [docs/README.md](../docs/README.md) — Master documentation index
- [docs/graph-definition-guide.md](../docs/graph-definition-guide.md) — 5-minute graph definition guide
- [docs/document-graph.md](../docs/document-graph.md) — Document Graph architecture and agent tools
- [docs/access-control-security-trimming.md](../docs/access-control-security-trimming.md) — Enterprise security trimming specification
