# KG Backend Abstraction & Neo4j Support

**Status**: outline (proposal) — connection lifecycle already implemented in `genai_graph/kg/connections.py`
**Depends on**: [01_knowledge_graph_architecture.md](01_knowledge_graph_architecture.md), [02_document_graph_decomposition.md](02_document_graph_decomposition.md)
**Goal**: run the Document Graph on a client/server engine (Neo4j) without forking the tool, ingestion or benchmark layers.

---

## 1. Where we are

`KgBackend` (`genai_graph/kg/backend.py`) is an ABC with `LadybugBackend` implemented and `Neo4jBackend` a
`NotImplementedError` placeholder. Two problems block a second engine:

1. **Engine-specific Cypher leaks into the query layer.** `document_graph_tools.py` calls
   `CALL table_info(...)`, `CALL show_tables()`, `CALL QUERY_VECTOR_INDEX(...)`,
   `CALL QUERY_FTS_INDEX(...)` and reads native result frames through `execute_get_as_df`.
   None of those exist on Neo4j.
2. **Connection lifecycle was ad hoc.** Fixed: `genai_graph/kg/connections.py` now owns
   "one connection per (thread, target), explicit release, ordered shutdown" behind
   `get_connection` / `release_connection` / `connection(...)`, with a manager per backend kind
   (`LadybugConnectionManager` implemented, `Neo4jConnectionManager` outlined). Callers are
   backend-agnostic already.

## 2. Target interface: capability protocols instead of one fat ABC

Split the single ABC into narrow protocols, so a backend implements what it can and the tool
layer depends only on the capabilities it uses:

| Protocol | Methods | Ladybug today | Neo4j |
|---|---|---|---|
| `QueryExecutor` | `execute`, `execute_get_as_df` | implemented (native `get_as_df`) | run + records → DataFrame |
| `SchemaIntrospector` | `list_labels`, `list_relationship_types`, `columns(label)`, `has_label` | `CALL table_info`, `CALL show_tables` | `CALL db.labels()`, per-label `keys(n)` sample |
| `SchemaWriter` | `create_node_table`, `create_relationship_table`, `drop_table`, `insert_node`, `insert_relationship`, `merge_node` | node/rel tables | uniqueness constraint + `UNWIND $rows` MERGE |
| `IndexManager` | `ensure_vector_index`, `query_vector_index`, `ensure_text_index`, `query_text_index` | `CALL CREATE_VECTOR_INDEX` / `QUERY_FTS_INDEX` | `CREATE VECTOR INDEX` + `db.index.vector.queryNodes`; `CREATE FULLTEXT INDEX` + `db.index.fulltext.queryNodes` |
| `TransactionManager` | `transaction()` context manager | per-statement (implicit) | explicit `session.execute_write/read` |
| `ConnectionLifecycle` | `close()` | explicit, orderly close | session close (driver stays) |

`KgBackend` then composes them; `document_graph_tools` keeps using only
`QueryExecutor` + `SchemaIntrospector` + `IndexManager`, so `_table_columns`, `_has_relationship`,
`_doc_key` and the vector/FTS legs become engine-agnostic by construction.

## 3. Neo4j mapping (concrete)

```
connect(uri, auth=(user, password), database="neo4j")     # shared Driver per process
create_node_table("Folder", {...}, pk="folder_id")
  -> CREATE CONSTRAINT folder_id_unique IF NOT EXISTS FOR (n:Folder) REQUIRE n.folder_id IS UNIQUE
columns("MarkdownSection")
  -> MATCH (n:MarkdownSection) RETURN keys(n) LIMIT 1     # fallback when APOC is unavailable
ensure_vector_index("SectionChunk", "chunk_embedding", 1024, "cosine")
  -> CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS FOR (c:SectionChunk)
     ON (c.chunk_embedding) OPTIONS {indexConfig: {`vector.dimensions`: 1024,
     `vector.similarity_function`: 'cosine'}}
query_vector_index(...)  -> CALL db.index.vector.queryNodes('chunk_embedding_index', $k, $vec)
                            YIELD node, score
ensure_text_index("MarkdownSection", ["title", "text", "description"])
  -> CREATE FULLTEXT INDEX section_fts IF NOT EXISTS FOR (s:MarkdownSection)
     ON EACH [s.title, s.text, s.description]
query_text_index(...)    -> CALL db.index.fulltext.queryNodes('section_fts', $q) YIELD node, score
```

Conventions that carry over unchanged: content-hash primary keys (never internal ids), `UNWIND $rows`
batched writes, the same node/relationship names, and ACL as a `allowed_principals` property (with
Neo4j roles as an optional enforcement upgrade).

## 4. What must change in the shared code

- **Vector/`node` shapes**: Ladybug's `RETURN node` yields a dict; Neo4j yields a `Node` object.
  `search_images` reads `node.get(...)` — convert with `dict(node)` in the Neo4j `IndexManager`.
- **Full-text scoring**: Kuzu FTS vs Lucene BM25 differ in scale. The RRF fusion in
  `search_sections` is rank-based, so it survives; raw-score thresholds must not be hard-coded.
- **Schema tolerance**: `_doc_key` (content_hash vs markdown_hash) and the missing-column fallbacks
  must ask `SchemaIntrospector`, never build `CALL table_info` strings.
- **Read-only semantics**: `read_only=True` (no WAL/checkpointing) has no Neo4j equivalent; the
  manager should ignore the option rather than fail.
- **`create_backend_from_config`**: add a `backend:` key to the docgraph profile
  (`ladybug` default, `neo4j` opt-in) and select the connection manager by it.

## 5. Migration phases

1. **Extract protocols** (no behaviour change): move `CALL table_info` / `CALL show_tables` /
   vector+FTS calls behind `SchemaIntrospector` / `IndexManager`; keep Ladybug tests green.
2. **Implement `Neo4jConnectionManager` + `Neo4jBackend`** (query, schema, indexes) and validate the
   existing `document_graph_tools` suite against a Neo4j container.
3. **Ingestion path**: bulk writer for Neo4j, plus the profile `backend:` key and CLI plumbing.
4. **Parity run**: the OfficeQA subset on both backends, reporting retrieval (evidence-hit rate,
   MMLongBench only) and answer quality separately from cost.

## 6. Risks

- Retrieval quality may shift (Lucene vs Kuzu FTS, HNSW parameters) — measure before switching.
- `execute_get_as_df` on Neo4j costs a record→DataFrame conversion per call; keep result sets bounded
  (the pack budget already caps the read side).
- Long-lived read transactions: use autocommit reads; reserve explicit transactions for ingestion.
- Introspection via `keys(n)` samples one node; a mixed-schema label needs `apoc.meta.data()` or a
  union over several samples.
