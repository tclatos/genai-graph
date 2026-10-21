# Study — Neo4j as a Portable Backend for genai-graph

*Status: draft for review · 2026-10-21*
*Scope: feasibility, capability comparison (Ladybug vs Neo4j), target architecture, refactoring plan. No code.*

---

## 1. Executive summary

Neo4j is a viable **opt-in** backend for genai-graph. The existing `KgBackend` abstraction (`genai_graph/kg/backend.py`) is already Cypher-shaped, so the real work is not "port a query language" but porting four things that are currently Ladybug-specific:

1. **DDL & schema** — `CREATE NODE TABLE / REL TABLE` has no Neo4j equivalent; Neo4j's new *graph types* (EE) or constraints (CE) must be used instead.
2. **Batch ingestion** — `LOAD FROM arrow_table MERGE …` is a Ladybug extension; Neo4j needs driver-side `UNWIND` batching, or offline `neo4j-admin … --input-type=parquet` import.
3. **Search surfaces** — vector index and FTS calls (`CALL QUERY_VECTOR_INDEX`, `QUERY_FTS_INDEX`) must be re-expressed with Neo4j's built-in vector/full-text indexes.
4. **Type mapping** — `_get_kuzu_type` and the Arrow schema builders need a dialect layer (Python/Pydantic → Ladybug types, Neo4j types, Arrow).

The graph factories' **Parquet artifacts are the key simplifier**: `ParquetCollector` already captures the exact merged data (`nodes/<Type>.parquet`, `rels/<RelType>.parquet`, `manifest.json`) independently of the target DB. That makes a Neo4j deployment a *replay of known-good data* instead of a second ingestion pipeline — both for offline bulk import (`neo4j-admin`) and online catch-up (driver `UNWIND`).

Main gaps to design around:

- **STRUCT columns** (embedded sub-models) have no Neo4j property equivalent → flatten or JSON-serialize.
- **Graph types and the native `VECTOR` property type are Enterprise-only**; Community gets vector *indexes on `LIST<FLOAT>`* and full-text indexes, which is enough for feature parity of search, without hard schema enforcement.
- **Primary-key semantics differ** (Ladybug immutable PK property vs Neo4j uniqueness constraint + `MERGE` key).

Recommendation: implement Neo4j as a second `KgBackend` behind config, keep Ladybug as default, share one schema/type-mapping layer and one Parquet artifact set between both directions (import *from* Neo4j and write *to* Neo4j).

---

## 2. Current architecture snapshot

### 2.1 Backend layer — `genai_graph/kg/backend.py`

- `QueryExecutor` (ABC): `execute(query, parameters)`.
- `KgBackend` (ABC): `connect`, `execute`, `create_node_table`, `create_relationship_table`, `drop_table`, `insert_node`, `insert_relationship`, `close`, `get_query_language`, plus concrete `execute_get_as_df` (assumes result objects expose `get_as_df()`).
- `LadybugBackend`: full implementation — extension install/load (`VECTOR`, `FTS`) under process-wide locks, `_NATIVE_CALL_LOCK` serialising `CALL` statements, `create_vector_index` / `drop_vector_index` / `query_vector_index` (HNSW: `mu`, `ml`, `pu`, `efc`, `efs`), `merge_node` (MERGE on introspected PK, PK excluded from `SET`), `attach()` for shared-DB worker pools.
- `Neo4jBackend`: **stub** — every method raises `NotImplementedError`. `get_query_language()` already returns `"Cypher"`.
- Factory helpers: `create_backend("ladybug"|"kuzu"|"neo4j")` and `create_backend_from_config(config_key)` already read `graph_db.<key>.{type, path, username, password}` and branch on `neo4j` (currently calling the stub).

### 2.2 Schema & factories

- `GraphSchema` / `GraphNode` / `GraphRelation` (`genai_graph/kg/schema/core.py`): Pydantic-model-driven schema, with `key_from`, `index_fields` (vector-indexed fields), embedded struct classes, and descriptions consumed by LLM prompts.
- Factories (`KgFactory` subclasses, e.g. `Neo4jImportFactory`) produce `NodeDataCollection` + `RelationshipRecord`s.

### 2.3 Ingestion — `genai_graph/kg/ingest/merge.py`

- `merge_nodes_batch` / `merge_relationships_batch` build explicit **Arrow schemas** from the Pydantic node classes (`_build_node_arrow_schema`, `pydantic_annotation_to_arrow`) and execute Ladybug-specific `LOAD FROM arrow_table MERGE …` statements, with row-by-row fallbacks.
- `ParquetCollector` (thread-safe, global per KG build) accumulates the **exact merged Arrow tables** during ingestion. This is the preferred export path (`save_parquet_from_collector`) because it avoids reading data back out of the DB.

### 2.4 Parquet artifacts — `genai_graph/kg/export/artifacts.py`

```
{kg_outputs}/{profile}/parquet/
  nodes/<NodeType>.parquet      # one file per node table, Arrow schema = table schema
  rels/<RelType>.parquet        # columns: from_id, to_id, props…,
                                #          _from_type, _to_type, _from_key_field, _to_key_field
  manifest.json                 # counts + fingerprints (schema, factory config, source content)
```

- `validate_parquet_cache` / `CacheFingerprints` make these artifacts a **deterministic, cacheable representation of the graph** — exactly what a second backend needs to ingest.
- Caveat: rel files currently carry *logical* endpoints (`from_id` + `_from_type` + `_from_key_field`), not Neo4j `:START_ID`/`:END_ID` — a small transformer is needed for `neo4j-admin` (see §6.2).

### 2.5 Neo4j-as-source (existing, months-old)

- `Neo4jFactory` / `Neo4jImportFactory` (`genai_graph/kg/factories/neo4j_factory.py`): streams Neo4j **JSONL exports** (`neo4j-admin export`), analyses them with `SchemaAnalyzer` (`genai_graph/neo4j_import/schema_analyzer.py`), maps labels/rels to typed Pydantic models via `Neo4jNodeMapping` / `Neo4jRelationMapping` (property renaming, filtering, `key_field`, `index_fields`), and emits `NodeDataCollection` + `RelationshipRecord`s for the *Ladybug* merge pipeline.
- `add_neo4j_data_to_graph` (`ingest/documents.py`) completes the flow.
- This code is **direction-inverted** (Neo4j → genai-graph) but it already contains a hand-rolled Neo4j↔Pydantic type-mapping concern (including JSON-string embedding parsing) that the new backend should reuse rather than duplicate.

### 2.6 Query-time surfaces that must stay portable

- Vector KNN (Document Graph sections, similarity factories) — `CALL QUERY_VECTOR_INDEX`.
- Full-text BM25 + hybrid search — `QUERY_FTS_INDEX`, `search_sections`.
- `export_schema_json` → `build_kg_agent_system_prompt` / text2cypher agents — backend-agnostic *if* the Cypher produced by agents stays within the common dialect subset.
- `export_parquet` / `export_html` read tables via `CALL show_tables()` and `MATCH (n:Type) RETURN n.*` — Ladybug introspection.

---

## 3. Neo4j capability review

> Version note: the graph-types and `SEARCH`-clause features are recent (2026.x releases). Exact version/edition matrices should be re-verified at implementation time; the statements below reflect the docs studied for this report.

### 3.1 Typing — "graph types" (`ALTER CURRENT GRAPH TYPE SET`)

Neo4j historically had no schema; **graph types** (Enterprise, introduced 2026.02) add optional schema enforcement over the property graph:

```cypher
ALTER CURRENT GRAPH TYPE SET
  (:Person => :Resident {name :: STRING NOT NULL, age :: INTEGER}),
  (:Resident)-[:LIVES_IN => {since :: DATE NOT NULL}]->(:City)
```

- Element types declare labels/relationship types, property types, `NOT NULL`, defaults; constraints (uniqueness, existence, key) attach per label/type.
- Setting graph types **overwrites** all previous types and constraints — recommended to define on an empty DB.
- Property types available: `STRING`, `INTEGER`, `FLOAT`, `BOOLEAN`, `DATE`, `DATETIME`, `ZONED DATETIME`, `TIME`, `DURATION`, `POINT`, `LIST<T>`, `VECTOR` (EE, block format), `BLOB`.

**Comparison with Ladybug.** Ladybug is schema-*full* per table at `CREATE NODE TABLE` time (typed columns, declared PK); it also supports STRUCT columns and richer nested types. Neo4j graph types are schema-*governance* layered on a flexible store: they validate writes, but do not give Ladybug-style columnar STRUCT storage, and missing properties (absent keys) remain a normal Neo4j state unless `NOT NULL` + existence constraints are declared.

Consequence for genai-graph: our Pydantic models are already the **single source of truth**; graph types would be *generated* from `GraphSchema` when targeting Neo4j EE, and degrade to uniqueness + (where supported) property-existence constraints on CE.

### 3.2 Vector search

| Aspect | Ladybug today | Neo4j |
|---|---|---|
| Storage | `FLOAT[]` column on the node table | `LIST<FLOAT>` property (CE & EE); native `VECTOR` property type (EE only) |
| Index | `CALL CREATE_VECTOR_INDEX(...)` HNSW extension (`metric`, `mu`, `ml`, `pu`, `efc`, `efs`) | `CREATE VECTOR INDEX … FOR (n:Label) ON n.embedding` with `vector.dimensions`, `vector.similarity_function` |
| Query | `CALL QUERY_VECTOR_INDEX('table','index',$vec,k, efs:=…)` | `db.index.vector.queryNodes(indexName, k, $vec)` — or the newer `SEARCH … IN (VECTOR INDEX … FOR $vec LIMIT k)` clause (2026.09+) |
| Availability | extension (install/load, heap-corruption locks in our code) | built-in; **CE and EE both support vector indexes**; only the native `VECTOR` *type* is EE |
| Semantics | returns node + distance | returns node + similarity (cosine) — needs a normalisation shim in `Neo4jBackend.query_vector_index` |

Notes: one vector index per (label, property) — mirrors our per-field indexes. Index creation is asynchronous (populate in background) — worth a wait/poll helper so builds don't immediately query an unpopulated index. Dimension is fixed at creation from the embedding model — same as our `embedding_field_dimensions`.

### 3.3 Full-text search

| Aspect | Ladybug today | Neo4j |
|---|---|---|
| Definition | FTS extension; index creation via extension calls | `CREATE FULLTEXT INDEX … FOR (n:Label) ON EACH [n.prop, …]` with `fulltext.analyzer`, `fulltext.eventually_consistent` |
| Engine | BM25 (`QUERY_FTS_INDEX`) | Lucene (per-analyzer scoring; default analyzers incl. stemming, stop-words, accent folding) |
| Query | `CALL QUERY_FTS_INDEX(...)` | `db.index.fulltext.queryNodes(indexName, query)` or `SEARCH` clause |

BM25 vs Lucene default scoring differ numerically → **hybrid-search thresholds/rerank weights must be calibrated per backend**, not shared verbatim. This is the subtlest behavioural difference in the whole port.

### 3.4 Bulk import (offline) and incremental (online)

**Offline — `neo4j-admin database import full`** (Parquet supported since 5.26):

- Node files: parquet columns + header mapping; a stable ID column is required for linking.
- Relationship files: mandatory `:START_ID`, `:END_ID`, `:TYPE` columns.
- Full import targets an **empty** database; indexes/constraints are created *after* import (EE's `--schema` flag applies to block format only, not parquet).
- High parallelism, off-heap tuning, two-line headers for type overrides.

**Online — Python driver:**

- Batched `UNWIND $rows AS row MERGE …` inside `execute_query`/session transactions; batching is the well-trodden path (equivalent role to our `LOAD FROM arrow_table`).
- Works everywhere including AuraDB, unlike `neo4j-admin` import.

### 3.5 Edition trade-offs

| Capability | Community (self-hosted, e.g. Docker) | Enterprise (self-hosted) | AuraDB (cloud) |
|---|---|---|---|
| Vector *indexes* on `LIST<FLOAT>` | ✅ | ✅ | ✅ |
| Native `VECTOR` property type | ❌ | ✅ | ✅ (verify tier) |
| Graph types (`ALTER CURRENT GRAPH TYPE SET`) | ❌ | ✅ (2026.02+) | ✅ (verify tier/version) |
| Full-text indexes + analyzers | ✅ | ✅ | ✅ |
| `neo4j-admin` offline parquet import | ✅ | ✅ | ❌ (managed; online ingest only) |
| Existence/property-not-null constraints | limited | ✅ | ✅ |
| Multi-writer ACID concurrency | ✅ | ✅ | ✅ |
| Ops burden / cost | lowest | licence | managed subscription |

Given the user preference that the study stay edition-agnostic: design for **CE feature set as the baseline** (vector index on list properties + full-text + uniqueness constraints), enable graph types / native VECTOR as optional enhancements when an EE/Aura target is configured.

---

## 4. Feature gap analysis (Ladybug → Neo4j)

| # | Concern | Ladybug today | Neo4j equivalent | Gap / action |
|---|---|---|---|---|
| 1 | Table DDL | `CREATE NODE TABLE t(... PRIMARY KEY(k))`, `CREATE REL TABLE` | No tables; labels + property schema via graph types (EE) or just indexes/constraints | `create_node_table` → generated graph-type declaration or constraint set; needs a schema registry to remember declared keys |
| 2 | Batch upsert | `LOAD FROM arrow_table MERGE …` (in-process Arrow scan) | Driver `UNWIND $rows` batching; or offline admin import | New `BatchWriter` abstraction; Arrow table → list-of-dicts pages |
| 3 | Primary key | Declared column, immutable (can't be `SET`) | No PK concept; uniqueness constraint on a property; `MERGE (n:Label {k: $k})` | Keep PK column as merge key; uniqueness constraint per label; adapt `merge_node` |
| 4 | Embedded STRUCT sub-models | Native `STRUCT` columns (typed nulls, full-struct normalisation) | **No map/struct property values** — only scalars + lists of scalars | Flatten into prefixed properties or JSON-string column; affects text2cypher prompts |
| 5 | `MAP(STRING,STRING)` dict fields | Native MAP | Not storable as property | JSON-serialise; document in schema |
| 6 | Embeddings | `FLOAT[]` + HNSW extension | `LIST<FLOAT>` property + vector index (CE OK) | Straightforward; type mapper emits `LIST<FLOAT>`; consider `VECTOR` on EE |
| 7 | FTS | FTS extension, BM25 | Built-in Lucene full-text indexes | New index DDL + query shim; recalibrate hybrid weights |
| 8 | Multi-statement execute | Semicolon-separated → list of results | One statement per transaction | Split queries in `Neo4jBackend.execute` |
| 9 | Introspection | `CALL show_tables()`, `CALL table_info('t')` | `SHOW TABLES`, `db.schema.nodeTypeProperties`, `SHOW CONSTRAINTS` | Used by `export_parquet`, PK introspection — needs per-backend impls |
| 10 | Node identity | PK property; `id()` internal | `elementId()` (opaque), our key property is authoritative | Export rel parquet must key on our properties (it already does) |
| 11 | Concurrency model | Single writer, native-heap locks (`_EXTENSION_LOCK`, `_NATIVE_CALL_LOCK`) | Full MVCC multi-writer | Locks become no-ops behind the backend interface — keep interface, not behaviour |
| 12 | Result → DataFrame | `result.get_as_df()` | Driver records → `pd.DataFrame` | Normalisation layer in `Neo4jBackend.execute_get_as_df` |
| 13 | Cypher dialect | Kuzu/Ladybug dialect | openCypher + extensions | Common-subset lint for agent-generated queries; dialect-specific guards |
| 14 | Delete/replace tables | `DROP TABLE` | `DROP CONSTRAINT`/`CALL { … } DELETE` batched or `MATCH (n:Label) DETACH DELETE n` | Bulk delete helper per backend |

---

## 5. Shared type-mapping matrix (proposed single source of truth)

One Pydantic field annotation should fan out to three representations. Proposed canonical module (replacing `_get_kuzu_type` and the ad-hoc mappings in the Neo4j import path):

| Python / Pydantic | Arrow (merge/parquet) | Ladybug column | Neo4j property type |
|---|---|---|---|
| `str` (and enums) | `string` | `STRING` | `STRING` |
| `int` | `int64` | `INT64` | `INTEGER` |
| `float` | `double` | `DOUBLE` | `FLOAT` |
| `bool` | `bool` | `BOOLEAN` | `BOOLEAN` |
| `list[float]` (`*_embedding`) | `list<double>` | `FLOAT[]` | `LIST<FLOAT>` (+ optional `VECTOR` on EE) |
| `list[str]` | `list<string>` | `STRING[]` | `LIST<STRING>` |
| `dict[str, str]` | `string` (JSON) | `MAP(STRING,STRING)` | `STRING` (JSON) — gap #5 |
| embedded sub-model | `struct<…>` | `STRUCT<…>` | flattened props or JSON `STRING` — gap #4 |
| `datetime`/`date` | `timestamp`/`string` | `TIMESTAMP`/`DATE` | `DATETIME`/`DATE` (ZONED choice to fix) |

Note the deliberate Arrow change for #5/#4: if Neo4j is a target, `dict`/struct fields should be **JSON-serialised in the Parquet artifacts too**, so both backends ingest identical artifacts. That is a breaking artefact-schema change — acceptable per the "prefer simplicity over backwards compatibility" rule, but it must version the manifest.

---

## 6. Proposed target architecture (refactoring plan, no code)

### 6.1 Phase 1 — Dialect layer (pure refactor, behaviour-preserving)

- New `genai_graph/kg/dialects/` (naming TBC): `TypeMapper` (matrix above), `DdlRenderer` (graph schema → Ladybug DDL / Neo4j constraints+graph types), `Introspection` (list tables, PK of table, counts).
- Refactor `create_schema`, `_build_node_arrow_schema`, `merge_nodes_batch` to consume the dialect instead of hard-coded Ladybug syntax.
- Refactor `Neo4jImportFactory`'s JSONL type coercion (embedding-string parsing, `Neo4jNodeMapping` type inference) to use the same `TypeMapper` — one mapping layer, both directions (per the agreed decision to unify with `neo4j_import/`).

### 6.2 Phase 2 — Parquet → Neo4j loaders

- **Offline bulk** (`neo4j-admin --input-type=parquet`): transformer over existing artifacts → import directory with Neo4j headers; rels get `:START_ID`/`:END_ID`/`:TYPE` resolved from `_from_type/_from_key_field` columns; nodes get an ID header from their key field. Shipped as a `cli kg` command (e.g. `cli kg export-neo4j-import <config>`), executed against a Dockerised Neo4j for local/CI.
- **Online incremental**: `Neo4jBatchLoader` reading the same artifacts (`nodes/*.parquet` → `UNWIND` pages of `MERGE`; rels → `MATCH … MERGE`); suitable for AuraDB and for catch-up syncs.
- Manifest gains `ingestion: {mode, target, timestamp}` fields; fingerprints untouched.

### 6.3 Phase 3 — `Neo4jBackend` implementation

Implement the stub against the `KgBackend` interface:

- `connect`: `neo4j.GraphDatabase.driver(uri, auth)`; config via existing `graph_db.<key>` block (`uri`, `database`, `username`, `password`; keep `path` as the `uri` for compat).
- `execute`: session-per-call; return a result wrapper supporting `get_as_df()`; split semicolon-joined statements.
- `create_node_table` → uniqueness constraint on PK + (EE) generated graph-type fragment; `create_relationship_table` → no-op beyond rel-type registration in the schema registry.
- `merge_node` / batch merges → parameterised `UNWIND … MERGE` (never string interpolation, unlike today's Ladybug literal escaping).
- `create_vector_index` / `query_vector_index` → `CREATE VECTOR INDEX` / `db.index.vector.queryNodes` with a similarity→distance shim.
- FTS: `create_fts_index` + `query_fts_index` methods promoted to the `KgBackend` ABC (Ladybug backend gets them too — today they are implicit in `search_sections`).
- Locks: keep call sites; make backend-specific lock providers (no-op for Neo4j).

### 6.4 Phase 4 — Query-time portability

- Hybrid search: define a `SearchBackend` protocol (vector KNN + FTS + rerank) implemented per backend; keep the Document Graph `search_sections` behaviour identical from the caller's perspective; recalibrate scores.
- text2cypher: `export_schema_json` gains a `dialect` hint; agent prompt includes backend-specific caveats (functions available, STRUCT flattening).
- Add a shared "common Cypher subset" test corpus executed against both backends (Docker Neo4j in CI).

### 6.5 Phase 5 — Configuration & docs

```yaml
graph_db:
  default:                 # unchanged, Ladybug
    type: ladybug
    path: {kg_outputs}/…db
  neo4j-prod:
    type: neo4j
    uri: neo4j://host:7687
    database: genai
    username: neo4j
    password: ${…}
    edition_features: {graph_types: true, native_vector: true}
```

- `create_backend_from_config` already branches on `neo4j` — only credentials plumbing is missing.
- Document the "Ladybug = dev/default, Neo4j = opt-in deployment target" model, per the agreed positioning.

---

## 7. Risks and mitigations

1. **STRUCT/MAP gap** — largest semantic risk. Mitigate: JSON-serialise in artifacts + schema registry marks such fields "opaque" for agents; consider flattening high-value sub-models later.
2. **Scoring drift in hybrid search** — BM25 vs Lucene. Mitigate: per-backend calibrated constants behind the `SearchBackend` protocol; golden-set regression tests.
3. **Silent dialect divergence in agent-generated Cypher** — text2cypher may emit Ladybug-isms. Mitigate: common-subset corpus + query validator per dialect (exactly the guard layer used for Ladybug today).
4. **Schema enforcement asymmetry** — Ladybug rejects untyped writes; CE Neo4j won't. Mitigate: Pydantic models remain the enforcement point before the driver is involved (already the case: `NodeDataCollection` is typed).
5. **Vector index population timing** — queries right after creation may return nothing. Mitigate: poll `SHOW INDEXES` for population state in build flow.
6. **Artifact schema coupling** — the JSON-serialisation change (§5) must version `manifest.json`; old artifacts get invalidated and rebuilt.
7. **`neo4j-admin` import is all-or-nothing into an empty DB** — always pair it with the incremental loader; treat bulk import as "bootstrap", `UNWIND` as "sync".
8. **Version pinning** — graph types / `SEARCH` clause are recent; feature-detect (`SHOW FUNCTIONS` / version check) and degrade rather than hard-require EE.

---

## 8. Alternatives considered

| Option | Description | Assessment |
|---|---|---|
| **A. One-way publish** | Keep Ladybug as the only engine; push Parquet artifacts to Neo4j for consumption only (no Neo4j backend code) | Cheapest; no query portability work; but Neo4j side can't participate in ingestion/hybrid search tooling — a dead-end if Neo4j usage grows |
| **B. Dual backend, opt-in (recommended)** | `KgBackend` abstraction with dialect layer; Neo4j implemented behind config; shared Parquet artifacts | Matches agreed positioning; moderate scope; keeps Ladybug default and CI cheap |
| **C. Neo4j primary** | Invert default; Ladybug for dev | Highest disruption; loses embedded STRUCT, gains ops burden; rejected for now |
| **D. Virtualisation/ federation layer** (e.g. keep Arrow as the store, Neo4j as index) | Complexity and poor Cypher coverage | Rejected |
| **E. Storage-agnostic GQL rewrite** | Replace Cypher usage with an internal query IR | Large rewrite; text2cypher ecosystem would need rework; revisit only if a third backend appears |

---

## 9. Open questions

1. **Neo4j property keys** — do we keep snake_case keys everywhere (current target mapping `irisCode → iris_code` implies yes) and generate camelCase mapping only when reading foreign Neo4j sources?
2. **Embedded structs** — for the Neo4j target: JSON-string columns (simpler, lossy for agents) vs flattened properties (queryable, schema-bloat). Preference?
3. **Temporal properties** — store timestamps as Neo4j `DATETIME` (native, queryable) vs ISO strings (parity with today's `_created_at` strings)? Native is nicer but changes the Parquet artifact types.
4. **Bulk-import tooling UX** — Docker-composed Neo4j managed by the CLI (`cli kg create --backend neo4j` spins up container, bulk-imports, online-syncs), or left as documented manual steps initially?
5. **AuraDB** — is it a realistic deployment target to design tests for, or only self-hosted Docker for the foreseeable future?

---

## 10. Rough phasing & effort

| Phase | Content | Rough size |
|---|---|---|
| 1. Dialect layer | TypeMapper, DdlRenderer, refactor merge/create_schema + Neo4j-import mapping reuse | S–M |
| 2. Parquet loaders | admin-import transformer + online `UNWIND` loader, CLI commands, manifest fields | M |
| 3. Neo4jBackend | Full `KgBackend` impl, result normalisation, vector/FTS shims | M |
| 4. Query portability | SearchBackend protocol, hybrid-search calibration, dual-backend test corpus | M–L (calibration is open-ended) |
| 5. Config/docs | Config schema, docs, docker-compose for CI | S |

Phases 1–3 alone yield a working opt-in Neo4j backend for build-and-query on the non-hybrid paths; Phase 4 is where behavioural parity is polished.

---

## 11. References

- Neo4j Cypher manual — graph types: https://neo4j.com/docs/cypher-manual/current/schema/graph-types/set-graph-types/
- Neo4j vector storage & indexes (property types, `CREATE VECTOR INDEX`, query procedures)
- Neo4j full-text indexes & analyzers; `SEARCH` clause (2026.09+)
- `neo4j-admin database import` — parquet format, headers, `:START_ID/:END_ID/:TYPE`
- Internal: `genai_graph/kg/backend.py`, `genai_graph/kg/ingest/merge.py`, `genai_graph/kg/ingest/extract.py`, `genai_graph/kg/export/artifacts.py`, `genai_graph/kg/factories/neo4j_factory.py`, `genai_graph/neo4j_import/schema_analyzer.py`
