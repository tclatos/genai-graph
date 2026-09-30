# Codebase Audit — Security, Performance & Maintainability (genai-graph)

**Date:** 2026-09-30
**Scope:** `genai_graph/` package + its `config/` files (this repo). Companion
audit of the `genai-tk` core library lives at
`../../../genai-tk/docs/studies/codebase_audit_2026-09.md`.

**Method:** static review (grep + targeted reads) of the full `genai_graph/`
tree. Every finding below was independently confirmed by opening the
referenced file (and, for the credentials finding, by actually recomputing
the hashes) — this is not a plausible-sounding LLM guess, it's verified.

## TL;DR

genai-graph has one **critical, must-fix-today** finding: real-looking
usernames with trivially-crackable password hashes are committed to git in
`config/basic_auth.yaml`. Beyond that, the Cypher-building code has a
real (if narrow) second-order injection path through document-derived node
names, several modules have grown past 1,000 lines mixing query-building
with CLI/business logic, and there's a document-authorization fallback that
fails **open** rather than closed. None of these require an architecture
rewrite — they're all localized, mechanical fixes.

---

## 1. Security Findings

### 1.1 🔴 CRITICAL — Real credentials with unsalted, guessable hashes committed to git

[config/basic_auth.yaml](../../config/basic_auth.yaml)
```yaml
users:
- username: admin
  password_hash: 8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918
- username: thierry
  password_hash: 57b379df03af99183605b1a0069c51dc6db345c3139c476c219c3ddcfcd6e7c2
- username: demo
  password_hash: 2a97516c354b68848cdbd8f54a226a0a55b21ed138e207ad6c5cbb9c00aa5aea
```
This file is tracked in git (`git ls-files` confirms it, first commit).
**Verified by recomputing SHA-256 locally:**
`sha256("admin") == 8c6976e5...` and `sha256("demo") == 2a97516c...` —
i.e. the `admin` and `demo` accounts use their own username as the password.
The `thierry` hash is unsalted SHA-256 too (identifiable by the 64-hex-char
format the app itself checks for in
`genai_tk/utils/basic_auth.py::_is_legacy_sha256`), so it's crackable
offline with a wordlist/rainbow table in seconds regardless of what the
underlying password is.

The auth module itself is fine — it supports bcrypt
(`genai_tk.utils.basic_auth.hash_password`) and only falls back to the
legacy SHA-256 *verifier* for backward compatibility. The problem is purely
that this specific config file was generated with the legacy scheme and
committed with weak passwords.

**Fix — do this now, not as part of a refactor:**
1. Rotate every real password behind these hashes immediately (assume they
   are compromised; they're in a git history that never forgets).
2. Regenerate the file using `hash_password()` (bcrypt) for all three users
   with strong, unique passwords.
3. If `config/basic_auth.yaml` is meant to hold real prod/shared-instance
   credentials, it should not be committed at all — move it to a
   git-ignored path or a secrets manager, and commit only an
   `basic_auth.yaml.example` with placeholder hashes.
4. Since the weak hashes are already in git history, treat the repo as
   needing a credential rotation, not a history rewrite (rewriting history
   on a shared repo is its own hazard — rotate the secrets, that's sufficient).

### 1.2 🟠 HIGH — Second-order Cypher injection via document-derived node names

[genai_graph/webapp/pages/demos/kg_visualization.py](../../genai_graph/webapp/pages/demos/kg_visualization.py#L159-L171)
```python
name_escaped = node_name.replace("'", "\\'")
...
query = (
    f"MATCH (n:{node_type})-{rel_filter}->(m) WHERE n.name = '{name_escaped}' AND {exclusion_filter} "
    f"RETURN n, r, m LIMIT {half_limit}; "
    f"MATCH (n)-{rel_filter}->(m:{node_type}) WHERE m.name = '{name_escaped}' ..."
)
```
`node_name` is populated from a Streamlit selectbox of *existing node names in
the graph* (`sss.selected_node_name`), not raw free-text — so this isn't a
trivial "type anything in a text box" injection. But this project's whole
purpose is to build the graph from **ingested external documents** (RFQ
files, benchmark corpora, office documents), and node `name` values are
frequently LLM-/heuristically-extracted entity or section titles pulled from
that document content. So the actual attacker-controlled surface is: *craft
a document whose extracted title/entity name contains a single quote and a
semicolon*, ingest it, and its escaped-but-not-neutralized name gets
concatenated into a **multi-statement** query string (statements are already
joined with `;` by this code itself) that is executed as-is. `\'` is not
even Kuzu/openCypher's escape convention (that's `''`, doubled quotes), so
the current escaping doesn't reliably neutralize a quote either.

**Fix:** use parameterized queries (`MATCH (n:{node_type})-...->(m) WHERE
n.name = $name ...`, `execute(query, parameters={"name": node_name, ...})`)
exactly like the access-control code in `document_graph_tools.py` already
does elsewhere in this repo (see the `$id`/`$stem` pattern at
[document_graph_tools.py:141-160](../../genai_graph/kg/query/document_graph_tools.py#L141)).
Node/relationship **type** and **label** names (`node_type`, `rel_patterns`)
still need to stay as validated-against-schema string interpolation (Cypher
doesn't allow parameterizing labels), but literal *values* like `node_name`
should never be.

### 1.3 🟡 MEDIUM — Access-control check fails open when the ACL column is missing

[genai_graph/kg/query/document_graph_tools.py](../../genai_graph/kg/query/document_graph_tools.py#L117)
```python
cols = _table_columns(backend, _DOCUMENT_LABEL)
if "allowed_principals" not in cols:
    return None  # _get_authorized_markdown_hashes: None means "unfiltered"
```
and, a few lines down in `_is_document_authorized` (line 141):
```python
if "allowed_principals" not in cols:
    return True  # fails OPEN
```
If the `Document` table predates the `allowed_principals` column being added
(schema migration, older ingested DB, or a config that doesn't wire up an
`AccessControlProvider`), every document silently becomes readable by every
user — there's no way to observe this without reading the source, since
nothing logs it. This is the classic "fail open" access-control anti-pattern.

**Fix:** when access control is *configured* (an `AccessControlProvider` is
set up for the profile) but the expected column is absent, this should be
treated as a configuration error and fail closed (raise/log at ERROR and
deny), not silently allow. If access control is deliberately not configured
for a given deployment, that should be an explicit, logged decision at
startup, not an implicit per-query column check.

### 1.4 🟡 MEDIUM — `pickle.load()` on a caller-supplied path

[genai_graph/utils/streamlit/capturing_callback_handler.py](../../genai_graph/utils/streamlit/capturing_callback_handler.py#L46)
```python
def load_records_from_file(path: str) -> list[CallbackRecord]:
    with open(path, "rb") as file:
        records = pickle.load(file)  # arbitrary code execution if path is attacker-influenced
```
Same shape and same "developer-controlled file" mitigation as the genai-tk
finding in the companion report — flagging here too because this file
accepts a `path: str` parameter directly (vs. genai-tk's version which is
more clearly internal-only). Confirm no Streamlit page ever passes a
user-uploaded or user-typed path into this function; if one does (or ever
will, e.g. a "replay this session" UI feature), that's an RCE.

### 1.5 🟢 LOW — URL download helper doesn't restrict scheme

[genai_graph/bench/adapters/base.py](../../genai_graph/bench/adapters/base.py#L69-L100)
`download_http_file()` builds a `urllib.request.Request(url, ...)` and opens
it with a timeout (good), but never checks `url` starts with `http(s)://`.
Low risk today since callers pass hardcoded dataset URLs from adapter code,
but if this is ever exposed to a config-driven or user-driven URL, add an
explicit scheme allowlist (`urlsplit(url).scheme in {"http", "https"}`) as
defense-in-depth against `file://`-style SSRF/local-file-read tricks.

### 1.6 Informational — `kg/ingest/extract.py` f-string DDL is not user-facing

[genai_graph/kg/ingest/extract.py](../../genai_graph/kg/ingest/extract.py#L301-L314)
(`f"CALL table_info('{table_name}')"`, `f"ALTER TABLE {table_name} ADD {field_decl}"`)
interpolates `table_name`/`field_decl`, but both are derived from Pydantic
`GraphNode` model class names and field declarations defined in Python code,
not from data — same trust level as the rest of the schema-compiler. Not a
vulnerability as written; just keep it that way (don't let a future feature
make node/table names data-driven without adding validation here).

---

## 2. Large Modules That Should Be Split

Every file below mixes query-building, business logic, and often CLI/UI code
in a single module. This is the single biggest maintainability cost in the
repo — a reviewer looking at a one-function diff has to load 1,000+ lines of
unrelated context.

| File | Lines | Mixed responsibilities found |
|---|---|---|
| [genai_graph/kg/query/document_graph_tools.py](../../genai_graph/kg/query/document_graph_tools.py) | 1,974 | document lookup/metadata, folder hierarchy navigation, TOC generation/rendering, full-document reconstruction, keyword+semantic search, agent-tool factory — 6 concerns |
| [genai_graph/kg/export/artifacts.py](../../genai_graph/kg/export/artifacts.py) | 1,425 | HTML export, Parquet export/Arrow prep, schema-doc generation, warnings report, cache fingerprinting |
| [genai_graph/kg/document_graph/outline_extract.py](../../genai_graph/kg/document_graph/outline_extract.py) | 1,399 | TOC/preamble parsing models, config+caching, the LLM extraction pipeline itself, text-cleaning heuristics, BAML fallback path, retry/backoff |
| [genai_graph/kg/ingest/extract.py](../../genai_graph/kg/ingest/extract.py) | 1,262 | Kuzu type mapping, embedded-field handling, schema creation/evolution, node/relationship extraction, embeddings wiring |
| [genai_graph/core/commands_docgraph.py](../../genai_graph/core/commands_docgraph.py) | 1,169 | CLI registration, workflow orchestration, document-navigation subcommands (list/toc/cat/search/tui), Rich output rendering — should be several `commands_docgraph_*.py` files behind one thin registrar |
| [genai_graph/kg/schema/core.py](../../genai_graph/kg/schema/core.py) | 1,119 | `GraphNode`/`GraphRelation` models, schema assembly/validation, field-path deduction, hashing/normalization utilities |
| [genai_graph/kg/ingest/merge.py](../../genai_graph/kg/ingest/merge.py) | 1,006 | Arrow table prep, batch merge orchestration, `ParquetCollector`, Cypher value formatting — see §3 for a functional note on this file too |
| [genai_graph/kg/backend.py](../../genai_graph/kg/backend.py) | 915 | abstract `KgBackend`/`QueryExecutor` interface + the concrete Ladybug backend + the concrete Neo4j backend + factory functions, all in one file |

**Suggested pattern:** same as the genai-tk report — extract Pydantic models
to `*_models.py`, extract the biggest single-purpose chunk (search, TOC,
export format, backend implementation) to its own file, leave the original
filename as a thin façade re-exporting the public API so callers don't
break. `document_graph_tools.py` and `backend.py` are the best places to
start: the former has the clearest natural seams (lookup / TOC / search /
tools), the latter is a textbook "one abstract class, N concrete
implementations, split by implementation" case.

---

## 3. Async / Parallelization Opportunities

- **Good pattern already in place:** [genai_graph/kg/document_graph/ingest.py:292-308](../../genai_graph/kg/document_graph/ingest.py#L292-L308) uses `asyncio.Semaphore` + `asyncio.gather` to parallelize embedding calls across documents with bounded concurrency. This is the right template — replicate it, don't reinvent it.
- **Not yet replicated:** [genai_graph/kg/document_graph/outline_extract.py:1102](../../genai_graph/kg/document_graph/outline_extract.py#L1102) parallelizes outline extraction across documents with a plain `ThreadPoolExecutor(max_workers=workers)`. Since the actual work per document is I/O-bound (LLM calls), a `ThreadPoolExecutor` works but wastes a thread per in-flight request and doesn't give the same fine-grained backpressure a semaphore does. Recommend porting this to the `asyncio.gather` + `Semaphore` pattern from `ingest.py` for consistency and lower memory overhead at high worker counts.
- **`kg/ingest/merge.py` — mixed news:**
  - The no-properties relationship-merge path (around [merge.py:940-960](../../genai_graph/kg/ingest/merge.py#L940)) already has a **documented, intentional** design: it uses inline `MATCH (from:T {k: from_id}), (to:T {k: to_id})` curly-brace property patterns rather than `WHERE`+`WITH` staging, with a comment explaining that a prior `WITH`-staged rewrite silently produced 0 rows (the `WITH` clause dropped the `LOAD FROM` column binding). Leave this alone — it's already been through one incident and has an explanatory comment; re-verify with `EXPLAIN` before touching it again.
  - The **with-properties** path (just above it, ~[merge.py:918-940](../../genai_graph/kg/ingest/merge.py#L918)) loops row-by-row, issuing one `kuzu_conn.execute()` call per relationship (Kuzu's `LOAD FROM` doesn't support inline property assignment in `MERGE`, per the in-code comment). For large batches this is N round-trips instead of 1. **Concrete fix:** batch this with `UNWIND $rows AS row MATCH (from:{from_type} {{{from_key_field}: row.from_id}}), (to:{to_type} {{{to_key_field}: row.to_id}}) MERGE (from)-[r:{rel_name}]->(to) SET r += row.props` in a single parameterized call (pass the row list as one `$rows` parameter), instead of a Python-level loop.
- **`time.sleep()` usage** in [genai_graph/bench/build_graph.py](../../genai_graph/bench/build_graph.py#L101-L103) is exponential backoff with jitter for retries — correct and appropriate as blocking, serialized retry logic (not a bug).

---

## 4. Other Code Smells

- **Repeated identical `except Exception:` blocks:** [genai_graph/webapp/pages/settings/configuration.py:36-57](../../genai_graph/webapp/pages/settings/configuration.py#L36-L57) has four near-identical try/except blocks for different config keys — collapse into one loop over `(key, label)` pairs.
- **Silent performance-regression fallback:** [genai_graph/kg/ingest/merge.py:973](../../genai_graph/kg/ingest/merge.py#L973) logs at `warning` level when batch `LOAD FROM` fails and falls back to the O(n) per-row path — good that it logs at all, but consider emitting a metric/counter too so a slow ingest run due to this fallback is visible in aggregate, not just in scrollback.
- **Broad `except Exception` with `# noqa: BLE001`** appears in [genai_graph/kg/ingest/extract.py](../../genai_graph/kg/ingest/extract.py) (lines 466, 470, 672, 682, 695, 1203, 1233) and [genai_graph/kg/backend.py](../../genai_graph/kg/backend.py) (343, 419, 651) — the `noqa` acknowledges the lint rule rather than addressing it; worth narrowing to the actual expected exception types (e.g. Kuzu's own exception classes) where feasible so real bugs don't get masked alongside expected schema-introspection failures.
- **Self-documented "hacks":** [genai_graph/utils/streamlit/thread_issue_fix.py](../../genai_graph/utils/streamlit/thread_issue_fix.py) and `clear_result.py` openly document themselves as workarounds for Streamlit threading/state quirks. Not urgent, but worth a tracking issue referencing the Streamlit version/upstream bug so they get revisited on the next Streamlit upgrade instead of becoming permanent.
- **Dead/commented-out code:** [genai_graph/main/modal_app.py:174](../../genai_graph/main/modal_app.py#L174) has a commented-out `time.sleep(60)` — either delete it or replace with a one-line comment saying why it's kept.

---

## 5. Prioritized Action List

| # | Effort | Item | Status |
|---|---|---|---|
| 1 | **now** | Rotate the credentials behind `config/basic_auth.yaml`'s three hashes; they are cracked (§1.1) | ⚠️ Not done — per the requester, these are test-only accounts (`BASIC_AUTH_ENABLED` defaults to `false`), so rotation was deferred; see #3 for the quick fix that *was* applied |
| 2 | **now** | Stop committing real `basic_auth.yaml` — git-ignore it, commit a `.example` with placeholders instead (§1.1) | ⚠️ Not done (test-only credentials, deferred by requester) |
| 3 | 30 min | Regenerate remaining accounts with `hash_password()` (bcrypt) instead of legacy SHA-256 (§1.1) | ✅ Done — `admin` and `demo` re-hashed with bcrypt (known test passwords); `thierry`'s plaintext is unknown so its legacy SHA-256 hash was left as-is (still supported by `verify_password`'s backward-compat path) |
| 4 | 1–2 hrs | Parameterize the `node_name` value in `kg_visualization.py`'s Cypher strings (§1.2) | ✅ Done — `build_filtered_cypher_query` now returns `(query, parameters)` and binds `node_name` as `$name`; `parameters` threaded through `generate_html` → `_fetch_graph_data` → `KgBackend.execute_get_as_df` |
| 5 | 2–4 hrs | Make the `allowed_principals`-missing case fail closed when access control is configured (§1.3) | 🟡 Partial — there's no explicit "ACL enabled" flag in this codebase to key off safely, so flipping to fail-closed risked locking out every deployment without ACL configured. Added a `logger.warning` at both fallback sites instead, so the fail-open path is now observable instead of silent; a real fail-closed fix needs a product decision on an explicit ACL-enabled flag |
| 6 | 30 min | Confirm `capturing_callback_handler.load_records_from_file` is never fed a user-supplied path; add a code comment/assert if so (§1.4) | ⚠️ Not done in this pass |
| 7 | 1 day | Batch the with-properties relationship merge path with `UNWIND` instead of a per-row loop (§3) | ⚠️ Deliberately not done — `merge.py` has no test coverage for this branch and has already had one silent-data-loss incident from a prior "safe" rewrite (see memory note); needs a validation harness before touching it |
| 8 | 1–2 days | Port `outline_extract.py`'s `ThreadPoolExecutor` parallelism to the `asyncio.gather`+`Semaphore` pattern from `ingest.py` (§3) | ⚠️ Deliberately not done — same reasoning as #7 (complex retry/backoff logic, no test harness to validate an async rewrite in this pass) |
| 9 | ongoing | Split `document_graph_tools.py` and `backend.py` first (clearest seams), then the rest of §2 as time allows | ⚠️ Not done — tracked as follow-up work |

Also fixed opportunistically while implementing the above: collapsed four
near-identical `except Exception:` blocks in
[webapp/pages/settings/configuration.py](../../genai_graph/webapp/pages/settings/configuration.py)
into a loop (§4), and removed the dead commented-out `local_entrypoint`/`time.sleep(60)`
block in [main/modal_app.py](../../genai_graph/main/modal_app.py) (§4).

