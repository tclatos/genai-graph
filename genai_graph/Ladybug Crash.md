Title: Native heap corruption (delayed SIGSEGV/abort) when executing QUERY_FTS_INDEX / QUERY_VECTOR_INDEX

Environment
•  ladybug 0.16.1 (Python client, pybind backend), Ubuntu 24.04, Python 3.12 (WSL2)
•  FTS and vector extensions installed/loaded normally

Summary
Executing QUERY_FTS_INDEX (and QUERY_VECTOR_INDEX) corrupts the native heap. Queries return correct results, but the process later crashes — typically at interpreter/DB teardown or on a subsequent allocation.

Symptoms
•  SIGSEGV (exit code 139) or abort with free(): invalid pointer (exit code 134)
•  Crash can be delayed well past the offending query (teardown, later unrelated allocation)
•  Reproduces single-threaded, on a fresh database, and on a copy of the database
•  Concurrency and connection GC mid-run make the crash happen earlier, but are not required

Trigger chain observed
1. INSTALL/LOAD of fts/vector extensions alone → safe, exits 0
2. Pure Cypher workload (no index queries) → safe, exits 0
3. Any successful QUERY_FTS_INDEX execution → delayed crash
4. Even a failed CREATE_FTS_INDEX on a fresh DB (catalog exception: _CREATE_FTS_INDEX does not exist) followed by a query attempt → crash with free(): invalid pointer

Workaround
Avoid native index queries entirely; fall back to plain Cypher (CONTAINS-based filtering). With this, our workload (graph build + concurrent LLM-agent tool queries) runs long sessions with zero crashes.