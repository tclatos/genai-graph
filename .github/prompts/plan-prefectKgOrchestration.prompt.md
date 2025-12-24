# Plan: Implement Prefect Orchestration for KG Injection Pipeline

Migrate the KG creation workflow from direct CLI execution to Prefect-orchestrated flows for improved observability, resilience, and operational management while preserving existing CLI behavior.

## Steps

### 1. create orchestration module structure


- Create new directory `genai_graph/orchestration/` with files: `__init__.py`, `flows.py`, `tasks.py`

### 2. Decompose KG creation into reusable Prefect tasks

- Extract logic from `create()` command (genai_graph/core/commands_ekg.py#L101-L230) into discrete `@task` functions in `tasks.py`
- Create tasks: `validate_config_task()`, `initialize_backend_task()`, `load_factory_task()`, `create_schema_task()`, `ingest_subgraph_task()`
- Preserve `KgContext` warning collection and `DocumentStats` aggregation patterns

### 3. Implement main KG creation flow

- Create `create_kg_flow()` in `flows.py` with `SequentialTaskRunner` for single-process execution
- Orchestrate tasks: config validation → backend initialization → factory loading → schema creation (Pass 1) → document ingestion (Pass 2)
- Return aggregated stats and warnings, integrate with Prefect's logging via `get_run_logger()`

### 4. Add optional Prefect execution mode to existing CLI

- Modify `kg create` command (genai_graph/core/commands_ekg.py#L101) to accept `--orchestration` flag with options: `direct` (default, current behavior) or `prefect`
- Import and invoke `create_kg_flow()` when Prefect mode selected
- Keep existing direct implementation as default for backward compatibility

### 5. Configure retry logic and observability features

- Add retry decorators to tasks: `create_schema_task(retries=2)`, `ingest_subgraph_task(retries=3, retry_delay_seconds=60)`
- Implement Prefect artifacts for HTML exports and execution stats
- Set up `persist_result=False` for tasks handling database connections (avoid serialization)

### 6. Create deployment configuration and update documentation

- Add `genai_graph/orchestration/deployments.py` with sample deployment configs for local agent and Prefect Cloud
- Update `README.md` with Prefect setup instructions and usage examples
- Add integration tests in `tests/integration_tests/` for flow execution

## Further Considerations

### 1. Incremental vs full migration

Start with simple flow wrapper around existing `create()` logic, then progressively decompose into granular tasks based on observability needs?

### 2. Parallel subgraph processing

Current implementation is sequential; consider using `ConcurrentTaskRunner` for independent subgraphs (DB writes may serialize anyway due to Kuzu's embedded nature)?

### 3. Secondary flows

Should `delete` and `export-html` commands also become Prefect flows, or keep them as direct CLI for simplicity?
