"""Prefect flows for orchestrating knowledge graph creation."""

from __future__ import annotations

from prefect import flow, get_run_logger
from prefect.artifacts import create_markdown_artifact
from prefect.task_runners import ThreadPoolTaskRunner

from genai_graph.core.kg_context import KgContext
from genai_graph.orchestration.tasks import (
    KgRunResult,
    create_schema_task,
    delete_backend_task,
    export_html_task,
    ingest_subgraphs_task,
    initialize_backend_task,
    load_factories_task,
    resolve_config_task,
    summarize_warnings_task,
)


# Kuzu is an embedded database; we must avoid multi-process execution.
# A single-worker thread pool keeps all access in one process while still
# going through Prefect's task infrastructure.
@flow(name="create_kg_flow", task_runner=ThreadPoolTaskRunner(max_workers=1))
def create_kg_flow(
    config_name: str | None = None,
    delete_first: bool = False,
    export_html: bool = True,
) -> KgRunResult:
    """Create the knowledge graph and ingest documents using Prefect.

    The high-level steps are:
    1. Optional backend deletion (fresh start).
    2. Resolve KG configuration name and load configuration.
    3. Initialize graph backend (Kuzu or other).
    4. Pass 1: load subgraph factories and create schemas.
    5. Pass 2: ingest documents into the graph.
    6. Collect warnings and create Prefect artifacts.
    7. Optionally export an HTML visualization of the KG.
    """

    logger = get_run_logger()

    if delete_first:
        logger.info("Deleting existing backend before KG creation")
        delete_backend_task.submit()

    cfg_name, kg_cfg = resolve_config_task.submit(config_name).result()

    context = KgContext(config_name=cfg_name, config_dict=kg_cfg)

    backend_handle = initialize_backend_task.submit().result()

    bundles = load_factories_task.submit(kg_cfg, context).result()
    bundles = create_schema_task.submit(bundles, backend_handle.backend, context).result()  # type: ignore[name-defined]

    stats = ingest_subgraphs_task.submit(bundles, backend_handle.backend, context).result()
    warnings = summarize_warnings_task.submit(context).result()

    # Create a markdown artifact summarizing the run
    summary_lines: list[str] = [
        "# KG Creation Summary",
        "",
        f"**Config name:** `{cfg_name}`",
        f"**DB path:** `{backend_handle.db_path}`",
        "",
        "## Document statistics",
        f"- Processed: {stats.total_processed}",
        f"- Failed: {stats.total_failed}",
        f"- Nodes created: {stats.nodes_created}",
        f"- Relationships created: {stats.relationships_created}",
        "",
    ]

    if warnings:
        summary_lines.append("## Warnings")
        summary_lines.extend([f"- {w}" for w in warnings])
    else:
        summary_lines.append("## Warnings")
        summary_lines.append("- None")

    # Creating artifacts requires a running Prefect server; treat this as
    # best-effort so local CLI invocations and tests do not fail if no
    # server is available.
    try:  # pragma: no cover - network / environment dependent
        create_markdown_artifact("\n".join(summary_lines), key="kg-create-summary")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Failed to create Prefect artifact for KG summary: %s",
            exc,
        )

    html_result = None
    if export_html:
        html_result = export_html_task.submit(cfg_name, backend_handle.backend).result()

    return KgRunResult(
        config_name=cfg_name,
        backend=backend_handle,
        stats=stats,
        warnings=warnings,
        html_export=html_result,
    )
