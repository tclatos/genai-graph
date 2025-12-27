"""Prefect tasks for orchestrating knowledge graph creation.

These tasks wrap the existing core KG creation primitives into Prefect tasks,
so that we can build observable, resilient flows while preserving the
underlying behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from genai_tk.utils.config_mngr import import_from_qualified
from loguru import logger
from prefect import get_run_logger, task
from prefect.exceptions import MissingContextError
from pydantic import BaseModel

from genai_graph.core.graph_backend import (
    GraphBackend,
    create_backend_from_config,
    delete_backend_storage_from_config,
    get_backend_storage_path_from_config,
)
from genai_graph.core.graph_core import create_schema as core_create_schema
from genai_graph.core.graph_documents import DocumentStats, add_documents_to_graph
from genai_graph.core.graph_html import generate_html
from genai_graph.core.graph_schema import GraphSchema
from genai_graph.core.kg_manager import get_kg_manager
from genai_graph.core.subgraph_factories import SubgraphFactory, TableBackedSubgraphFactory


def _get_prefect_logger_or_default() -> Any:
    """Return a Prefect run logger when in a flow, else fall back to loguru logger.

    This allows the same task functions to be reused both inside Prefect flows
    and in direct local execution (e.g. from the CLI) without requiring a
    running Prefect server.
    """

    try:
        return get_run_logger()
    except MissingContextError:
        return logger


class SubgraphBundle(BaseModel):
    """In-memory representation of a configured subgraph during KG creation."""

    config: dict[str, Any]
    factory: SubgraphFactory
    # Schema type is kept as Any to avoid circular imports in type checkers
    schema_obj: GraphSchema | None = None


class HtmlExportResult(BaseModel):
    """Result of HTML export task."""

    config_name: str
    output_path: Path


class KgRunResult(BaseModel):
    """Aggregated result of a KG creation run."""

    config_name: str
    db_path: Path
    stats: DocumentStats
    warnings: list[str]
    html_export: HtmlExportResult | None = None


@task
def resolve_config_task(config_name: str | None) -> tuple[str, dict[str, Any]]:
    """Resolve KG profile and return its configuration dictionary via KgManager."""

    logger_pf = _get_prefect_logger_or_default()

    from genai_graph.core.kg_manager import get_kg_manager

    manager = get_kg_manager()
    effective, _ = manager.activate()
    kg_cfg = manager.get_profile_dict()

    logger_pf.info(
        "Loaded KG config '%s', subgraphs=%d.",
        effective,
        len(kg_cfg.get("subgraphs", [])),
    )

    return effective, kg_cfg


@task
def initialize_backend_task(config_key: str = "default", kg_config_name: str | None = None) -> GraphBackend:
    """Create and return the graph backend instance.

    The flow is expected to run with a single-process task runner so that the
    embedded Kuzu backend is never accessed concurrently from multiple
    processes.

    Args:
        config_key: Key in graph_db config section
        kg_config_name: Optional KG configuration name for organized output folders
    """

    logger_pf = _get_prefect_logger_or_default()

    backend = create_backend_from_config(config_key, kg_config_name)
    db_path = get_backend_storage_path_from_config(config_key, kg_config_name)

    logger_pf.info("Initialized backend '%s' at path '%s'", config_key, db_path)
    return backend


@task
def load_factories_task(kg_cfg: dict[str, Any]) -> list[SubgraphBundle]:
    """Load and instantiate subgraph factories from KG configuration."""

    logger_pf = _get_prefect_logger_or_default()
    manager = get_kg_manager()
    subgraphs_cfg = kg_cfg.get("subgraphs", [])

    bundles: list[SubgraphBundle] = []
    for subgraph_cfg in subgraphs_cfg:
        if not isinstance(subgraph_cfg, dict):
            continue

        factory_path = subgraph_cfg.get("factory")
        if not factory_path:
            continue

        try:
            imported = import_from_qualified(factory_path)
            if isinstance(imported, SubgraphFactory):
                subgraph_impl = imported
            elif isinstance(imported, type) and issubclass(imported, SubgraphFactory):
                constructor_kwargs = {
                    k: v for k, v in subgraph_cfg.items() if k not in {"factory", "initial_load", "trigger"}
                }
                subgraph_impl = imported(**constructor_kwargs)  # type: ignore[misc]
            else:
                msg = f"Factory {factory_path} is not a SubgraphFactory"
                logger.warning(msg)
                manager.add_warning(msg)
                continue

            logger_pf.info("Loaded subgraph factory: %s", subgraph_impl.name)
            bundles.append(SubgraphBundle(config=subgraph_cfg, factory=subgraph_impl))
        except Exception as exc:  # pragma: no cover - defensive logging
            import traceback

            msg = f"Failed to import factory {factory_path}: {exc}"
            logger.error(msg)
            logger.error(traceback.format_exc())
            manager.add_warning(msg)

    return bundles


@task
def create_schema_task(
    bundles: list[SubgraphBundle],
    backend: GraphBackend,
) -> list[SubgraphBundle]:
    """Create graph schema for all loaded subgraphs (Pass 1)."""

    logger_pf = _get_prefect_logger_or_default()
    manager = get_kg_manager()

    for bundle in bundles:
        subgraph_impl = bundle.factory
        subgraph_impl.register()

        schema = subgraph_impl.build_schema()
        try:
            core_create_schema(backend, schema.nodes, schema.relations, manager)
            schema.validate_with_context(manager)
            logger_pf.info(
                "Created schema for subgraph '%s'",
                getattr(subgraph_impl, "name", "<unknown>"),
            )
        except Exception as exc:  # pragma: no cover - defensive
            import traceback

            msg = f"Schema creation failed for subgraph {getattr(subgraph_impl, 'name', '<unknown>')}: {exc}"
            logger.error(msg)
            logger.error(traceback.format_exc())
            manager.add_warning(msg)

        bundle.schema_obj = schema

    return bundles


@task
def ingest_subgraphs_task(
    bundles: list[SubgraphBundle],
    backend: GraphBackend,
) -> DocumentStats:
    """Ingest documents for all configured subgraphs (Pass 2)."""

    logger_pf = _get_prefect_logger_or_default()
    manager = get_kg_manager()

    total_stats = DocumentStats()

    for bundle in bundles:
        subgraph_cfg = bundle.config
        subgraph_impl = bundle.factory
        schema = bundle.schema_obj

        factory_path = subgraph_cfg.get("factory", "<unknown>")

        # For table-backed subgraphs configured with `pull`, do not
        # load all rows by default. They act as an on-demand source.
        pull_cfg = getattr(subgraph_impl, "pull", None)
        keys = subgraph_cfg.get("initial_load", [])

        if not keys and pull_cfg and isinstance(subgraph_impl, TableBackedSubgraphFactory):
            logger_pf.info(
                "Skipping automatic ingestion for pull-only subgraph: %s",
                getattr(subgraph_impl, "name", factory_path),
            )
            continue

        if not keys and isinstance(subgraph_impl, TableBackedSubgraphFactory):
            try:
                keys = subgraph_impl.get_all_keys()
                logger_pf.info(
                    "Retrieved %d keys from table-backed factory",
                    len(keys),
                )
            except Exception as exc:  # pragma: no cover - defensive
                msg = f"Failed to get keys from table for {factory_path}: {exc}"
                logger.warning(msg)
                manager.add_warning(msg)
                keys = []

        if not keys:
            continue

        try:
            assert schema is not None, "Schema must be created before ingestion"
            stats = add_documents_to_graph(keys, subgraph_impl, backend, schema, manager)
            logger_pf.info(
                "Ingest stats for %s: processed=%d failed=%d nodes=%d rels=%d",
                factory_path,
                stats.total_processed,
                stats.total_failed,
                stats.nodes_created,
                stats.relationships_created,
            )
            total_stats.total_processed += stats.total_processed
            total_stats.total_failed += stats.total_failed
            total_stats.nodes_created += stats.nodes_created
            total_stats.relationships_created += stats.relationships_created
        except Exception as exc:  # pragma: no cover - defensive
            import traceback

            msg = f"Ingestion error for {factory_path}: {exc}"
            logger.error(msg)
            logger.error(traceback.format_exc())
            manager.add_warning(msg)
            # Assume all keys failed when we cannot be more precise
            total_stats.total_failed += len(keys)

    logger_pf.info(
        "Total ingest stats: processed=%d failed=%d nodes=%d rels=%d",
        total_stats.total_processed,
        total_stats.total_failed,
        total_stats.nodes_created,
        total_stats.relationships_created,
    )
    return total_stats


@task
def delete_backend_task(config_key: str = "default", kg_config_name: str | None = None) -> None:
    """Delete the knowledge graph backend storage for a given config key.

    Args:
        config_key: Key in graph_db config section
        kg_config_name: Optional KG configuration name for organized output folders
    """

    logger_pf = _get_prefect_logger_or_default()
    path = get_backend_storage_path_from_config(config_key, kg_config_name)

    if path.exists():
        logger_pf.info(
            "Deleting backend storage at '%s' for config '%s'",
            path,
            config_key,
        )
        delete_backend_storage_from_config(config_key, kg_config_name)
    else:
        logger_pf.info(
            "No backend storage found at '%s' for config '%s'",
            path,
            config_key,
        )


@task
def export_html_task(
    config_name: str,
    backend: GraphBackend,
    output_dir: Path | None = None,
) -> HtmlExportResult:
    """Export an HTML visualization of the current KG and return its path.

    Args:
        config_name: Name of the KG configuration
        backend: The graph backend to export from
        output_dir: Optional custom output directory (if None, uses KG outcome manager)
    """

    logger_pf = _get_prefect_logger_or_default()

    if output_dir is None:
        # Use KgManager for organized output
        from genai_graph.core.kg_manager import get_kg_manager

        manager = get_kg_manager()
        manager.activate()
        destination = manager.html_path
        manager.ensure_directories()
    else:
        # Custom output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"{config_name}_graph.html"

    generate_html(backend, destination_file_path=str(destination))
    logger_pf.info("Exported KG HTML visualization to '%s'", destination)

    return HtmlExportResult(config_name=config_name, output_path=destination)


@task
def summarize_warnings_task(config_name: str | None = None) -> list[str]:
    """Return collected warnings from KgManager and log them if config_name provided.

    Args:
        config_name: Optional KG configuration name to log warnings to file
    """

    logger_pf = _get_prefect_logger_or_default()
    manager = get_kg_manager()
    warnings = manager.get_warnings()

    if warnings:
        logger_pf.warning(
            "KG creation completed with %d warning(s)",
            len(warnings),
        )

        # Log warnings to file if config_name is provided
        if config_name:
            manager.activate()
            manager.log_warnings(warnings)
    else:
        logger_pf.info("KG creation completed with no warnings")

    return warnings
