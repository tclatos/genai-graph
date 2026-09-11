"""Workflow step wrappers for use with the genai-tk workflow engine.

These functions are thin adapters that translate workflow engine parameters
into calls to the existing Prefect flows. They are referenced by dotted path
in ``config/workflows.yaml``.
"""

from __future__ import annotations

from typing import Any

from genai_tk.workflow.registry import workflow
from loguru import logger


def _clear_factory_caches() -> None:
    from genai_graph.kg.factories import (
        JsonFileBackedFactory,
        MarkdownBamlFactory,
        Neo4jFactory,
        TableBackedFactory,
    )

    JsonFileBackedFactory.clear_cache()
    MarkdownBamlFactory.clear_cache()
    TableBackedFactory.clear_cache()
    Neo4jFactory.clear_cache()


def _result_dict(config_name: str, result: Any) -> dict[str, Any]:
    return {
        "config_name": config_name,
        "total_processed": result.stats.total_processed,
        "total_failed": result.stats.total_failed,
        "warnings_count": len(result.warnings),
        "db_path": str(result.db_path),
    }


@workflow(name="kg_create", description="Execute the KG creation flow for a given config profile")
def kg_create_step(
    *,
    config_name: str,
    delete_first: bool = False,
    export_html: bool = True,
    force_stage: str | None = None,
) -> dict[str, Any]:
    """Execute the KG creation flow for a given config profile.

    This wrapper handles:
    - Clearing factory caches (prevents cross-contamination)
    - Running the full create_kg_flow

    Args:
        config_name: KG profile name to build.
        delete_first: Whether to delete the existing database before building.
        export_html: Whether to export an HTML visualization.
        force_stage: One of `parquet`/`graph`/`embed`/`all` (see
            `genai_tk.workflow.force`). `parquet` (and above) rebuilds import
            caches; `graph` (and above) also drops the destination database.

    Returns a summary dict suitable for workflow engine result tracking.
    """
    from genai_tk.workflow.force import ForceStage, stage_active

    from genai_graph.orchestration.flows import create_kg_flow

    _clear_factory_caches()
    logger.info("Running KG creation flow for config: {}", config_name)

    result = create_kg_flow(
        config_name=config_name,
        delete_first=delete_first or stage_active(force_stage, ForceStage.graph),
        export_html=export_html,
        force_rebuild=stage_active(force_stage, ForceStage.parquet),
    )

    return _result_dict(config_name, result)


@workflow(name="kg_build", description="Build a KG from a single graph factory configuration", hidden=True)
def kg_build_step(
    *,
    graph: dict[str, Any],
    kg_name: str = "inline",
    delete_first: bool = False,
    export_html: bool = True,
    force_stage: str | None = None,
) -> dict[str, Any]:
    """Execute the KG creation flow with a single inline graph configuration.

    Instead of looking up a ``config_name`` in ``kg_configs``, this step
    receives a graph factory definition directly and registers it as a
    temporary KG profile before running the build flow.

    Args:
        graph: Graph factory configuration (a dict with a ``factory`` key,
            same format as entries in workflow YAML).
        kg_name: Name used for the database directory and profile identity.
        delete_first: Whether to delete existing database before building.
        export_html: Whether to export an HTML visualization.
        force_stage: One of `parquet`/`graph`/`embed`/`all` (see
            `genai_tk.workflow.force`). `parquet` (and above) rebuilds import
            caches; `graph` (and above) also drops the destination database.
    """
    from genai_tk.workflow.force import ForceStage, stage_active

    from genai_graph.kg.manager import KgGraphConfig, KgProfileConfig, get_kg_manager
    from genai_graph.orchestration.flows import create_kg_flow

    force_rebuild = stage_active(force_stage, ForceStage.parquet)
    if stage_active(force_stage, ForceStage.graph):
        delete_first = True

    _clear_factory_caches()

    # Register the inline graph as a temporary profile in the KgManager
    manager = get_kg_manager()
    profile_cfg = KgProfileConfig(graphs=[KgGraphConfig(**graph)])
    manager.ekg_config.kg_configs[kg_name] = profile_cfg
    manager.profile = kg_name
    manager.reset_cached_paths()

    logger.info("Running KG build flow for inline config '{}' with factory '{}'", kg_name, graph.get("factory", "?"))

    result = create_kg_flow(
        config_name=kg_name,
        delete_first=delete_first,
        export_html=export_html,
        force_rebuild=force_rebuild,
    )

    return _result_dict(kg_name, result)


@workflow(
    name="docgraph_build",
    description="Markdownize documents, then build a document graph + entity sub-graphs into one KG",
    hidden=True,
)
def docgraph_build_step(
    *,
    kg_name: str,
    sources: list[str] | str,
    factories: list[dict[str, Any]] | None = None,
    markdownize_profile: str | None = None,
    md_output_dir: str | None = None,
    cache_dir: str | None = None,
    build_document_graph: bool = True,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    delete_first: bool = False,
    export_html: bool = True,
    force_stage: str | None = None,
) -> dict[str, Any]:
    """Build a Document Graph + entity sub-graphs from a set of documents.

    Pipeline:
    1. Optionally markdownize *sources* (PPT/PDF/… or pre-existing Markdown) into
       *md_output_dir* (already-Markdown files are copied through).
    2. Run each entity *factory* (e.g. a `MarkdownBamlFactory` subclass) into a
       single KG named *kg_name* via the standard extraction flow.
    3. Optionally ingest the Folder → Document → Section document graph over the
       Markdown into the *same* database (Document nodes MERGE by content hash).

    Args:
        kg_name: Name used for the database directory and profile identity.
        sources: Directories/files/zip archives to ingest.
        factories: Entity factory configs (dicts with a ``factory`` key).
        markdownize_profile: When set, markdownize *sources* first with this profile.
        md_output_dir: Where converted Markdown is written (required when
            markdownizing or building the document graph).
        cache_dir: Markdownize intermediates directory.
        build_document_graph: Ingest the Folder/Document/Section graph.
        include: Glob patterns for document-graph ingestion (default ``['*.md']``).
        exclude: Glob patterns to exclude.
        delete_first: Delete the existing database before building.
        export_html: Export an HTML visualization after entity extraction.
        force_stage: Cache-invalidation stage (see `genai_tk.workflow.force`).
    """
    from genai_tk.workflow.force import ForceStage, stage_active

    from genai_graph.kg.manager import KgGraphConfig, KgProfileConfig, get_kg_manager
    from genai_graph.orchestration.flows import create_kg_flow

    source_list = [sources] if isinstance(sources, str) else list(sources)
    _clear_factory_caches()

    if stage_active(force_stage, ForceStage.graph):
        delete_first = True

    md_dir = md_output_dir
    if markdownize_profile:
        if not md_output_dir:
            raise ValueError("docgraph_build_step: md_output_dir is required when markdownize_profile is set")
        from genai_tk.workflow.markdownize import markdownize_flow

        logger.info("Markdownizing {} source(s) -> {}", len(source_list), md_output_dir)
        markdownize_flow(
            sources=source_list,
            md_output_dir=md_output_dir,
            cache_dir=cache_dir,
            profile=markdownize_profile,
            force_stage=force_stage,
        )
        md_dir = md_output_dir

    # --- entity sub-graphs (single KG profile) ---------------------------
    entity_result: Any = None
    if factories:
        manager = get_kg_manager()
        profile_cfg = KgProfileConfig(graphs=[KgGraphConfig(**f) for f in factories])
        manager.ekg_config.kg_configs[kg_name] = profile_cfg
        manager.profile = kg_name
        manager.reset_cached_paths()
        logger.info("Running entity extraction for '{}' ({} factory/ies)", kg_name, len(factories))
        entity_result = create_kg_flow(
            config_name=kg_name,
            delete_first=delete_first,
            export_html=export_html,
            force_rebuild=stage_active(force_stage, ForceStage.parquet),
        )
        delete_first = False  # already applied

    # --- document graph (same DB) ----------------------------------------
    doc_result: dict[str, Any] | None = None
    if build_document_graph:
        if not md_dir:
            raise ValueError("docgraph_build_step: md_output_dir is required to build the document graph")
        from genai_graph.kg.document_graph.build import build_document_graph as build_doc_graph
        from genai_graph.kg.manager import get_kg_manager

        doc_stats = build_doc_graph(
            sources=[md_dir],
            db_path=str(get_kg_manager().get_db_path_for(kg_name)),
            include=include or ["*.md"],
            exclude=exclude or [],
            delete_first=delete_first,  # already applied above when entity factories ran
            force=stage_active(force_stage, ForceStage.graph),
            # Keep this step's algo-only, retrieval-free document-graph behavior.
            structure_strategy="algo",
            outline_pre_pass=False,
            embeddings_id=None,
            fts=False,
        )
        doc_result = {
            "db_path": doc_stats["db_path"],
            "documents_processed": doc_stats["documents_processed"],
            "sections_created": doc_stats["sections_created"],
            "relationships_created": doc_stats["relationships_created"],
        }

    summary: dict[str, Any] = {"kg_name": kg_name}
    if entity_result is not None:
        summary.update(_result_dict(kg_name, entity_result))
    if doc_result is not None:
        summary["document_graph"] = doc_result
    return summary
