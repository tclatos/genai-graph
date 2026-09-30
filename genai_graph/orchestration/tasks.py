"""Prefect tasks for orchestrating knowledge graph creation.

These tasks wrap the existing core KG creation primitives into Prefect tasks,
so that we can build observable, resilient flows while preserving the
underlying behavior.

The tasks are designed as DAG building blocks:

- ``resolve_config_task`` — resolve KG profile name and load config
- ``initialize_backend_task`` — create graph backend (Kuzu)
- ``load_factories_task`` — instantiate factory objects from config
- ``create_schema_task`` — create graph schema for all bundles (Pass 1)
- ``create_bundle_schema_task`` — create schema for a single bundle
- ``ingest_bundle_task`` — ingest documents for a single bundle (Pass 2)
- ``import_kg_task`` — import a single KG dependency from parquet
- ``export_*_task`` — export artifacts (schema, HTML, parquet, …)
- ``summarize_warnings_task`` — collect and log warnings
"""

from __future__ import annotations

from datetime import UTC
from pathlib import Path
from typing import Any

from genai_tk.config_mgmt.import_utils import ImportResolver
from loguru import logger
from prefect import get_run_logger, task
from prefect.cache_policies import NO_CACHE
from prefect.exceptions import MissingContextError

from genai_graph.kg.backend import (
    KgBackend,
    create_backend_from_config,
    delete_backend_storage_from_config,
    get_backend_storage_path_from_config,
)
from genai_graph.kg.factories import (
    JsonFileBackedFactory,
    KgFactory,
    Neo4jFactory,
    Neo4jImportFactory,
    TableBackedFactory,
)
from genai_graph.kg.ingest import DocumentStats, add_documents_to_graph, add_neo4j_data_to_graph
from genai_graph.kg.ingest import create_schema as core_create_schema
from genai_graph.kg.manager import get_kg_manager
from genai_graph.orchestration.models import BundleResult, GraphBundle, GraphFilter, ImportResult, WarningsCollector


def _get_prefect_logger_or_default() -> Any:
    """Return a Prefect run logger when in a flow, else fall back to loguru logger."""
    try:
        return get_run_logger()
    except MissingContextError:
        return logger


def _query_existing_pks(backend: KgBackend, gf: GraphFilter) -> set[str]:
    """Return the set of all PK values for *node_label.property* in the graph."""
    query = f"MATCH (n:{gf.node_label}) RETURN n.{gf.property} AS pk"
    try:
        df = backend.execute_get_as_df(query)
        if df.empty or "pk" not in df.columns:
            return set()
        return {str(v) for v in df["pk"].dropna()}
    except Exception as exc:
        logger.warning("filter_by_existing query failed ({}): {}", gf, exc)
        return set()


@task(cache_policy=NO_CACHE)
def resolve_config_task(config_name: str | None) -> tuple[str, dict[str, Any]]:
    """Resolve KG profile and return its configuration dictionary."""
    logger_pf = _get_prefect_logger_or_default()

    manager = get_kg_manager()
    effective = config_name or manager.profile

    if effective not in manager.ekg_config.kg_configs:
        raise ValueError(
            f"KG config '{effective}' not found. Available: {sorted(manager.ekg_config.kg_configs.keys())}"
        )

    kg_cfg = manager.ekg_config.kg_configs[effective].model_dump()

    logger_pf.debug(
        "Loaded KG config '%s', subgraphs=%d.",
        effective,
        len(kg_cfg.get("graphs", [])),
    )

    return effective, kg_cfg


@task(cache_policy=NO_CACHE)
def initialize_backend_task(config_key: str = "default", kg_config_name: str | None = None) -> KgBackend:
    """Create and return the graph backend instance."""
    logger_pf = _get_prefect_logger_or_default()

    backend = create_backend_from_config(config_key, kg_config_name)
    db_path = get_backend_storage_path_from_config(config_key, kg_config_name)

    logger_pf.debug("Initialized backend '%s' at path '%s'", config_key, db_path)
    return backend


@task(cache_policy=NO_CACHE)
def load_factories_task(kg_cfg: dict[str, Any]) -> list[GraphBundle]:
    """Load and instantiate subgraph factories from KG configuration."""
    logger_pf = _get_prefect_logger_or_default()
    manager = get_kg_manager()
    graphs_cfg = kg_cfg.get("graphs", [])

    bundles: list[GraphBundle] = []
    for graph_cfg in graphs_cfg:
        if not isinstance(graph_cfg, dict):
            continue

        factory_path = graph_cfg.get("factory")
        if not factory_path:
            continue

        try:
            imported = ImportResolver.import_from_qualified(factory_path)
            if isinstance(imported, KgFactory):
                graph_impl = imported
            elif isinstance(imported, type) and issubclass(imported, KgFactory):
                constructor_kwargs = {
                    k: v
                    for k, v in graph_cfg.items()
                    if k not in {"factory", "initial_load", "trigger", "filter_by_existing"}
                }
                graph_impl = imported(**constructor_kwargs)  # type: ignore[misc]
            else:
                msg = f"Factory {factory_path} is not a KgFactory"
                logger.warning(msg)
                manager.add_warning(msg)
                continue

            logger_pf.debug("Loaded subgraph factory: %s", graph_impl.name)
            bundles.append(GraphBundle(config=graph_cfg, factory=graph_impl))
        except Exception as exc:  # pragma: no cover - defensive logging
            import traceback

            msg = f"Failed to import factory {factory_path}: {exc}"
            logger.error(msg)
            logger.error(traceback.format_exc())
            manager.add_warning(msg)

    return bundles


# ---------------------------------------------------------------------------
# Schema creation — as a proper task
# ---------------------------------------------------------------------------


@task(cache_policy=NO_CACHE)
def create_schema_task(bundles: list[GraphBundle], backend: KgBackend) -> list[GraphBundle]:
    """Create graph schema for all loaded subgraphs (Pass 1).

    Iterates over bundles in order, registering factories and creating
    their Kuzu schema.  Returns the bundles with ``schema_obj`` populated.
    """
    manager = get_kg_manager()

    for bundle in bundles:
        graph_impl = bundle.factory
        graph_impl.register()

        schema = graph_impl.build_schema()
        try:
            core_create_schema(backend, schema.nodes, schema.relations, manager)
            schema.validate_with_context(manager)
            logger.info("Created schema for subgraph {}", getattr(graph_impl, "name", "<unknown>"))
        except Exception as exc:  # pragma: no cover - defensive
            import traceback

            msg = f"Schema creation failed for subgraph {getattr(graph_impl, 'name', '<unknown>')}: {exc}"
            logger.error(msg)
            logger.error(traceback.format_exc())
            manager.add_warning(msg)

        bundle.schema_obj = schema

    return bundles


# ---------------------------------------------------------------------------
# Ingestion — per-bundle task
# ---------------------------------------------------------------------------


@task(cache_policy=NO_CACHE)
def ingest_bundle_task(bundle: GraphBundle, backend: KgBackend) -> BundleResult:
    """Ingest documents for a single subgraph bundle (Pass 2).

    Replaces the monolithic ``ingest_subgraphs_task`` with per-bundle
    granularity.  Each bundle is now a separate Prefect task visible
    in the DAG.
    """
    logger_pf = _get_prefect_logger_or_default()
    manager = get_kg_manager()
    collector = WarningsCollector(source=getattr(bundle.factory, "name", ""))

    graph_cfg = bundle.config
    graph_impl = bundle.factory
    schema = bundle.schema_obj
    factory_path = graph_cfg.get("factory", "<unknown>")

    result = BundleResult(factory_path=factory_path)

    keys = graph_cfg.get("initial_load", [])

    # --- JsonFileBackedFactory -------------------------------------------------
    if not keys and isinstance(graph_impl, JsonFileBackedFactory):
        try:
            file_paths = graph_impl.get_all_file_paths()
            keys = [str(fp) for fp in file_paths]
            logger_pf.debug("Retrieved %d file paths from JSON-backed factory", len(keys))
        except Exception as exc:  # pragma: no cover - defensive
            msg = f"Failed to get file paths from JSON factory {factory_path}: {exc}"
            logger.warning(msg)
            manager.add_warning(msg)
            collector.add(msg)
            keys = []

    # --- TableBackedFactory ----------------------------------------------------
    if not keys and isinstance(graph_impl, TableBackedFactory):
        try:
            keys = graph_impl.get_all_keys()
            logger_pf.debug("Retrieved %d keys from table-backed factory", len(keys))
            filter_cfg = graph_cfg.get("filter_by_existing")
            if filter_cfg:
                gf = GraphFilter.model_validate(filter_cfg)
                allowed = _query_existing_pks(backend, gf)
                before = len(keys)
                keys = [k for k in keys if k in allowed]
                logger_pf.info(
                    "filter_by_existing (%s.%s): %d/%d keys kept for %s",
                    gf.node_label,
                    gf.property,
                    len(keys),
                    before,
                    factory_path,
                )
        except Exception as exc:  # pragma: no cover - defensive
            msg = f"Failed to get keys from table for {factory_path}: {exc}"
            logger.warning(msg)
            manager.add_warning(msg)
            collector.add(msg)
            keys = []

    # --- Neo4jImportFactory — direct import path --------------------------------
    if isinstance(graph_impl, Neo4jImportFactory):
        try:
            logger_pf.info("Using direct Neo4j import for %s", factory_path)
            stats = add_neo4j_data_to_graph(graph_impl, backend, manager)
            logger_pf.debug(
                "Neo4j import stats for %s: processed=%d failed=%d nodes=%d rels=%d",
                factory_path,
                stats.total_processed,
                stats.total_failed,
                stats.nodes_created,
                stats.relationships_created,
            )
            result.stats = stats
        except Exception as exc:  # pragma: no cover - defensive
            import traceback

            msg = f"Neo4j import error for {factory_path}: {exc}"
            logger.error(msg)
            logger.error(traceback.format_exc())
            manager.add_warning(msg)
            collector.add(msg)
            result.stats.total_failed += 1
        result.warnings = collector
        return result

    # --- Neo4jFactory (legacy JSONL) -------------------------------------------
    if not keys and isinstance(graph_impl, Neo4jFactory):
        try:
            keys = graph_impl.get_all_keys()
            logger_pf.debug("Retrieved %d keys from Neo4j JSONL factory", len(keys))
        except Exception as exc:  # pragma: no cover - defensive
            msg = f"Failed to get keys from Neo4j factory {factory_path}: {exc}"
            logger.warning(msg)
            manager.add_warning(msg)
            collector.add(msg)
            keys = []

    # --- Standard document ingestion path --------------------------------------
    if not keys:
        result.warnings = collector
        return result

    try:
        assert schema is not None, "Schema must be created before ingestion"
        stats = add_documents_to_graph(keys, graph_impl, backend, schema, manager)
        logger_pf.debug(
            "Ingest stats for %s: processed=%d failed=%d nodes=%d rels=%d",
            factory_path,
            stats.total_processed,
            stats.total_failed,
            stats.nodes_created,
            stats.relationships_created,
        )
        result.stats = stats
    except Exception as exc:  # pragma: no cover - defensive
        import traceback

        msg = f"Ingestion error for {factory_path}: {exc}"
        logger.error(msg)
        logger.error(traceback.format_exc())
        manager.add_warning(msg)
        collector.add(msg)
        result.stats.total_failed += len(keys)

    result.warnings = collector
    return result


# ---------------------------------------------------------------------------
# Import task
# ---------------------------------------------------------------------------


@task(cache_policy=NO_CACHE)
def import_kg_task(import_name: str, backend: KgBackend, force_rebuild: bool = False) -> ImportResult:
    """Import a single KG dependency by creating it (if needed) and loading its parquet.

    Before rebuilding, validates the existing parquet cache using schema
    and factory fingerprints.  If valid, the import is skipped (unless
    *force_rebuild* is ``True``).

    Each import is now a distinct Prefect task visible in the DAG,
    replacing the old recursive ``_ensure_kg_exists`` pattern.
    """
    from genai_graph.kg.export.artifacts import (
        import_from_parquet,
        load_parquet_manifest,
        validate_parquet_cache,
    )

    logger_pf = _get_prefect_logger_or_default()
    collector = WarningsCollector(source=import_name)

    # 1. Check whether a valid parquet cache already exists
    manifest = load_parquet_manifest(import_name)

    need_rebuild = True
    if manifest is not None and not force_rebuild:
        cache_valid, reasons = validate_parquet_cache(import_name)
        if cache_valid:
            msg = f"Parquet cache for '{import_name}' is valid (exported at {manifest.exported_at}) — skipping rebuild"
            logger.info(msg)
            logger_pf.info(msg)
            need_rebuild = False
        else:
            msg = f"Parquet cache for '{import_name}' is stale: {'; '.join(reasons)} — rebuilding"
            logger.info(msg)
            logger_pf.info(msg)
    elif manifest is None:
        msg = f"No parquet export for '{import_name}' — building it first…"
        logger.info(msg)
        logger_pf.info(msg)
    else:
        msg = f"force requested for '{import_name}'"
        logger.info(msg)
        logger_pf.info(msg)

    if need_rebuild:
        from genai_tk.workflow.prefect.run import ephemeral_prefect_settings

        from genai_graph.orchestration.flows import create_kg_flow

        with ephemeral_prefect_settings():
            sub_result = create_kg_flow(config_name=import_name, delete_first=True, export_html=False)
        # Capture warnings from the sub-flow
        for w in sub_result.warnings:
            collector.add(w)

    # 2. Create schema from the imported KG's factories
    logger_pf.info("Creating schema from imported KG '%s'", import_name)
    _create_schema_for_import(import_name, backend, logger_pf)

    # 3. Load from parquet
    try:
        nodes, rels = import_from_parquet(import_name, backend)
        logger_pf.info("Imported %d nodes, %d rels from '%s'", nodes, rels, import_name)
        return ImportResult(
            config_name=import_name,
            nodes_imported=nodes,
            rels_imported=rels,
            warnings=collector,
            skipped=not need_rebuild,
        )
    except FileNotFoundError as exc:
        msg = f"Failed to import '{import_name}': {exc}"
        logger_pf.error(msg)
        collector.add(msg)
        raise


# ---------------------------------------------------------------------------
# Export tasks (read-only — safe to run in parallel later)
# ---------------------------------------------------------------------------


@task(cache_policy=NO_CACHE)
def export_schema_task(config_name: str) -> None:
    """Export schema as text + JSON + HTML."""
    from genai_graph.kg.export.artifacts import export_schema, export_schema_html, export_schema_json

    export_schema(config_name)
    export_schema_json(config_name)
    export_schema_html(config_name)


@task(cache_policy=NO_CACHE)
def export_info_task(config_name: str, backend: KgBackend) -> None:
    """Export KG info markdown."""
    from genai_graph.kg.export.artifacts import export_info

    export_info(config_name, backend)


@task(cache_policy=NO_CACHE)
def export_html_task(config_name: str, backend: KgBackend) -> None:
    """Export HTML graph visualization."""
    from genai_graph.kg.export import export_html

    export_html(config_name, backend)


@task(cache_policy=NO_CACHE)
def export_parquet_task(config_name: str, collector: Any) -> None:
    """Save parquet from collected DataFrames, including cache fingerprints."""
    from genai_graph.kg.export.artifacts import compute_fingerprints_for_config, save_parquet_from_collector

    logger_pf = _get_prefect_logger_or_default()
    try:
        fingerprints = compute_fingerprints_for_config(config_name)
        parquet_result = save_parquet_from_collector(config_name, collector, fingerprints=fingerprints)
        logger_pf.info("Saved parquet: %d nodes, %d rels", parquet_result.node_count, parquet_result.rel_count)
        if fingerprints.schema_fingerprint:
            logger_pf.debug(
                "Manifest fingerprints: schema=%s config=%s content=%s",
                fingerprints.schema_fingerprint[:8],
                (fingerprints.factory_config_hash or "")[:8],
                (fingerprints.source_content_hash or "")[:8],
            )
    except Exception as exc:
        logger_pf.warning("Failed to save parquet: %s", exc)


@task(cache_policy=NO_CACHE)
def export_warnings_task(config_name: str, warnings: list[str]) -> None:
    """Export warnings to structured Markdown report."""
    from genai_graph.kg.export.artifacts import export_warnings

    if warnings:
        export_warnings(config_name, warnings)


# ---------------------------------------------------------------------------
# Warning summarisation — as a proper task
# ---------------------------------------------------------------------------


@task
def summarize_warnings_task(config_name: str | None = None) -> list[str]:
    """Return collected warnings from KgManager and log to file."""
    manager = get_kg_manager()
    warnings = manager.get_warnings()

    if warnings:
        logger.warning("KG creation completed with {} warning(s)", len(warnings))
        if config_name:
            manager.activate()
            manager.log_warnings(warnings)
    else:
        logger.info("KG creation completed with no warnings")

    return warnings


@task
def delete_backend_task(config_key: str = "default", kg_config_name: str | None = None) -> None:
    """Delete the knowledge graph backend storage."""
    logger_pf = _get_prefect_logger_or_default()
    path = get_backend_storage_path_from_config(config_key, kg_config_name)

    if path.exists():
        logger_pf.info("Deleting backend storage at '%s' for config '%s'", path, config_key)
        delete_backend_storage_from_config(config_key, kg_config_name)
    else:
        logger_pf.info("No backend storage found at '%s' for config '%s'", path, config_key)


# ---------------------------------------------------------------------------
# Document nodes task — separate task to allow future extensions
# (chunking, summarization, embedding nodes, …)
# ---------------------------------------------------------------------------


@task(cache_policy=NO_CACHE)
def create_document_nodes_task(bundles: list[GraphBundle], backend: KgBackend) -> DocumentStats:
    """Create Document nodes and CONTAINS relationships for file-backed factories.

    Must run *after* ``ingest_bundle_task`` so that root entities already exist
    in the graph before CONTAINS edges are created.  Being a separate Prefect task
    makes it straightforward to add follow-up file-processing steps (chunking,
    summarization, embedding sub-nodes, …) as additional tasks later.

    Args:
        bundles: All GraphBundles from the current flow; non-DocumentMixin bundles
            are silently skipped.
        backend: Active KgBackend.

    Returns:
        Aggregate :class:`DocumentStats` across all processed bundles.
    """
    from genai_graph.kg.factories import DocumentMixin
    from genai_graph.kg.ingest.merge import _format_value_for_cypher  # type: ignore[attr-defined]

    log = _get_prefect_logger_or_default()
    total = DocumentStats()

    for bundle in bundles:
        factory = bundle.factory
        schema = bundle.schema_obj

        if not isinstance(factory, DocumentMixin) or schema is None:
            continue

        root_class = schema.root_model_class
        if root_class is None:
            continue

        # Find the GraphNode for the root model class
        root_graph_node = next((n for n in schema.nodes if n.node_class is root_class), None)
        if root_graph_node is None:
            log.warning("No GraphNode found for root class %s; skipping Document nodes", root_class.__name__)
            continue

        root_type_name = root_class.__name__
        key_from = root_graph_node.key_from
        pk_field = "id" if (key_from == "AUTO_ID" or callable(key_from)) else key_from

        try:
            file_paths = factory.get_all_file_paths()
        except Exception as exc:
            log.warning("Could not retrieve file paths from %s: %s", getattr(factory, "name", "?"), exc)
            continue

        log.info("Creating Document nodes for %d file(s) from %s", len(file_paths), getattr(factory, "name", "?"))

        for raw_path in file_paths:
            fp = Path(raw_path) if not isinstance(raw_path, Path) else raw_path
            file_key = str(fp)

            # ── 1. Build Document node ─────────────────────────────────────
            try:
                doc = factory.create_document_node(fp)
            except Exception as exc:
                log.warning("Failed to build Document node for %s: %s", fp, exc)
                total.total_failed += 1
                continue

            # ── 2. Compute root-entity primary key ─────────────────────────
            try:
                raw_data = factory.get_struct_data_by_key(file_key)
                if raw_data is None:
                    log.debug("No data for key %s; skipping Document", file_key)
                    total.total_failed += 1
                    continue
                data_dict = raw_data.model_dump()
                root_entity_pk = root_graph_node.get_key_value(data_dict, root_type_name)
            except Exception as exc:
                log.warning("Could not derive root-entity PK for %s: %s", fp, exc)
                total.total_failed += 1
                continue

            # ── 3. MERGE Document node into the graph ──────────────────────
            try:
                from datetime import datetime

                now = datetime.now(UTC).isoformat().replace("+00:00", "Z")

                def _v(val: object) -> str:
                    return _format_value_for_cypher(val)

                merge_doc_cypher = (
                    f"MERGE (d:Document {{content_hash: {_v(doc.content_hash)}}})\n"
                    f"ON CREATE SET\n"
                    f"  d.name = {_v(doc.filename)},\n"
                    f"  d._original_name = {_v(doc.filename)},\n"
                    f"  d.filename = {_v(doc.filename)},\n"
                    f"  d.path = {_v(doc.path)},\n"
                    f"  d.file_size = {_v(doc.file_size)},\n"
                    f"  d.mime_type = {_v(doc.mime_type)},\n"
                    f"  d.modified_at = {_v(doc.modified_at)},\n"
                    f"  d.content_hash = {_v(doc.content_hash)},\n"
                    f"  d.access_level = {_v(doc.access_level)},\n"
                    f"  d.allowed_roles = {_v(doc.allowed_roles)},\n"
                    f"  d.allowed_users = {_v(doc.allowed_users)},\n"
                    f"  d._created_at = {_v(now)},\n"
                    f"  d._updated_at = {_v(now)}\n"
                    f"ON MATCH SET\n"
                    f"  d.content_hash = {_v(doc.content_hash)},\n"
                    f"  d._updated_at = {_v(now)}"
                )
                backend.execute(merge_doc_cypher)
            except Exception as exc:
                log.warning("Failed to MERGE Document node for %s: %s", fp, exc)
                total.total_failed += 1
                continue

            # ── 4. Create CONTAINS relationship ────────────────────────────
            try:
                contains_cypher = (
                    f"MATCH (d:Document {{content_hash: {_v(doc.content_hash)}}}),\n"
                    f"      (r:{root_type_name} {{{pk_field}: {_v(root_entity_pk)}}})\n"
                    f"MERGE (d)-[:CONTAINS]->(r)"
                )
                backend.execute(contains_cypher)
            except Exception as exc:
                log.warning("CONTAINS edge failed for %s → %s pk=%s: %s", fp, root_type_name, root_entity_pk, exc)
                total.total_failed += 1
                continue

            total.nodes_created += 1
            total.relationships_created += 1
            total.total_processed += 1

    log.info(
        "Document task complete: created=%d rels=%d failed=%d",
        total.nodes_created,
        total.relationships_created,
        total.total_failed,
    )
    return total


@task(cache_policy=NO_CACHE)
def compute_similarities_task(bundles: list[GraphBundle], backend: KgBackend) -> list:
    """Create similarity-based relationships for all SimilarityFactory bundles.

    Runs after ``create_vector_indexes_task`` so that all HNSW indexes are
    available before any vector queries are issued.

    Args:
        bundles: List of GraphBundles; only those whose factory is a
            :class:`~genai_graph.kg.factories.SimilarityFactory` are processed.
        backend: Active KgBackend (must be KuzuBackend for similarity queries).

    Returns:
        List of :class:`~genai_graph.kg.factories.SimilarityResult` objects,
        one per processed SimilarityFactory bundle.
    """
    from genai_graph.kg.factories.similarity import SimilarityFactory, SimilarityResult

    log = _get_prefect_logger_or_default()
    results: list[SimilarityResult] = []

    for bundle in bundles:
        if not isinstance(bundle.factory, SimilarityFactory):
            continue

        factory_name = getattr(bundle.factory, "name", bundle.config.get("factory", "<unknown>"))
        log.info("Running similarity computation for factory '%s'", factory_name)

        try:
            result = bundle.factory.compute_similarities(backend)
            results.append(result)
            log.info(
                "Similarity factory '%s': evaluated %d source nodes, created %d relationships",
                result.factory_name,
                result.pairs_evaluated,
                result.relationships_created,
            )
        except Exception as exc:  # pragma: no cover - defensive
            import traceback

            log.error(
                "Similarity computation failed for '%s': %s\n%s",
                factory_name,
                exc,
                traceback.format_exc(),
            )

    if not results:
        log.debug("No SimilarityFactory bundles found; skipping similarity computation")

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _create_schema_for_import(import_name: str, backend: Any, log: Any, visited: set | None = None) -> None:
    """Create schema from an imported KG's configuration (recursive for nested imports)."""
    if visited is None:
        visited = set()
    if import_name in visited:
        return
    visited.add(import_name)

    manager = get_kg_manager()

    if import_name not in manager.ekg_config.kg_configs:
        raise ValueError(f"Imported KG config '{import_name}' not found")

    import_cfg = manager.ekg_config.kg_configs[import_name].model_dump()

    # Recursively process nested imports first
    nested_imports = import_cfg.get("imports", []) or import_cfg.get("import", [])
    if nested_imports:
        log.info(f"Processing {len(nested_imports)} nested import(s) for '{import_name}': {nested_imports}")
    for nested_import in nested_imports:
        _create_schema_for_import(nested_import, backend, log, visited)

    # Load factories and create schemas
    bundles = load_factories_task.fn(import_cfg)
    bundles = create_schema_task.fn(bundles, backend)

    log.info(f"Created schema from '{import_name}' with {len(bundles)} subgraph(s)")


@task(cache_policy=NO_CACHE)
def create_vector_indexes_task(bundles: list[GraphBundle], backend: KgBackend) -> None:
    """Create vector indexes for embedding fields in the knowledge graph.

    After all nodes are ingested, create HNSW vector indexes on fields
    marked as embeddings in the node configurations.

    Args:
        bundles: List of GraphBundles with node configurations
        backend: KgBackend instance (Kuzu)
    """
    log = _get_prefect_logger_or_default()

    if not isinstance(backend, type(backend).__bases__[0]) and not hasattr(backend, "create_vector_index"):
        log.warning("Backend does not support vector indexes, skipping")
        return

    from genai_graph.kg.backend import KuzuBackend

    if not isinstance(backend, KuzuBackend):
        log.debug("Skipping vector index creation for non-Kuzu backend")
        return

    index_count = 0
    for bundle in bundles:
        if bundle.schema_obj is None:
            log.debug("Skipping vector index creation; bundle schema is missing")
            continue
        for config in bundle.schema_obj.nodes:
            # Skip if not computing embeddings or no index fields
            if not config.compute_embeddings or not config.index_fields:
                continue

            table_name = config.node_class.__name__
            try:
                table_info = backend.execute(f"CALL table_info('{table_name}') RETURN *;")
                existing_columns = {row[1] for row in table_info}
            except Exception as e:
                log.warning(f"Failed to inspect table {table_name}: {e}")
                continue
            for field_name, _model in config.index_field_specs:
                index_name = f"{field_name}_index"
                embedding_field = f"{field_name}_embedding"
                if embedding_field not in existing_columns:
                    log.debug(f"Skipping vector index {index_name}; column {embedding_field} not found in {table_name}")
                    continue
                try:
                    backend.create_vector_index(table_name, embedding_field, index_name, metric="cosine")
                    index_count += 1
                except Exception as e:
                    log.warning(f"Failed to create vector index {index_name}: {e}")

    if index_count > 0:
        log.info(f"Created {index_count} vector index(es)")


@task
def drop_vector_indexes_task(bundles: list[GraphBundle], backend: KgBackend) -> None:
    """Drop all HNSW vector indexes before re-ingestion.

    Ladybug (Kuzu) forbids updating a vector property in-place when it is
    covered by a HNSW index ("Cannot set property vec in table embeddings …").
    Dropping indexes here before the MERGE pass and recreating them afterwards
    via :func:`create_vector_indexes_task` is the recommended workaround.

    Args:
        bundles: List of GraphBundles with node configurations.
        backend: KgBackend instance.
    """
    log = _get_prefect_logger_or_default()

    from genai_graph.kg.backend import KuzuBackend

    if not isinstance(backend, KuzuBackend):
        log.debug("Skipping vector index drop for non-Kuzu backend")
        return

    drop_count = 0
    for bundle in bundles:
        if bundle.schema_obj is None:
            continue
        for config in bundle.schema_obj.nodes:
            if not config.compute_embeddings or not config.index_fields:
                continue
            table_name = config.node_class.__name__
            for field_name, _model in config.index_field_specs:
                index_name = f"{field_name}_index"
                backend.drop_vector_index(table_name, index_name)
                drop_count += 1

    if drop_count > 0:
        log.info(f"Dropped {drop_count} vector index(es) (will be recreated after ingestion)")


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------


def create_schema(bundles: list[GraphBundle], backend: KgBackend) -> list[GraphBundle]:
    """Plain-function wrapper for ``create_schema_task``."""
    return create_schema_task.fn(bundles, backend)


def summarize_warnings(config_name: str | None = None) -> list[str]:
    """Plain-function wrapper for ``summarize_warnings_task``."""
    return summarize_warnings_task.fn(config_name)


@task(cache_policy=NO_CACHE)
def ingest_subgraphs_task(bundles: list[GraphBundle], backend: KgBackend) -> DocumentStats:
    """Ingest documents for all configured subgraphs — legacy wrapper.

    New code should use ``ingest_bundle_task`` per bundle instead.
    """
    total_stats = DocumentStats()
    for bundle in bundles:
        result = ingest_bundle_task.fn(bundle, backend)
        total_stats.total_processed += result.stats.total_processed
        total_stats.total_failed += result.stats.total_failed
        total_stats.nodes_created += result.stats.nodes_created
        total_stats.relationships_created += result.stats.relationships_created
    return total_stats
