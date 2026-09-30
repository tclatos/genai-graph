"""Export utilities for Knowledge Graph artifacts.

This module provides functions to export various KG artifacts:
- HTML visualization
- Schema documentation (text file)
- KG info report (markdown file)
- Parquet export for KG data (for import mechanism)
"""

from __future__ import annotations

import contextlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from pydantic import BaseModel

from genai_graph.kg.backend import KgBackend
from genai_graph.kg.manager import get_kg_manager


class HtmlExportResult(BaseModel):
    """Result of HTML export task."""

    config_name: str
    output_path: Path

    model_config = {"arbitrary_types_allowed": True}


class ParquetExportResult(BaseModel):
    """Result of parquet export task."""

    config_name: str
    nodes_path: Path
    rels_path: Path
    manifest_path: Path
    node_count: int
    rel_count: int

    model_config = {"arbitrary_types_allowed": True}


class ParquetManifest(BaseModel):
    """Manifest file for parquet export.

    Fingerprint fields enable smart cache invalidation:
    - ``schema_fingerprint`` — captures the graph schema structure
    - ``factory_config_hash`` — captures factory configuration
    - ``source_content_hash`` — captures actual source file contents
    """

    config_name: str
    exported_at: str
    node_tables: list[str]
    rel_tables: list[str]
    node_count: int
    rel_count: int
    source_files: list[str] = []
    source_files_hash: str | None = None

    # Smart cache fingerprints (Phase 2)
    schema_fingerprint: str | None = None
    factory_config_hash: str | None = None
    source_content_hash: str | None = None

    model_config = {"arbitrary_types_allowed": True}


class CacheFingerprints(BaseModel):
    """Current fingerprints computed from live factories/schemas.

    Used to compare against a saved ``ParquetManifest`` to decide
    whether the cached parquet is still valid.
    """

    schema_fingerprint: str | None = None
    factory_config_hash: str | None = None
    source_content_hash: str | None = None

    def matches(self, manifest: ParquetManifest) -> bool:
        """Return True if all non-None fingerprints match the manifest.

        A ``None`` value on either side is treated as "unknown" and
        does not cause a mismatch — this keeps backward compatibility
        with manifests generated before fingerprints were added.
        """
        for field in ("schema_fingerprint", "factory_config_hash", "source_content_hash"):
            current = getattr(self, field)
            cached = getattr(manifest, field)
            if current is not None and cached is not None and current != cached:
                return False
        return True

    def mismatch_reasons(self, manifest: ParquetManifest) -> list[str]:
        """Return human-readable descriptions of fingerprint mismatches."""
        reasons: list[str] = []
        labels = {
            "schema_fingerprint": "schema structure",
            "factory_config_hash": "factory configuration",
            "source_content_hash": "source file contents",
        }
        for field, label in labels.items():
            current = getattr(self, field)
            cached = getattr(manifest, field)
            if current is not None and cached is not None and current != cached:
                reasons.append(f"{label} changed ({cached[:8]}… → {current[:8]}…)")
        return reasons


def export_html(
    config_name: str,
    backend: KgBackend,
    output_dir: Path | None = None,
) -> HtmlExportResult:
    """Export an HTML visualization of the current KG and return its path.

    Args:
        config_name: Name of the KG configuration
        backend: The graph backend to export from
        output_dir: Optional custom output directory (if None, uses KG outcome manager)
    """
    from genai_graph.kg.export.html import generate_html

    if output_dir is None:
        # Use KgManager for organized output with the specified config
        manager = get_kg_manager()
        destination = manager.get_html_path_for(config_name)
        manager.ensure_directories_for(config_name)
    else:
        # Custom output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"{config_name}_graph.html"

    generate_html(backend, destination_file_path=str(destination))
    logger.debug("Exported KG HTML visualization to '{}'", destination)

    return HtmlExportResult(config_name=config_name, output_path=destination)


def _ensure_factories_registered_for(config_name: str) -> None:
    """Load and register all factory instances for a config, resolving imports recursively.

    This replicates what ``create_schema_task`` does during a full build, so that
    standalone commands (e.g. ``cli kg schema --regen``) can populate the registry
    without running a complete ingestion flow.  Factories are instantiated with their
    full YAML kwargs; ``build_schema()`` does not require real data files so this is
    safe even when source files are absent.
    """
    from genai_tk.config_mgmt.import_utils import ImportResolver

    from genai_graph.kg.factories.base import KgFactory
    from genai_graph.kg.schema.registry import GraphRegistry

    manager = get_kg_manager()
    ekg_config = manager.ekg_config
    registry = GraphRegistry.get_instance()

    visited: set[str] = set()

    def _load_config(name: str) -> None:
        if name in visited:
            return
        visited.add(name)
        if name not in ekg_config.kg_configs:
            logger.warning("KG config '{}' not found, skipping import", name)
            return
        profile = ekg_config.kg_configs[name]
        # Recurse into imports first (depth-first)
        for imp in profile.imports:
            _load_config(imp)
        # Then register this profile's own factories
        for graph_cfg in profile.graphs:
            factory_path = graph_cfg.factory
            try:
                imported = ImportResolver.import_from_qualified(factory_path)
                constructor_kwargs = {
                    k: v
                    for k, v in graph_cfg.model_dump(exclude_none=True).items()
                    if k not in {"factory", "initial_load"}
                }
                if isinstance(imported, KgFactory):
                    graph_impl = imported
                elif isinstance(imported, type) and issubclass(imported, KgFactory):
                    graph_impl = imported(**constructor_kwargs)
                else:
                    continue
                graph_impl.register(registry)
            except Exception as exc:
                logger.warning("Failed to load factory {} for schema regen: {}", factory_path, exc)

    _load_config(config_name)


def export_schema(config_name: str) -> Path:
    """Export the KG schema as a text file.

    Args:
        config_name: Name of the KG configuration

    Returns:
        Path to the exported schema file
    """
    manager = get_kg_manager()
    manager.ensure_directories_for(config_name)
    _ensure_factories_registered_for(config_name)

    from genai_graph.kg.schema.resolved import ResolvedSchema

    resolved = ResolvedSchema.from_registry()
    destination = manager.get_schema_path_for(config_name)
    destination.write_text(resolved.to_markdown(), encoding="utf-8")
    logger.debug("Exported KG schema to '{}'", destination)
    return destination


def export_schema_json(config_name: str) -> Path:
    """Export the KG schema as a canonical JSON file (nodes, links, vector_indexes).

    Args:
        config_name: Name of the KG configuration.

    Returns:
        Path to the exported schema JSON file.
    """

    manager = get_kg_manager()
    manager.ensure_directories_for(config_name)
    _ensure_factories_registered_for(config_name)

    from genai_graph.kg.schema.resolved import ResolvedSchema

    resolved = ResolvedSchema.from_registry()
    destination = manager.get_schema_json_path_for(config_name)
    destination.write_text(resolved.to_json_str(), encoding="utf-8")
    logger.debug("Exported KG schema JSON to '{}'", destination)
    return destination


def export_schema_html(config_name: str) -> Path:
    """Export the KG schema visualization as an HTML file.

    Args:
        config_name: Name of the KG configuration.

    Returns:
        Path to the exported schema HTML file.
    """
    manager = get_kg_manager()
    manager.ensure_directories_for(config_name)

    # Reuse the JSON if it already exists; otherwise generate from registry.
    schema_json_path = manager.get_schema_json_path_for(config_name)
    if schema_json_path.exists():
        from genai_graph.kg.schema.resolved import ResolvedSchema

        resolved = ResolvedSchema.from_json_file(str(schema_json_path))
        schema_data = resolved.to_d3_json()
    else:
        import json as _json

        schema_data = _json.loads(export_schema_json(config_name).read_text(encoding="utf-8"))

    destination = manager.get_schema_html_path_for(config_name)
    from genai_graph.kg.schema.schema_html import generate_schema_html

    generate_schema_html(schema_data, destination_file_path=str(destination))
    logger.debug("Exported KG schema HTML visualization to '{}'", destination)
    return destination


def export_info(config_name: str, backend: KgBackend) -> Path:
    """Export the KG info as a markdown file.

    Args:
        config_name: Name of the KG configuration
        backend: The graph backend to query from

    Returns:
        Path to the exported info file
    """
    import warnings

    from genai_graph.kg.backend import get_backend_storage_path_from_config
    from genai_graph.kg.schema import GraphRegistry, find_embedded_field_for_class, get_graph

    manager = get_kg_manager()
    manager.ensure_directories_for(config_name)

    # Build registry and get all graphs
    registry = GraphRegistry.get_instance()
    selected_graphs = registry.list_graphs()

    # Suppress validation warnings for combined schemas (type mismatches between
    # extended and base types are expected when merging different graphs)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="Graph schema validation.")
            schema = registry.build_combined_schema(selected_graphs)
    except ValueError as exc:
        logger.error(f"Failed to build schema: {exc}")
        schema = None

    graph_title = ", ".join(selected_graphs) if selected_graphs else "ALL"

    # Start building markdown content
    lines: list[str] = []
    lines.append(f"# {graph_title} EKG Database Information")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Database Information section
    lines.append("## Database Information")
    lines.append("")
    db_path = get_backend_storage_path_from_config("default", config_name)
    active_cfg = manager.profile
    default_kg = manager.ekg_config.kg_config

    lines.append(f"- **Database Path**: `{db_path}`")
    lines.append("- **Database Type**: Cypher Graph Database")
    lines.append("- **Backend**: Cypher (via KgBackend abstraction)")
    lines.append("- **Storage**: Persistent File Storage")
    lines.append(f"- **Active KG Config**: `{active_cfg}@{manager.tag}`")
    lines.append(f"- **Default KG Config**: `{default_kg}`")
    lines.append(f"- **Graph(s)**: **{graph_title}**")
    lines.append("")

    # KG Outputs & Outcomes section
    outcome_info = manager.get_info()
    if outcome_info.get("exists"):
        lines.append("## KG Outputs & Outcomes")
        lines.append("")
        lines.append(f"- **Base Path**: `{outcome_info['base_path']}`")

        if outcome_info.get("database"):
            db_info = outcome_info["database"]
            lines.append(f"- **Database Size**: **{db_info['size_mb']:.2f} MB**")

        if outcome_info.get("html_export"):
            html_info = outcome_info["html_export"]
            lines.append(f"- **HTML Export**: **{html_info['size_mb']:.2f} MB**")

        if outcome_info.get("outcomes"):
            out_info = outcome_info["outcomes"]
            lines.append(f"- **Logged Outcomes**: **{out_info['count']}** events")

        if outcome_info.get("warnings"):
            warn_info = outcome_info["warnings"]
            lines.append(f"- **Logged Warnings**: **{warn_info['count']}** warnings")

        if outcome_info.get("warnings_report"):
            report_info = outcome_info["warnings_report"]
            report_path = Path(report_info["file"])
            report_name = report_path.name
            lines.append(f"- **Warnings Report**: [📊 {report_name}]({report_name})")

        lines.append("")
        lines.append("---")
        lines.append("")

    # Graph Factories section
    lines.append("## Graph Factories")
    lines.append("")
    lines.append("| Name | Type | Module |")
    lines.append("|------|------|--------|")
    for name in selected_graphs or registry.list_graphs():
        try:
            graph_impl = get_graph(name)
            factory_type = type(graph_impl).__name__
            factory_module = type(graph_impl).__module__
            lines.append(f"| **{name}** | `{factory_type}` | `{factory_module}` |")
        except ValueError:
            lines.append(f"| **{name}** | *Not Found* | - |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Schema Overview and Statistics
    if schema:
        try:
            tables_df = backend.execute_get_as_df("CALL show_tables() RETURN *", union=False)

            node_tables: list[str] = []
            rel_tables: list[str] = []

            for _, row in tables_df.iterrows():
                if row.get("type") == "NODE":
                    node_tables.append(row["name"])
                elif row.get("type") == "REL":
                    rel_tables.append(row["name"])

            allowed_node_labels = {n.node_class.__name__ for n in schema.nodes}
            allowed_rel_types = {r.name for r in schema.relations}

            node_tables = [t for t in node_tables if t in allowed_node_labels]
            rel_tables = [t for t in rel_tables if t in allowed_rel_types]

            # Schema Overview
            lines.append("## Schema Overview")
            lines.append("")
            lines.append(f"- **Node Tables**: **{len(node_tables)}**")
            lines.append(f"- **Relationship Tables**: **{len(rel_tables)}**")
            lines.append("")
            lines.append("---")
            lines.append("")

            # Node Counts
            if node_tables:
                lines.append("### Node Counts")
                lines.append("")
                lines.append("| Node Type | Count |")
                lines.append("|-----------|------:|")
                for node_type in sorted(node_tables):
                    try:
                        result_df = backend.execute_get_as_df(
                            f"MATCH (n:{node_type}) RETURN count(n) as count", union=False
                        )
                        count = result_df.iloc[0]["count"]
                        lines.append(f"| **{node_type}** | **{count:,}** |")
                    except Exception as exc:
                        lines.append(f"| **{node_type}** | Error: {exc} |")
                lines.append("")

            # Relationship Counts
            if rel_tables:
                lines.append("### Relationship Counts")
                lines.append("")
                lines.append("| Relationship Type | Count |")
                lines.append("|-------------------|------:|")
                for rel_type in sorted(rel_tables):
                    try:
                        result_df = backend.execute_get_as_df(
                            f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count", union=False
                        )
                        count = result_df.iloc[0]["count"]
                        lines.append(f"| **{rel_type}** | **{count:,}** |")
                    except Exception as exc:
                        lines.append(f"| **{rel_type}** | Error: {exc} |")
                lines.append("")
                lines.append("---")
                lines.append("")

        except Exception as exc:
            lines.append(f"*Error retrieving schema statistics: {exc}*")
            lines.append("")

        # Node Mapping
        lines.append(f"## Node Mapping for {graph_title}")
        lines.append("")
        lines.append("| Node Type | Description | Primary Key |")
        lines.append("|-----------|-------------|-------------|")
        for node in schema.nodes:
            node_type = node.node_class.__name__
            description = node.description or ""

            if node.key_from == "AUTO_ID":
                key_label = "`id` (SERIAL auto-generated)"
            elif isinstance(node.key_from, str):
                key_label = f"`{node.key_from}`"
            else:
                key_label = "`id` (computed)"

            lines.append(f"| **{node_type}** | {description} | {key_label} |")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Relationship Mapping
        lines.append("## Relationship Mapping")
        lines.append("")
        lines.append("| Relationship | From -> To | Meaning | Field Paths |")
        lines.append("|--------------|-----------|---------|-------------|")
        for relation in schema.relations:
            rel_type = relation.name
            direction = f"**{relation.from_node.label}** -> **{relation.to_node.label}**"
            meaning = relation.description or ""

            if relation.field_paths:
                paths_display = "; ".join(f"`{fp or '(root)'} -> {tp or '(root)'}`" for fp, tp in relation.field_paths)
            else:
                paths_display = "*(none)*"

            # Escape pipe characters in the content
            meaning = meaning.replace("|", "\\|")
            paths_display = paths_display.replace("|", "\\|")

            lines.append(f"| **{rel_type}** | {direction} | {meaning} | {paths_display} |")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Embedded Fields
        lines.append("## Embedded Fields")
        lines.append("")
        has_embedded = False
        embedded_rows = []
        for node in schema.nodes:
            for embedded_class in getattr(node, "embedded_struct_classes", []) or []:
                field_name = find_embedded_field_for_class(node.node_class, embedded_class)
                if field_name:
                    has_embedded = True
                    embedded_rows.append(
                        f"| **{node.node_class.__name__}** | `{field_name}` | `{embedded_class.__name__}` |"
                    )

        if has_embedded:
            lines.append("| Parent Node | Embedded Field | Embedded Class |")
            lines.append("|-------------|----------------|----------------|")
            lines.extend(embedded_rows)
        else:
            lines.append("*No embedded fields configured*")
        lines.append("")

    # Write info to file using the specified config
    destination = manager.get_info_path_for(config_name)
    destination.write_text("\n".join(lines), encoding="utf-8")

    logger.debug("Exported KG info to '{}'", destination)

    return destination


def get_parquet_export_dir(config_name: str) -> Path:
    """Get the directory path for parquet exports of a KG config.

    Args:
        config_name: Name of the KG configuration

    Returns:
        Path to the parquet export directory
    """
    manager = get_kg_manager()
    return manager.get_base_path_for(config_name) / "parquet"


def export_parquet(
    config_name: str,
    backend: KgBackend,
    source_files: list[str] | None = None,
    fingerprints: CacheFingerprints | None = None,
) -> ParquetExportResult:
    """Export all nodes and relationships from a KG to parquet files.

    This creates:
    - nodes/<NodeType>.parquet for each node table
    - rels/<RelType>.parquet for each relationship table
    - manifest.json with metadata

    Args:
        config_name: Name of the KG configuration
        backend: The graph backend to export from
        source_files: Optional list of source files (for hash-based caching)
        fingerprints: Optional cache fingerprints to store in the manifest

    Returns:
        ParquetExportResult with export details
    """
    from genai_tk.utils.hashing import buffer_digest

    manager = get_kg_manager()
    manager.ensure_directories_for(config_name)

    export_dir = get_parquet_export_dir(config_name)
    nodes_dir = export_dir / "nodes"
    rels_dir = export_dir / "rels"

    # Create directories
    nodes_dir.mkdir(parents=True, exist_ok=True)
    rels_dir.mkdir(parents=True, exist_ok=True)

    # Get all tables
    tables_df = backend.execute_get_as_df("CALL show_tables() RETURN *", union=False)

    node_tables: list[str] = []
    rel_tables: list[str] = []

    for _, row in tables_df.iterrows():
        if row.get("type") == "NODE":
            node_tables.append(row["name"])
        elif row.get("type") == "REL":
            rel_tables.append(row["name"])

    total_nodes = 0
    total_rels = 0

    # Export node tables
    for node_type in node_tables:
        try:
            df = backend.execute_get_as_df(f"MATCH (n:{node_type}) RETURN n.*", union=False)
            if not df.empty:
                # Clean column names (remove 'n.' prefix)
                df.columns = [c.replace("n.", "") for c in df.columns]
                parquet_path = nodes_dir / f"{node_type}.parquet"
                df.to_parquet(str(parquet_path), index=False)
                total_nodes += len(df)
                logger.debug(f"Exported {len(df)} {node_type} nodes to parquet")
        except Exception as exc:
            logger.warning(f"Failed to export {node_type} nodes: {exc}")

    # Export relationship tables
    for rel_type in rel_tables:
        try:
            # Get relationships with source and target IDs
            df = backend.execute_get_as_df(
                f"MATCH (a)-[r:{rel_type}]->(b) RETURN id(a) as _from_id, id(b) as _to_id, r.*",
                union=False,
            )
            if not df.empty:
                # Clean column names (remove 'r.' prefix)
                df.columns = [c.replace("r.", "") if c.startswith("r.") else c for c in df.columns]
                parquet_path = rels_dir / f"{rel_type}.parquet"
                df.to_parquet(str(parquet_path), index=False)
                total_rels += len(df)
                logger.debug(f"Exported {len(df)} {rel_type} relationships to parquet")
        except Exception as exc:
            logger.warning(f"Failed to export {rel_type} relationships: {exc}")

    # Calculate source files hash if provided
    source_hash = None
    if source_files:
        combined = "|".join(sorted(source_files))
        source_hash = buffer_digest(combined.encode(), algorithm="sha256")[:16]

    # Write manifest
    manifest = ParquetManifest(
        config_name=config_name,
        exported_at=datetime.now().isoformat(),
        node_tables=node_tables,
        rel_tables=rel_tables,
        node_count=total_nodes,
        rel_count=total_rels,
        source_files=source_files or [],
        source_files_hash=source_hash,
        schema_fingerprint=fingerprints.schema_fingerprint if fingerprints else None,
        factory_config_hash=fingerprints.factory_config_hash if fingerprints else None,
        source_content_hash=fingerprints.source_content_hash if fingerprints else None,
    )

    manifest_path = export_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    logger.info(f"Exported KG '{config_name}' to parquet: {total_nodes} nodes, {total_rels} rels")

    return ParquetExportResult(
        config_name=config_name,
        nodes_path=nodes_dir,
        rels_path=rels_dir,
        manifest_path=manifest_path,
        node_count=total_nodes,
        rel_count=total_rels,
    )


def save_parquet_from_collector(
    config_name: str,
    collector: Any,
    source_files: list[str] | None = None,
    fingerprints: CacheFingerprints | None = None,
) -> ParquetExportResult:
    """Save collected DataFrames from merge operations to parquet files.

    This is the preferred method for parquet export as it captures the
    exact data being merged, avoiding the need to query it back out
    (which can hit Kuzu bugs with relationships).

    Args:
        config_name: Name of the KG configuration
        collector: ParquetCollector with accumulated node/relationship DataFrames
        source_files: Optional list of source files (for hash-based caching)
        fingerprints: Optional cache fingerprints to store in the manifest

    Returns:
        ParquetExportResult with export details
    """
    from genai_tk.utils.hashing import buffer_digest

    from genai_graph.kg.ingest.merge import ParquetCollector

    # Validate collector type
    if not isinstance(collector, ParquetCollector):
        raise TypeError(f"Expected ParquetCollector, got {type(collector)}")

    manager = get_kg_manager()
    manager.ensure_directories_for(config_name)

    export_dir = get_parquet_export_dir(config_name)
    nodes_dir = export_dir / "nodes"
    rels_dir = export_dir / "rels"

    # Create directories
    nodes_dir.mkdir(parents=True, exist_ok=True)
    rels_dir.mkdir(parents=True, exist_ok=True)

    total_nodes = 0
    total_rels = 0
    node_tables: list[str] = []
    rel_tables: list[str] = []

    # Export node tables
    for node_type, table in collector.nodes.items():
        if table.num_rows == 0:
            continue
        try:
            parquet_path = nodes_dir / f"{node_type}.parquet"
            # Struct field order is already baked into the Arrow schema,
            # so pq.write_table preserves it correctly.
            pq.write_table(table, str(parquet_path))
            total_nodes += table.num_rows
            node_tables.append(node_type)
            logger.debug(f"Saved {table.num_rows} {node_type} nodes to parquet from collector")
        except Exception as exc:
            logger.warning(f"Failed to save {node_type} nodes to parquet: {exc}")

    # Export relationship tables
    for rel_type, table in collector.relationships.items():
        if table.num_rows == 0:
            continue
        try:
            parquet_path = rels_dir / f"{rel_type}.parquet"
            pq.write_table(table, str(parquet_path))
            total_rels += table.num_rows
            rel_tables.append(rel_type)
            logger.debug(f"Saved {table.num_rows} {rel_type} relationships to parquet from collector")
        except Exception as exc:
            logger.warning(f"Failed to save {rel_type} relationships to parquet: {exc}")

    # Calculate source files hash if provided
    source_hash = None
    if source_files:
        combined = "|".join(sorted(source_files))
        source_hash = buffer_digest(combined.encode(), algorithm="sha256")[:16]

    # Write manifest (with optional fingerprints for cache validation)
    manifest = ParquetManifest(
        config_name=config_name,
        exported_at=datetime.now().isoformat(),
        node_tables=node_tables,
        rel_tables=rel_tables,
        node_count=total_nodes,
        rel_count=total_rels,
        source_files=source_files or [],
        source_files_hash=source_hash,
        schema_fingerprint=fingerprints.schema_fingerprint if fingerprints else None,
        factory_config_hash=fingerprints.factory_config_hash if fingerprints else None,
        source_content_hash=fingerprints.source_content_hash if fingerprints else None,
    )

    manifest_path = export_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    logger.info(f"Saved KG '{config_name}' to parquet from collector: {total_nodes} nodes, {total_rels} rels")

    return ParquetExportResult(
        config_name=config_name,
        nodes_path=nodes_dir,
        rels_path=rels_dir,
        manifest_path=manifest_path,
        node_count=total_nodes,
        rel_count=total_rels,
    )


def load_parquet_manifest(config_name: str) -> ParquetManifest | None:
    """Load the parquet manifest for a KG config if it exists.

    Args:
        config_name: Name of the KG configuration

    Returns:
        ParquetManifest or None if not found
    """
    export_dir = get_parquet_export_dir(config_name)
    manifest_path = export_dir / "manifest.json"

    if not manifest_path.exists():
        return None

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return ParquetManifest(**data)
    except Exception as exc:
        logger.warning(f"Failed to load parquet manifest for {config_name}: {exc}")
        return None


def compute_fingerprints_for_config(config_name: str) -> CacheFingerprints:
    """Compute current cache fingerprints for a KG configuration.

    Loads the factories and schemas for *config_name*, hashes them,
    and also hashes the discovered source file contents.

    Returns:
        ``CacheFingerprints`` with all computable fields populated.
    """
    from genai_tk.config_mgmt.import_utils import ImportResolver
    from genai_tk.utils.hashing import buffer_digest, file_digest

    manager = get_kg_manager()

    if config_name not in manager.ekg_config.kg_configs:
        return CacheFingerprints()

    kg_cfg = manager.ekg_config.kg_configs[config_name].model_dump()
    graphs_cfg = kg_cfg.get("graphs", [])

    schema_parts: list[str] = []
    config_parts: list[str] = []
    content_parts: list[str] = []

    for graph_cfg in graphs_cfg:
        if not isinstance(graph_cfg, dict):
            continue
        factory_path = graph_cfg.get("factory")
        if not factory_path:
            continue

        try:
            from genai_graph.kg.factories import KgFactory

            imported = ImportResolver.import_from_qualified(factory_path)
            if isinstance(imported, type) and issubclass(imported, KgFactory):
                constructor_kwargs = {
                    k: v for k, v in graph_cfg.items() if k not in {"factory", "initial_load", "trigger"}
                }
                factory = imported(**constructor_kwargs)
            elif isinstance(imported, KgFactory):
                factory = imported
            else:
                continue

            # Schema fingerprint
            schema = factory.build_schema()
            schema_parts.append(schema.fingerprint())

            # Config fingerprint
            config_parts.append(factory.config_fingerprint())

            # Source content hash (file-based factories)
            from genai_graph.kg.factories import JsonFileBackedFactory, Neo4jFactory

            if isinstance(factory, JsonFileBackedFactory):
                for fp in factory.get_all_file_paths():
                    with contextlib.suppress(Exception):
                        content_parts.append(file_digest(fp))
            elif isinstance(factory, Neo4jFactory):
                for key in factory.get_all_keys():
                    content_parts.append(buffer_digest(key.encode()))

        except Exception as exc:
            logger.debug("Could not compute fingerprint for {}: {}", factory_path, exc)

    # Combine per-factory hashes into single values
    result = CacheFingerprints()
    if schema_parts:
        result.schema_fingerprint = buffer_digest("|".join(schema_parts).encode())
    if config_parts:
        result.factory_config_hash = buffer_digest("|".join(config_parts).encode())
    if content_parts:
        result.source_content_hash = buffer_digest("|".join(sorted(content_parts)).encode())

    return result


def validate_parquet_cache(config_name: str) -> tuple[bool, list[str]]:
    """Check whether the parquet cache for *config_name* is still valid.

    Loads the manifest, recomputes fingerprints, and compares.

    Returns:
        ``(is_valid, reasons)`` — ``is_valid`` is True when the cache
        can be reused.  ``reasons`` lists human-readable mismatch
        descriptions (empty when valid or when no manifest exists).
    """
    manifest = load_parquet_manifest(config_name)
    if manifest is None:
        return False, ["no parquet cache exists"]

    # If manifest has no fingerprints (legacy), treat as valid
    if manifest.schema_fingerprint is None and manifest.factory_config_hash is None:
        return True, []

    current = compute_fingerprints_for_config(config_name)
    if current.matches(manifest):
        return True, []

    return False, current.mismatch_reasons(manifest)


def _parse_struct_field_types(type_str: str) -> dict[str, str]:
    """Parse field names and types from a Kuzu STRUCT type string, preserving order.

    Example:
        >>> _parse_struct_field_types("STRUCT(objectives STRING[], scope STRING)")
        {'objectives': 'STRING[]', 'scope': 'STRING'}

    Returns:
        Ordered dict of field_name -> kuzu_type
    """
    if not type_str.startswith("STRUCT(") or not type_str.endswith(")"):
        return {}

    inner = type_str[len("STRUCT(") : -1]

    fields: dict[str, str] = {}
    depth = 0
    current = ""
    for ch in inner:
        if ch in "([":
            depth += 1
            current += ch
        elif ch in ")]":
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            field_def = current.strip()
            if field_def:
                parts = field_def.split(None, 1)
                fields[parts[0]] = parts[1] if len(parts) > 1 else "STRING"
            current = ""
        else:
            current += ch

    field_def = current.strip()
    if field_def:
        parts = field_def.split(None, 1)
        fields[parts[0]] = parts[1] if len(parts) > 1 else "STRING"

    return fields


def _parse_struct_field_order(type_str: str) -> list[str]:
    """Parse field names from a Kuzu STRUCT type string, preserving order.

    Example:
        >>> _parse_struct_field_order("STRUCT(objectives STRING[], scope STRING, requirements STRING[])")
        ['objectives', 'scope', 'requirements']

    Args:
        type_str: Kuzu type string like "STRUCT(field1 TYPE1, field2 TYPE2, ...)"

    Returns:
        Ordered list of field names
    """
    return list(_parse_struct_field_types(type_str).keys())


def _pandas_to_arrow_with_structs(
    df: pd.DataFrame,
    struct_field_types: dict[str, dict[str, str]] | None = None,
) -> pa.Table:
    """Convert a pandas DataFrame to a PyArrow Table with struct type enforcement.

    For struct columns, an explicit Arrow schema is built so that field order
    matches the Ladybug schema definition (Ladybug rejects mismatched order).
    This is used only in the parquet import path where data arrives as pandas.
    """
    from genai_graph.kg.ingest.arrow_utils import ladybug_type_to_arrow

    if not struct_field_types:
        return pa.Table.from_pandas(df)

    schema_overrides: dict[str, pa.DataType] = {}
    for col_name, sub_fields in struct_field_types.items():
        if col_name in df.columns:
            arrow_fields = [(fname, ladybug_type_to_arrow(ftype)) for fname, ftype in sub_fields.items()]
            schema_overrides[col_name] = pa.struct(arrow_fields)

    if not schema_overrides:
        return pa.Table.from_pandas(df)

    arrays = []
    names = []
    for col in df.columns:
        if col in schema_overrides:
            arrays.append(pa.array(df[col].tolist(), type=schema_overrides[col]))
        else:
            arrays.append(pa.Array.from_pandas(df[col]))
        names.append(col)

    return pa.table(dict(zip(names, arrays, strict=False)))


def _reorder_struct_dict(d: dict[str, Any], expected_fields: list[str]) -> dict[str, Any]:
    """Reorder dict keys to match expected struct field order.

    Keys in `expected_fields` come first (in that order), followed by any
    extra keys not in the expected list (preserving their original order).

    Args:
        d: Dictionary with potentially wrong key order
        expected_fields: Expected field order from Kuzu schema

    Returns:
        New dict with keys in correct order
    """
    result: dict[str, Any] = {}
    for field in expected_fields:
        if field in d:
            result[field] = d[field]
    # Add any remaining keys not in expected_fields
    for key in d:
        if key not in result:
            result[key] = d[key]
    return result


def import_from_parquet(
    config_name: str,
    backend: KgBackend,
) -> tuple[int, int]:
    """Import nodes and relationships from parquet files into the graph.

    Uses LOAD FROM df MERGE for efficient batch loading.

    Args:
        config_name: Name of the KG configuration to import from
        backend: The graph backend to import into

    Returns:
        Tuple of (nodes_imported, rels_imported)
    """
    from genai_graph.kg.ingest.merge import get_parquet_collector

    export_dir = get_parquet_export_dir(config_name)
    nodes_dir = export_dir / "nodes"
    rels_dir = export_dir / "rels"

    manifest = load_parquet_manifest(config_name)
    if manifest is None:
        raise FileNotFoundError(f"No parquet export found for KG '{config_name}'")

    nodes_imported = 0
    rels_imported = 0

    # Get the active parquet collector (if any) to also collect imported data
    collector = get_parquet_collector()

    # Get kuzu connection
    kuzu_conn = backend.conn if hasattr(backend, "conn") else backend

    # Get primary key info for each table by querying the database schema
    pk_info: dict[str, str] = {}
    try:
        # Query the schema to get primary key columns
        tables_result = kuzu_conn.execute("CALL show_tables() RETURN *")
        tables_df = tables_result.get_as_df() if hasattr(tables_result, "get_as_df") else pd.DataFrame()

        for _, row in tables_df.iterrows():
            if row.get("type") == "NODE":
                table_name = row["name"]
                try:
                    # Get table info including primary key
                    info_result = kuzu_conn.execute(f"CALL table_info('{table_name}') RETURN *")
                    info_df = info_result.get_as_df() if hasattr(info_result, "get_as_df") else pd.DataFrame()
                    # Find the primary key column - column is named "primary key" with a space
                    pk_col_name = "primary key" if "primary key" in info_df.columns else "is_primary_key"
                    if pk_col_name in info_df.columns:
                        pk_rows = info_df[info_df[pk_col_name]]
                        if not pk_rows.empty:
                            pk_info[table_name] = pk_rows.iloc[0]["name"]
                except Exception as exc:
                    logger.debug(f"Could not get table info for {table_name}: {exc}")
    except Exception as exc:
        logger.debug(f"Could not query schema for primary keys: {exc}")

    # Import nodes
    for node_type in manifest.node_tables:
        parquet_path = nodes_dir / f"{node_type}.parquet"
        if not parquet_path.exists():
            logger.warning(f"Node parquet file not found: {parquet_path}")
            continue

        try:
            # Read via pyarrow and strip stored pandas metadata before converting to
            # pandas.  Without this, parquet files that contain Arrow-backed list<float64>
            # columns (embedding vectors) crash on read with:
            #   TypeError: data type 'list<item: double>[pyarrow]' not understood
            # Stripping the metadata makes pyarrow return plain object-dtype columns
            # (numpy arrays) which we can later re-wrap for Kuzu.
            raw_table = pq.read_table(str(parquet_path))
            df = raw_table.replace_schema_metadata({}).to_pandas()
            if df.empty:
                continue

            # Check for missing columns in the table schema and add them if needed
            # This handles schema evolution when importing from different sources
            struct_col_fields: dict[str, list[str]] = {}  # col_name -> expected field order
            struct_col_types: dict[str, dict[str, str]] = {}  # col_name -> {sub_field: kuzu_type}
            try:
                info_result = kuzu_conn.execute(f"CALL table_info('{node_type}') RETURN *")
                info_df = info_result.get_as_df() if hasattr(info_result, "get_as_df") else pd.DataFrame()
                existing_cols = set(info_df["name"].tolist()) if "name" in info_df.columns else set()

                # Parse struct column field order from Kuzu schema types
                if "type" in info_df.columns and "name" in info_df.columns:
                    for _, row in info_df.iterrows():
                        col_type = str(row["type"])
                        if col_type.startswith("STRUCT("):
                            expected_fields = _parse_struct_field_order(col_type)
                            if expected_fields:
                                struct_col_fields[row["name"]] = expected_fields
                            parsed_types = _parse_struct_field_types(col_type)
                            if parsed_types:
                                struct_col_types[row["name"]] = parsed_types

                for col in df.columns:
                    if col not in existing_cols:
                        # Infer Kuzu type from pandas dtype
                        dtype = df[col].dtype
                        if dtype == "object":
                            kuzu_type = "STRING"
                        elif dtype == "int64":
                            kuzu_type = "INT64"
                        elif dtype == "float64":
                            kuzu_type = "DOUBLE"
                        elif dtype == "bool":
                            kuzu_type = "BOOL"
                        else:
                            kuzu_type = "STRING"  # fallback

                        try:
                            kuzu_conn.execute(f"ALTER TABLE {node_type} ADD {col} {kuzu_type}")
                            logger.debug(f"Added column {col} ({kuzu_type}) to {node_type}")
                        except Exception as alter_exc:
                            logger.debug(f"Could not add column {col} to {node_type}: {alter_exc}")
            except Exception as schema_exc:
                logger.debug(f"Could not check schema for {node_type}: {schema_exc}")

            # Convert numpy arrays to Python lists/values for Kuzu compatibility
            import numpy as np

            def convert_numpy_recursive(obj: object) -> object:
                """Recursively convert numpy types to Python native types."""
                if isinstance(obj, np.ndarray):
                    return [convert_numpy_recursive(item) for item in obj.tolist()]
                elif isinstance(obj, dict):
                    return {k: convert_numpy_recursive(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_numpy_recursive(item) for item in obj]
                elif isinstance(obj, (np.integer,)):
                    return int(obj)
                elif isinstance(obj, (np.floating,)):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                return obj

            for col in df.columns:
                if df[col].dtype == "object" and len(df) > 0:
                    # Check first non-null value to see if conversion is needed
                    sample_idx = df[col].first_valid_index()
                    if sample_idx is not None:
                        sample = df[col].loc[sample_idx]
                        if isinstance(sample, (np.ndarray, dict, list)):
                            # Convert numpy types to Python native types
                            df[col] = df[col].apply(lambda x: convert_numpy_recursive(x) if x is not None else None)

            # Reorder struct column dict keys to match Kuzu's expected field order.
            # PyArrow may alphabetize struct fields when writing parquet, but Kuzu
            # requires them in the exact schema order. See docs/cache_invalidation_strategy.md
            for col, expected_fields in struct_col_fields.items():
                if col in df.columns and df[col].dtype == "object" and len(df) > 0:
                    _fields = expected_fields  # bind loop variable for closure
                    df[col] = df[col].apply(
                        lambda x, ef=_fields: _reorder_struct_dict(x, ef) if isinstance(x, dict) else x
                    )
                    logger.debug(f"Reordered struct fields for {node_type}.{col}: {expected_fields}")

            # Re-wrap embedding columns (name ends with _embedding) as Arrow-backed
            # list<float64> so Kuzu's LOAD FROM df scanner maps them to FLOAT[N] columns.
            # After the pyarrow-stripped read above the values are numpy.ndarray objects;
            # plain object-dtype lists would be inferred as STRING by Kuzu.
            try:
                import numpy as np

                for col in list(df.columns):
                    if not col.endswith("_embedding"):
                        continue
                    if df[col].dtype != object or df[col].empty:
                        continue
                    sample_idx = df[col].first_valid_index()
                    if sample_idx is None:
                        continue
                    sample = df[col].loc[sample_idx]
                    # Accept both numpy arrays and Python lists of numeric values
                    if not isinstance(sample, (np.ndarray, list)):
                        continue
                    raw_lists = [list(v) if v is not None else None for v in df[col]]
                    arrow_arr = pa.array(raw_lists, type=pa.list_(pa.float64()), from_pandas=True)
                    df[col] = pd.Series(pd.arrays.ArrowExtensionArray(arrow_arr), index=df.index)
            except Exception as _cast_exc:
                logger.debug(f"Could not re-wrap embedding columns for {node_type}: {_cast_exc}")

            # Determine primary key from schema info or fallback to 'id' or first column
            target_pk = pk_info.get(node_type)
            pk_field = target_pk or ("id" if "id" in df.columns else df.columns[0])

            # Verify the pk_field exists in the dataframe
            if pk_field not in df.columns:
                logger.warning(
                    f"Primary key '{pk_field}' not found in parquet columns for {node_type}: {list(df.columns)}"
                )
                pk_field = "id" if "id" in df.columns else df.columns[0]

            # Filter out nodes with NULL or empty primary keys (Fix 1: BAML extraction issue)
            initial_count = len(df)
            df = df[df[pk_field].notna() & (df[pk_field] != "")]
            filtered_count = initial_count - len(df)
            if filtered_count > 0:
                logger.warning(
                    f"Filtered out {filtered_count} {node_type} node(s) with NULL or empty primary key '{pk_field}'"
                )

            if df.empty:
                logger.debug(f"No valid {node_type} nodes to import after filtering")
                continue

            # Build MERGE query - merge on primary key, set all other fields
            other_cols = [c for c in df.columns if c != pk_field]
            on_create_set = ", ".join([f"n.{c} = {c}" for c in other_cols]) if other_cols else ""
            on_match_set = ", ".join([f"n.{c} = {c}" for c in other_cols]) if other_cols else ""

            # Convert to Arrow table to bypass Ladybug's buggy NumPy scanner.
            # Use struct_col_types so struct columns preserve schema-defined field order
            # (pa.Table.from_pandas alphabetises struct dict keys).
            # NOTE: arrow_df is read by name from this frame by Ladybug's LOAD FROM scanner.
            arrow_df = _pandas_to_arrow_with_structs(df, struct_field_types=struct_col_types or None)

            if on_create_set:
                merge_query = f"""
                    LOAD FROM arrow_df
                    MERGE (n:{node_type} {{{pk_field}: {pk_field}}})
                    ON CREATE SET {on_create_set}
                    ON MATCH SET {on_match_set}
                """
            else:
                merge_query = f"""
                    LOAD FROM arrow_df
                    MERGE (n:{node_type} {{{pk_field}: {pk_field}}})
                """

            kuzu_conn.execute(merge_query)
            nodes_imported += len(df)
            logger.debug(f"Imported {len(df)} {node_type} nodes from parquet (pk={pk_field})")

            # Also add to collector if active (so the KG's parquet export includes imported data)
            if collector is not None:
                collector.add_nodes(node_type, arrow_df)

        except Exception as exc:
            # Check for struct field order mismatch error
            err_msg = str(exc)
            if "STRUCT" in err_msg and "but expected STRUCT" in err_msg and "Implicit cast is not supported" in err_msg:
                logger.error(
                    f"Failed to import {node_type} nodes: Schema mismatch detected. "
                    f"This usually happens when the BAML schema changed but cached parquet files have the old structure. "
                    f"\n\n💡 Solution: Run 'cli kg create --clear-all-caches' to regenerate all caches."
                )
                logger.error(f"Technical details: {exc}")
            elif "incorrect list entry to ARRAY" in err_msg:
                # Embedding dimension mismatch (e.g., 1024 vs 1536 from different models).
                # Retry without the incompatible embedding columns so non-embedding fields are still imported.
                embedding_cols = [c for c in df.columns if c.endswith("_embedding")]
                logger.warning(
                    f"Embedding dimension mismatch for {node_type} ({err_msg}). "
                    f"Retrying import without embedding columns: {embedding_cols}"
                )
                try:
                    df = df.drop(columns=embedding_cols, errors="ignore")
                    retry_other_cols = [c for c in df.columns if c != pk_field]
                    retry_on_create = ", ".join([f"n.{c} = {c}" for c in retry_other_cols]) if retry_other_cols else ""
                    retry_on_match = ", ".join([f"n.{c} = {c}" for c in retry_other_cols]) if retry_other_cols else ""
                    # NOTE: retry_arrow is read by name from this frame by Ladybug's LOAD FROM scanner.
                    retry_arrow = _pandas_to_arrow_with_structs(df, struct_field_types=struct_col_types or None)
                    if retry_on_create:
                        retry_query = f"""
                            LOAD FROM retry_arrow
                            MERGE (n:{node_type} {{{pk_field}: {pk_field}}})
                            ON CREATE SET {retry_on_create}
                            ON MATCH SET {retry_on_match}
                        """
                    else:
                        retry_query = f"""
                            LOAD FROM retry_arrow
                            MERGE (n:{node_type} {{{pk_field}: {pk_field}}})
                        """
                    kuzu_conn.execute(retry_query)
                    nodes_imported += len(df)
                    logger.info(
                        f"Imported {len(df)} {node_type} nodes without embeddings "
                        f"(dimension mismatch: dropped {embedding_cols})"
                    )
                    if collector is not None:
                        collector.add_nodes(node_type, retry_arrow)
                except Exception as retry_exc:
                    logger.error(f"Failed to import {node_type} nodes even without embeddings: {retry_exc}")
            else:
                logger.error(f"Failed to import {node_type} nodes: {exc}")

    # Import relationships
    for rel_type in manifest.rel_tables:
        parquet_path = rels_dir / f"{rel_type}.parquet"
        if not parquet_path.exists():
            logger.warning(f"Relationship parquet file not found: {parquet_path}")
            continue

        try:
            df = pd.read_parquet(str(parquet_path))
            if df.empty:
                continue

            # Check if this is a collector-saved parquet with metadata columns
            if "_from_type" in df.columns and "_to_type" in df.columns:
                # Group by from_type/to_type combination (there should typically be one)
                for (from_type, to_type), group_df in df.groupby(["_from_type", "_to_type"]):
                    from_key = group_df["_from_key_field"].iloc[0] if "_from_key_field" in group_df.columns else "id"
                    to_key = group_df["_to_key_field"].iloc[0] if "_to_key_field" in group_df.columns else "id"

                    # Remove metadata columns for the merge
                    merge_df = group_df.drop(
                        columns=["_from_type", "_to_type", "_from_key_field", "_to_key_field"], errors="ignore"
                    )

                    # Build property columns (excluding from_id, to_id)
                    prop_cols = [c for c in merge_df.columns if c not in ("from_id", "to_id")]
                    props_str = " {" + ", ".join([f"{c}: {c}" for c in prop_cols]) + "}" if prop_cols else ""

                    # Create relationships using MATCH + CREATE
                    # NOTE: arrow_merge_df is read by name from this frame by Ladybug's LOAD FROM scanner.
                    arrow_merge_df = pa.Table.from_pandas(merge_df)  # noqa: F841
                    create_query = f"""
                        LOAD FROM arrow_merge_df
                        MATCH (a:{from_type} {{{from_key}: from_id}}), (b:{to_type} {{{to_key}: to_id}})
                        CREATE (a)-[:{rel_type}{props_str}]->(b)
                    """

                    kuzu_conn.execute(create_query)
                    rels_imported += len(merge_df)
                    logger.debug(
                        f"Imported {len(merge_df)} {rel_type} relationships from parquet ({from_type}->{to_type})"
                    )

                    # Also add to collector if active (so the KG's parquet export includes imported data)
                    if collector is not None:
                        collector.add_relationships(rel_type, pa.Table.from_pandas(group_df))

        except Exception as exc:
            logger.error(f"Failed to import {rel_type} relationships: {exc}")

    logger.info(f"Imported from parquet '{config_name}': {nodes_imported} nodes, {rels_imported} rels")

    return nodes_imported, rels_imported


def export_warnings(config_name: str, warnings: list[str]) -> Path:
    """Export warnings to a structured Markdown report.

    Args:
        config_name: Name of the KG configuration
        warnings: List of warning messages collected during KG creation

    Returns:
        Path to the exported warnings markdown file
    """
    from genai_graph.kg.export.warnings_report import generate_warnings_markdown

    manager = get_kg_manager()
    manager.ensure_directories_for(config_name)

    # Generate markdown report
    markdown_content = generate_warnings_markdown(warnings)

    # Write to file
    destination = manager.get_warnings_md_path_for(config_name)
    destination.write_text(markdown_content, encoding="utf-8")

    logger.info("Exported warnings report to '{}'", destination)

    return destination


def clear_all_parquet_caches() -> int:
    """Clear all parquet caches from kg_outputs directories.

    This removes all parquet/ subdirectories to force complete regeneration
    of cached data. Useful when schema changes cause incompatibilities.

    Returns:
        Number of cache directories cleared
    """
    import shutil

    from genai_tk.config_mgmt.config_mngr import global_config

    try:
        kg_outputs = global_config().get_dir_path("paths.kg_outputs", create_if_not_exists=False)
    except Exception:
        return 0

    if not kg_outputs.exists():
        return 0

    cleared = 0
    for kg_dir in kg_outputs.iterdir():
        if kg_dir.is_dir():
            parquet_dir = kg_dir / "parquet"
            if parquet_dir.exists():
                shutil.rmtree(str(parquet_dir))
                cleared += 1
                logger.info(f"Cleared parquet cache: {parquet_dir}")

    return cleared
