"""CLI commands for interacting with the Enterprise Knowledge Graph.

This module provides the ``kg`` top-level command (as configured via
``config/overrides.yaml``) and routes the ``create`` command through the
Prefect-based orchestration flow defined in ``genai_graph.orchestration``.
"""

from __future__ import annotations

import os
from typing import Annotated

import typer
from genai_tk.main.cli import CliTopCommand
from genai_tk.utils.config_mngr import global_config
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from genai_graph.core.graph_backend import (
    create_backend_from_config,
    get_backend_storage_path_from_config,
)
from genai_graph.core.graph_registry import GraphRegistry, get_subgraph
from genai_graph.core.graph_schema import _find_embedded_field_for_class
from genai_graph.core.schema_doc_generator import (
    _format_schema_description,
    _parse_baml_descriptions,
)
from genai_graph.orchestration.flows import create_kg_flow

GRAPH_DB_CONFIG = "default"

console = Console()


def _get_kg_config_name(config_name: str | None) -> str:
    """Resolve the KG configuration name and update global config.

    This mirrors the behavior of the historical ``get_kg_config_name`` helper
    so that other components (e.g. GraphRegistry) see the same selected config.
    """

    cfg = global_config()

    if config_name:
        cfg.set("kg_config", config_name)
        return config_name

    default_from_config = cfg.get("default_kg_config")
    if default_from_config:
        cfg.set("kg_config", default_from_config)
        return default_from_config

    cfg.set("kg_config", "default")
    return "default"


class EkgCommands(CliTopCommand):
    """Commands for interacting with a Knowledge Graph."""

    def get_description(self) -> tuple[str, str]:  # type: ignore[override]
        return "kg", "Knowledge Graph commands."

    def register_sub_commands(self, cli_app: typer.Typer) -> None:  # type: ignore[override]
        """Register ``kg`` subcommands on the given Typer application."""

        @cli_app.command("create")
        def create(
            config_name: Annotated[
                str | None,
                typer.Option(
                    "--config",
                    help=(
                        "Name of the KG config to use from config/ekg.yaml "
                        "(default: value of key 'kg_config' or 'default_kg_config')"
                    ),
                ),
            ] = None,
            delete_first: Annotated[
                bool,
                typer.Option(
                    "--delete-first/--no-delete-first",
                    help="Delete existing KG database before creation",
                ),
            ] = True,
            export_html: Annotated[
                bool,
                typer.Option(
                    "--export-html/--no-export-html",
                    help="Export HTML visualization after creation",
                ),
            ] = True,
        ) -> None:
            """Create the KG database and ingest documents using Prefect.

            This command now delegates the heavy lifting to the Prefect flow
            ``create_kg_flow`` while preserving the observable behavior
            (configuration handling, warning display, and summary output).
            """

            # Resolve config name and keep global_config in sync for the rest
            # of the system (GraphRegistry, etc.).
            cfg_name = _get_kg_config_name(config_name)

            console.print(f"[bold]Creating KG using config[/bold] [cyan]{cfg_name}[/cyan]...")

            # Prefect 3 starts a temporary local server when no API URL is
            # configured. In proxy-heavy environments this can cause localhost
            # traffic to be routed through an HTTP proxy, leading to timeouts.
            # Be explicit and strip proxy settings for this command while
            # keeping localhost in NO_PROXY.
            for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
                os.environ.pop(var, None)
            os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
            os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

            try:
                result = create_kg_flow(
                    config_name=cfg_name,
                    delete_first=delete_first,
                    export_html=export_html,
                )
            except Exception as exc:  # pragma: no cover - defensive
                import traceback as tb

                logger.error(f"KG creation failed: {exc}")
                logger.error(tb.format_exc())
                console.print(f"[red]❌ KG creation failed: {exc}[/red]")
                raise typer.Exit(1) from exc

            stats = result.stats
            warnings = result.warnings

            console.print("")
            console.print(
                f"[green]✓ KG creation completed.[/green] Processed: "
                f"{stats.total_processed} ok, {stats.total_failed} failed. "
                f"Path: {result.db_path}",
            )

            if warnings:
                console.print(Panel.fit("[bold yellow]⚠️  Warnings[/bold yellow]", border_style="yellow"))
                for idx, warning in enumerate(warnings, 1):
                    console.print(f"  [yellow]{idx}.[/yellow] {warning}")
                console.print("")
            else:
                console.print("[green]✓ No warnings[/green]")

            if result.html_export and export_html:
                file_url = f"file://{result.html_export.output_path}"
                console.print(
                    Panel(
                        f"[bold green]🌐 HTML export created:[/bold green]\n\n"
                        f"[link={file_url}]{file_url}[/link]\n\n"
                        f"[dim]Click the link above to open the visualization[/dim]",
                        title="HTML Visualization Ready",
                        border_style="green",
                    )
                )

        @cli_app.command("info")
        def info(
            config_name: Annotated[
                str | None,
                typer.Option(
                    "--config",
                    help=(
                        "Name of the KG config to use from config/ekg.yaml "
                        "(default: value of key 'kg_config' or 'default_kg_config')"
                    ),
                ),
            ] = None,
            subgraphs: Annotated[
                list[str],
                typer.Option(
                    "--subgraph",
                    "-g",
                    help="Subgraph(s) to display info for; default is all registered",
                ),
            ] = [],
        ) -> None:
            """Display EKG database information, schema overview, and mappings."""

            _get_kg_config_name(config_name)

            # Build registry and select subgraphs
            registry = GraphRegistry()
            selected_subgraphs = subgraphs or registry.listsubgraphs()

            try:
                schema = registry.build_combined_schema(selected_subgraphs)
            except ValueError as exc:
                import traceback as tb

                console.print(f"[red]❌ {exc}[/red]")
                console.print("[red]" + tb.format_exc() + "[/red]")
                raise typer.Exit(1) from exc

            subgraph_title = ", ".join(selected_subgraphs) if selected_subgraphs else "ALL"
            console.print(
                Panel(
                    f"[bold cyan]{subgraph_title} EKG Database Information[/bold cyan]",
                )
            )

            # Connect to backend and gather DB-level details
            try:
                backend = create_backend_from_config(GRAPH_DB_CONFIG)
            except Exception as exc:  # pragma: no cover - defensive
                import traceback as tb

                logger.error(f"Failed to connect to backend: {exc}")
                logger.error(tb.format_exc())
                console.print(f"[red]❌ Unable to connect to EKG backend: {exc}[/red]")
                raise typer.Exit(1) from exc

            db_path = get_backend_storage_path_from_config(GRAPH_DB_CONFIG)
            cfg = global_config()
            active_cfg = cfg.get("kg_config", default="(not set)")
            default_kg = cfg.get("default_kg_config", default="(not set)")

            info_table = Table(title="Database Information")
            info_table.add_column("Property", style="cyan", no_wrap=True)
            info_table.add_column("Value", style="green")

            info_table.add_row("Database Path", str(db_path))
            info_table.add_row("Database Type", "Cypher Graph Database")
            info_table.add_row("Backend", "Cypher (via GraphBackend abstraction)")
            info_table.add_row("Storage", "Persistent File Storage")
            info_table.add_row("Active KG Config", active_cfg)
            info_table.add_row("Default KG Config", default_kg)
            info_table.add_row("Subgraph(s)", subgraph_title)

            console.print(info_table)
            console.print("")

            # Subgraph factories
            console.print("[bold cyan]Subgraph Factories[/bold cyan]")
            factory_table = Table(title="Registered Subgraph Factories")
            factory_table.add_column("Name", style="cyan", no_wrap=True)
            factory_table.add_column("Type", style="yellow")
            factory_table.add_column("Module", style="dim")

            for name in selected_subgraphs or registry.listsubgraphs():
                try:
                    subgraph_impl = get_subgraph(name)
                    factory_type = type(subgraph_impl).__name__
                    factory_module = type(subgraph_impl).__module__
                    factory_table.add_row(name, factory_type, factory_module)
                except ValueError:
                    factory_table.add_row(name, "[red]Not Found[/red]", "")

            console.print(factory_table)
            console.print("")

            # Schema-level statistics from backend
            from rich.table import Table as RichTable

            try:
                tables_result = backend.execute("CALL show_tables() RETURN *")
                tables_df = tables_result.get_as_df()

                node_tables: list[str] = []
                rel_tables: list[str] = []

                for _, row in tables_df.iterrows():
                    if row.get("type") == "NODE":
                        node_tables.append(row["name"])
                    elif row.get("type") == "REL":
                        rel_tables.append(row["name"])

                if schema:
                    allowed_node_labels = {n.node_class.__name__ for n in schema.nodes}
                    allowed_rel_types = {r.name for r in schema.relations}

                    node_tables = [t for t in node_tables if t in allowed_node_labels]
                    rel_tables = [t for t in rel_tables if t in allowed_rel_types]

                schema_table = RichTable(title="Schema Overview")
                schema_table.add_column("Component", style="cyan", no_wrap=True)
                schema_table.add_column("Count", justify="right", style="magenta")

                schema_table.add_row("Node Tables", str(len(node_tables)))
                schema_table.add_row("Relationship Tables", str(len(rel_tables)))

                console.print(schema_table)
                console.print("")

                # Node counts
                if node_tables:
                    node_stats_table = RichTable(title="Node Counts")
                    node_stats_table.add_column("Node Type", style="cyan", no_wrap=True)
                    node_stats_table.add_column("Count", justify="right", style="magenta")

                    for node_type in sorted(node_tables):
                        try:
                            result_df = backend.execute(
                                f"MATCH (n:{node_type}) RETURN count(n) as count"
                            ).get_as_df()
                            count = result_df.iloc[0]["count"]
                            node_stats_table.add_row(node_type, str(count))
                        except Exception as exc:  # pragma: no cover - defensive
                            import traceback as tb

                            node_stats_table.add_row(node_type, f"[red]Error: {exc}[/red]")
                            logger.debug("Failed to get count for {}: {}", node_type, tb.format_exc())

                    console.print(node_stats_table)
                    console.print("")

                # Relationship counts
                if rel_tables:
                    rel_stats_table = RichTable(title="Relationship Counts")
                    rel_stats_table.add_column("Relationship Type", style="cyan", no_wrap=True)
                    rel_stats_table.add_column("Count", justify="right", style="magenta")

                    for rel_type in sorted(rel_tables):
                        try:
                            result_df = backend.execute(
                                f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count"
                            ).get_as_df()
                            count = result_df.iloc[0]["count"]
                            rel_stats_table.add_row(rel_type, str(count))
                        except Exception as exc:  # pragma: no cover - defensive
                            import traceback as tb

                            rel_stats_table.add_row(rel_type, f"[red]Error: {exc}[/red]")
                            logger.debug("Failed to get count for {}: {}", rel_type, tb.format_exc())

                    console.print(rel_stats_table)
                    console.print("")

            except Exception as exc:  # pragma: no cover - defensive
                import traceback as tb

                console.print(f"[red]Error retrieving schema information: {exc}[/red]")
                console.print("[red]" + tb.format_exc() + "[/red]")

            # Node mapping
            console.print(
                Panel(
                    f"[bold cyan]Node Mapping for {subgraph_title}[/bold cyan]",
                )
            )

            mapping_table = Table(title="Node Type → Description and Deduplication")
            mapping_table.add_column("Node Type", style="cyan", no_wrap=True)
            mapping_table.add_column("Description", style="yellow")
            mapping_table.add_column("Dedup Key", style="magenta")
            mapping_table.add_column("Alt Names Field", style="green")

            for node in schema.nodes:
                node_type = node.node_class.__name__
                description = node.description or ""

                if node.deduplication_key is None:
                    dedup_label = "_name (default)"
                elif isinstance(node.deduplication_key, str):
                    dedup_label = node.deduplication_key
                else:
                    dedup_label = "callable"

                alt_label = "" if node.deduplication_key is None else "alternate_names"

                mapping_table.add_row(node_type, description, dedup_label, alt_label)

            console.print(mapping_table)
            console.print("")

            # Relationship mapping
            rel_mapping_table = Table(title="Relationship Type → Semantic Meaning")
            rel_mapping_table.add_column("Relationship", style="cyan", no_wrap=True)
            rel_mapping_table.add_column("From → To", style="green")
            rel_mapping_table.add_column("Meaning", style="yellow")
            rel_mapping_table.add_column("Field Paths", style="magenta")

            for relation in schema.relations:
                rel_type = relation.name
                direction = f"{relation.from_node.__name__} → {relation.to_node.__name__}"
                meaning = relation.description or ""

                if relation.field_paths:
                    paths_display = "\n".join(
                        f"{fp or '(root)'} → {tp or '(root)'}" for fp, tp in relation.field_paths
                    )
                else:
                    paths_display = "[dim](none)[/dim]"

                rel_mapping_table.add_row(rel_type, direction, meaning, paths_display)

            console.print(rel_mapping_table)
            console.print("")

            # Embedded fields
            console.print("[bold cyan]Embedded Fields[/bold cyan]")
            embedded_table = Table(title="Fields Embedded in Parent Nodes")
            embedded_table.add_column("Parent Node", style="cyan", no_wrap=True)
            embedded_table.add_column("Embedded Field", style="green")
            embedded_table.add_column("Embedded Class", style="magenta")

            has_embedded = False
            for node in schema.nodes:
                for embedded_class in getattr(node, "embedded_struct_classes", []) or []:
                    field_name = _find_embedded_field_for_class(node.node_class, embedded_class)
                    if not field_name:
                        continue
                    has_embedded = True
                    embedded_table.add_row(
                        node.node_class.__name__,
                        field_name,
                        embedded_class.__name__,
                    )

            if has_embedded:
                console.print(embedded_table)
            else:
                console.print("[dim]No embedded fields configured[/dim]")

        @cli_app.command("schema")
        def schema(
            config_name: Annotated[
                str | None,
                typer.Option(
                    "--config",
                    help=(
                        "Name of the KG config to use from config/ekg.yaml "
                        "(default: value of key 'kg_config' or 'default_kg_config')"
                    ),
                ),
            ] = None,
            subgraphs: Annotated[
                list[str],
                typer.Option(
                    "--subgraph",
                    "-g",
                    help="Subgraph(s) to display schema for; default is all registered",
                ),
            ] = [],
            with_enums: Annotated[
                bool,
                typer.Option(
                    "--with-enums/--no-enums",
                    help="Include or exclude enumerations in the schema output (default: include)",
                    show_default=True,
                ),
            ] = True,
        ) -> None:
            """Display knowledge graph schema as used in LLM context.

            Generates a comprehensive, compact Markdown description of the
            graph schema, including node types, relationships, properties, and
            indexed fields. This output is suitable for feeding into LLMs for
            query generation.
            """

            _get_kg_config_name(config_name)

            registry = GraphRegistry()
            selected_subgraphs = subgraphs or registry.listsubgraphs()

            try:
                schema_obj = registry.build_combined_schema(selected_subgraphs)
            except ValueError as exc:
                import traceback as tb

                console.print(f"[red]❌ {exc}[/red]")
                console.print("[red]" + tb.format_exc() + "[/red]")
                raise typer.Exit(1) from exc

            baml_docs = _parse_baml_descriptions()
            description = _format_schema_description(
                schema=schema_obj,
                baml_docs=baml_docs,
                print_enums=with_enums,
            )

            subgraph_title = ", ".join(selected_subgraphs) if selected_subgraphs else "ALL"
            console.print(
                Panel(
                    f"[bold cyan]Schema for {subgraph_title}[/bold cyan]",
                )
            )
            console.print(description)

        # TODO: other commands (delete, export-html, query) can be
        # migrated to Prefect-based flows in a similar fashion if needed.

        
