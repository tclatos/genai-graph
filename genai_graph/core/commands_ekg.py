"""CLI commands for interacting with the Enterprise Knowledge Graph.

This module provides the ``kg`` top-level command (as configured via
``config/overrides.yaml``). The ``create`` command runs a Prefect flow
using an in-process runner so no long-lived Prefect server is required.
"""

from __future__ import annotations

import os
from typing import Annotated

import typer
from genai_tk.main.cli import CliTopCommand
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from genai_graph.core.graph_backend import (
    create_backend_from_config,
    get_backend_storage_path_from_config,
)
from genai_graph.core.kg_manager import get_kg_manager

GRAPH_DB_CONFIG = "default"

console = Console()


def _get_kg_config_name() -> str:
    """Get the configured KG profile from KgManager.

    This keeps command implementations simple while centralising the
    actual logic in :mod:`genai_graph.core.kg_manager`.
    """

    manager = get_kg_manager()
    profile, _ = manager.activate()
    return profile


class EkgCommands(CliTopCommand):
    """Commands for interacting with a Knowledge Graph."""

    def get_description(self) -> tuple[str, str]:  # type: ignore[override]
        return "kg", "Knowledge Graph commands."

    def register_sub_commands(self, cli_app: typer.Typer) -> None:  # type: ignore[override]
        """Register ``kg`` subcommands on the given Typer application."""

        @cli_app.command("create")
        def create(
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
            """Create the KG database and ingest documents using a Prefect flow.

            The flow is executed with an in-process runner and ephemeral client
            so that no long-lived Prefect server or agent is required.
            """

            # Get the configured KG config name.
            from prefect.settings import (
                PREFECT_API_URL,
                PREFECT_SERVER_ALLOW_EPHEMERAL_MODE,
                PREFECT_SERVER_EPHEMERAL_ENABLED,
                temporary_settings,
            )

            from genai_graph.orchestration.flows import create_kg_flow

            cfg_name = _get_kg_config_name()

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

            # Run the Prefect flow with an ephemeral, in-process server by
            # temporarily disabling any configured API URL and enabling
            # Prefect's ephemeral server mode.
            try:
                with temporary_settings(
                    {
                        PREFECT_API_URL: None,
                        PREFECT_SERVER_EPHEMERAL_ENABLED: True,
                        PREFECT_SERVER_ALLOW_EPHEMERAL_MODE: True,
                    }
                ):
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
                        f"[dim]Click the link above or run[/dim] [bold cyan]cli kg view[/bold cyan] [dim]to open it in your browser[/dim]",
                        title="HTML Visualization Ready",
                        border_style="green",
                    )
                )

        @cli_app.command("info")
        def info(
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
            from genai_graph.core.graph_registry import GraphRegistry, get_subgraph
            from genai_graph.core.graph_schema import find_embedded_field_for_class

            _get_kg_config_name()

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

            # Get KG config name
            kg_config_name = _get_kg_config_name()

            # Connect to backend and gather DB-level details
            try:
                backend = create_backend_from_config(GRAPH_DB_CONFIG, kg_config_name)
            except Exception as exc:  # pragma: no cover - defensive
                import traceback as tb

                logger.error(f"Failed to connect to backend: {exc}")
                logger.error(tb.format_exc())
                console.print(f"[red]❌ Unable to connect to EKG backend: {exc}[/red]")
                raise typer.Exit(1) from exc

            db_path = get_backend_storage_path_from_config(GRAPH_DB_CONFIG, kg_config_name)

            manager = get_kg_manager()
            manager.activate()
            active_cfg = manager.profile
            default_kg = manager.ekg_config.kg_config

            info_table = Table(title="Database Information")
            info_table.add_column("Property", style="cyan", no_wrap=True)
            info_table.add_column("Value", style="green")

            info_table.add_row("Database Path", str(db_path))
            info_table.add_row("Database Type", "Cypher Graph Database")
            info_table.add_row("Backend", "Cypher (via GraphBackend abstraction)")
            info_table.add_row("Storage", "Persistent File Storage")
            info_table.add_row("Active KG Config", f"{active_cfg}@{manager.tag}")
            info_table.add_row("Default KG Config", default_kg)
            info_table.add_row("Subgraph(s)", subgraph_title)

            console.print(info_table)
            console.print("")

            # Show KG manager info
            manager = get_kg_manager()
            manager.activate()
            outcome_info = manager.get_info()

            if outcome_info.get("exists"):
                outcome_table = Table(title="KG Outputs & Outcomes")
                outcome_table.add_column("Category", style="cyan", no_wrap=True)
                outcome_table.add_column("Details", style="green")

                outcome_table.add_row("Base Path", outcome_info["base_path"])

                if outcome_info.get("database"):
                    db_info = outcome_info["database"]
                    outcome_table.add_row(
                        "Database Size",
                        f"{db_info['size_mb']:.2f} MB",
                    )

                if outcome_info.get("html_exports"):
                    html_info = outcome_info["html_exports"]
                    outcome_table.add_row(
                        "HTML Exports",
                        f"{html_info['count']} file(s): {', '.join(html_info['files'])}",
                    )

                if outcome_info.get("outcomes"):
                    out_info = outcome_info["outcomes"]
                    outcome_table.add_row("Logged Outcomes", f"{out_info['count']} events")

                if outcome_info.get("warnings"):
                    warn_info = outcome_info["warnings"]
                    outcome_table.add_row("Logged Warnings", f"{warn_info['count']} warnings")

                console.print(outcome_table)
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
                            result_df = backend.execute(f"MATCH (n:{node_type}) RETURN count(n) as count").get_as_df()
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
                    paths_display = "\n".join(f"{fp or '(root)'} → {tp or '(root)'}" for fp, tp in relation.field_paths)
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
                    field_name = find_embedded_field_for_class(node.node_class, embedded_class)
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

            _get_kg_config_name()

            selected_subgraphs = subgraphs or []

            try:
                from genai_graph.core.schema_doc_generator import generate_schema_description

                description = generate_schema_description(selected_subgraphs, print_enums=with_enums)
            except ValueError as exc:
                import traceback as tb

                console.print(f"[red]❌ {exc}[/red]")
                console.print("[red]" + tb.format_exc() + "[/red]")
                raise typer.Exit(1) from exc

            subgraph_title = ", ".join(selected_subgraphs) if selected_subgraphs else "ALL"
            console.print(
                Panel(
                    f"[bold cyan]Schema for {subgraph_title}[/bold cyan]",
                )
            )
            console.print(description)

        # TODO: other commands (delete, export-html, query) can be
        # migrated to Prefect-based flows in a similar fashion if needed.
        @cli_app.command("agent")
        def agent(
            input: Annotated[
                str | None,
                typer.Option(
                    "--input",
                    "-i",
                    help="Input query or '-' to read from stdin",
                ),
            ] = None,
            chat: Annotated[
                bool,
                typer.Option(
                    "--chat",
                    "-s",
                    help="Start an interactive chat session with the EKG agent",
                ),
            ] = False,
            llm: Annotated[
                str | None,
                typer.Option(
                    "--llm",
                    "-m",
                    help="LLM identifier (ID or tag) to use; default comes from configuration",
                ),
            ] = None,
            mcp: Annotated[
                list[str],
                typer.Option(
                    "--mcp",
                    help="MCP server names to connect to (e.g. playwright, filesystem, ..)",
                ),
            ] = [],
            subgraphs: Annotated[
                list[str],
                typer.Option(
                    "--subgraph",
                    "-g",
                    help="Subgraph(s) whose combined schema is used to instruct the agent (default: all)",
                ),
            ] = [],
            debug: Annotated[
                bool,
                typer.Option(
                    "--debug",
                    "-d",
                    help="Display generated Cypher queries before execution",
                ),
            ] = False,
            lc_verbose: Annotated[
                bool,
                typer.Option(
                    "--verbose",
                    "-v",
                    help="Enable LangChain verbose mode",
                ),
            ] = False,
            lc_debug: Annotated[
                bool,
                typer.Option(
                    "--debug-lc",
                    help="Enable LangChain debug mode",
                ),
            ] = False,
        ) -> None:
            """Run an EKG-aware LangChain ReAct agent over the knowledge graph.

            The agent answers questions about enterprise data and can call a
            Cypher execution tool to query the graph when needed.

            Examples:
                uv run cli kg agent -i "List the names of all competitors"
                uv run cli kg agent --chat
                uv run cli kg agent --mcp filesystem -i "List recent EKG exports on disk"
            """
            import asyncio
            import sys

            from genai_tk.cli.langchain_agent import (
                run_langchain_agent_direct,
                run_langchain_agent_shell,
            )
            from genai_tk.extra.agents.langchain_setup import setup_langchain

            from genai_graph.core.ekg_agent import (
                build_ekg_agent_system_prompt,
                create_ekg_cypher_tool,
            )
            from genai_graph.core.graph_registry import GraphRegistry

            # Get KG config name
            kg_config_name = _get_kg_config_name()

            registry = GraphRegistry.get_instance()
            selected_subgraphs = subgraphs or registry.listsubgraphs()

            if not selected_subgraphs:
                console.print("[red]❌ No subgraphs are currently registered.[/red]")
                raise typer.Exit(1)

            setup_langchain(llm, lc_debug, lc_verbose)

            system_prompt = build_ekg_agent_system_prompt(selected_subgraphs)
            ekg_tool = create_ekg_cypher_tool(
                backend_config=GRAPH_DB_CONFIG,
                kg_config_name=kg_config_name,
                console=console,
                debug=debug,
            )

            if chat:
                # Interactive chat mode using the shared LangChain shell
                asyncio.run(
                    run_langchain_agent_shell(
                        llm,
                        tools=[ekg_tool],
                        mcp_server_names=mcp,
                        system_prompt=system_prompt,
                    )
                )
            else:
                # Handle input from --input parameter or stdin
                if not input and not sys.stdin.isatty():
                    input = sys.stdin.read()
                if not input or len(input.strip()) < 3:
                    console.print("[red]❌ Input parameter or something in stdin is required[/red]")
                    raise typer.Exit(1)

                # Reuse the common ReAct helper from genai-tk
                asyncio.run(
                    run_langchain_agent_direct(
                        input.strip(),
                        llm_id=llm,
                        mcp_server_names=mcp,
                        additional_tools=[ekg_tool],
                        pre_prompt=system_prompt,
                    )
                )

        @cli_app.command("cypher")
        def cypher(
            query: str = typer.Argument(help="Cypher query to execute"),
            subgraphs: Annotated[
                list[str],
                typer.Option(
                    "--subgraph",
                    "-g",
                    help="Subgraph(s) whose combined context is being queried (default: all)",
                ),
            ] = [],
        ) -> None:
            """Execute Cypher queries on the EKG database.

            The selected subgraphs are currently used for informational
            purposes only (the Cypher query is executed as-is), but the
            default follows the same semantics as other commands: when
            ``--subgraph`` is omitted, all registered subgraphs are
            considered.
            """

            from rich.panel import Panel
            from rich.table import Table

            from genai_graph.core.graph_backend import (
                create_backend_from_config,
            )
            from genai_graph.core.graph_registry import GraphRegistry

            # Get KG config name
            kg_config_name = _get_kg_config_name()

            registry = GraphRegistry.get_instance()
            selected_subgraphs = subgraphs or registry.listsubgraphs()
            subgraph_label = ", ".join(selected_subgraphs) if selected_subgraphs else "<none>"

            console.print(
                Panel(f"[bold cyan]Querying EKG Database[/bold cyan]\n[dim]Subgraphs: {subgraph_label}[/dim]")
            )

            # Get database connection
            backend = create_backend_from_config(GRAPH_DB_CONFIG, kg_config_name)
            if not backend:
                console.print("[red]❌ No EKG database found[/red]")
                console.print("[yellow]💡 Add data first: [bold]cli kg add --key <data_key>[/bold][/yellow]")
                raise typer.Exit(1)

            def execute_query(cypher_query: str) -> None:
                """Execute a single Cypher query and display results."""
                if not cypher_query.strip():
                    return

                try:
                    console.print(f"[dim]Executing: {cypher_query}[/dim]")
                    result = backend.execute(cypher_query)
                    df = result.get_as_df()

                    if df.empty:
                        console.print("[yellow]Query returned no results[/yellow]")
                        return

                    # Create a Rich table for results
                    table = Table(title=f"Query Results ({len(df)} rows)")

                    # Add columns
                    for col in df.columns:
                        table.add_column(str(col), style="cyan")

                    # Add rows (limit to first 20 for readability)
                    max_rows = 20
                    for i, (_, row) in enumerate(df.iterrows()):
                        if i >= max_rows:
                            table.add_row(*["..." for _ in df.columns])
                            break
                        table.add_row(*[str(val) for val in row])

                    console.print(table)

                    if len(df) > max_rows:
                        console.print(f"[dim]Showing first {max_rows} of {len(df)} results[/dim]")

                except Exception as e:
                    import traceback as tb

                    console.print(f"[red]❌ Query error: {e}[/red]")
                    console.print("[red]" + tb.format_exc() + "[/red]")

            # Execute single query if provided
            if query:
                execute_query(query)
                return

        @cli_app.command("query")
        def query_ekg(
            query: str = typer.Argument(help="query to execute"),
            llm: Annotated[
                str | None,
                typer.Option(help="Name or tag of the LLM to use by BAML"),
            ] = None,
            subgraphs: Annotated[
                list[str],
                typer.Option(
                    "--subgraph",
                    "-g",
                    help="Subgraph(s) whose combined schema is used for text-to-Cypher (default: all)",
                ),
            ] = [],
        ) -> None:
            """Execute queries in natural language (Text-2-Cypher) on the EKG database.

            ex:  List the names of all competitors for opportunities created after January 1, 2012."""

            from genai_graph.core.text2cypher import query_kg

            try:
                from rich.table import Table

                from genai_graph.core.graph_registry import GraphRegistry

                # Get the configured KG config name.
                _get_kg_config_name()

                # If no subgraphs are provided, use all registered ones
                registry = GraphRegistry.get_instance()
                selected_subgraphs = subgraphs or registry.listsubgraphs()

                df = query_kg(query, subgraphs=selected_subgraphs, llm_id=llm)

                if df.empty:
                    console.print("[yellow]Query returned no results[/yellow]")
                    return

                # Create a Rich table for results
                table = Table(title="Query Results")
                for col in df.columns:
                    table.add_column(str(col), style="cyan")
                MAX_ROWS = 20
                for i, (_, row) in enumerate(df.iterrows()):
                    if i >= MAX_ROWS:
                        table.add_row(*["..." for _ in df.columns])
                        break
                    table.add_row(*[str(val) for val in row])
                console.print(table)

                if len(df) > MAX_ROWS:
                    console.print(f"[dim]Showing first {MAX_ROWS} of {len(df)} results[/dim]")

            except Exception as e:
                import traceback as tb

                logger.error(f"Failed to process query: {e}")
                console.print(f"[red]❌ Query error: {e}[/red]")
                console.print("[red]" + tb.format_exc() + "[/red]")
                return

        @cli_app.command("view")
        def view_html() -> None:
            """Open the HTML visualization of the current KG configuration in a browser.

            Opens the most recently generated HTML export file for the active KG
            configuration in the default web browser.
            """
            import webbrowser

            # Get the current KG config
            _get_kg_config_name()
            manager = get_kg_manager()

            # Check if HTML directory exists
            if not manager.html_dir.exists():
                console.print(
                    "[red]❌ No HTML exports found.[/red]\n"
                    "[yellow]💡 Run [bold]cli kg create[/bold] to generate a visualization[/yellow]"
                )
                raise typer.Exit(1)

            # Find the most recent HTML file for this profile
            html_files = list(manager.html_dir.glob(f"{manager.profile}-{manager.tag}*.html"))

            if not html_files:
                console.print(
                    f"[red]❌ No HTML export found for config '{manager.profile}@{manager.tag}'[/red]\n"
                    "[yellow]💡 Run [bold]cli kg create --export-html[/bold] to generate one[/yellow]"
                )
                raise typer.Exit(1)

            # Get the most recent file (by modification time)
            html_file = max(html_files, key=lambda p: p.stat().st_mtime)
            file_url = html_file.as_uri()

            console.print(f"[bold cyan]🌐 Opening HTML visualization:[/bold cyan] {html_file.name}")

            # Open in browser
            webbrowser.open(file_url)

            console.print("[green]✓ Opened in your default browser[/green]")
