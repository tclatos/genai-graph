#!/usr/bin/env python3
"""Interactive EKG (Enhanced Knowledge Graph) CLI.

A comprehensive Typer-based CLI for managing a single Cypher knowledge graph
created from BAML-structured data. Provides commands for adding opportunity data,
querying with Cypher, and exporting visualizations.

Features:
    - Add opportunity data to shared knowledge base
    - Execute interactive Cypher queries
    - Display database schema and statistics
    - Export HTML visualizations with clickable links
    - Rich console output with colors and tables

Commands:
    cli kg add --key OPPORTUNITY_KEY     Add opportunity data to KB
    cli kg delete                        Delete entire KB
    cli kg query                         Interactive Cypher query shell
    cli kg info                          Display DB info and schema
    cli kg export-html                   Export HTML visualization

Usage Examples:
    ```bash
    # Add opportunity data to knowledge base
    uv run cli kg add --key cnes-venus-tma

    # Query the knowledge base interactively
    uv run cli kg query

    # Display database information and mapping
    uv run cli kg info

    # Export HTML visualization
    uv run cli kg export-html
    ```
"""

import webbrowser
from pathlib import Path
from typing import Annotated

import typer
from genai_tk.main.cli import CliTopCommand
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

# Initialize Rich console
console = Console()

# Configuration constants
KV_STORE_ID = "file"
GRAPH_DB_CONFIG = "default"


class EkgCommands(CliTopCommand):
    """Commands for interacting with a Knowledge Graph."""

    def get_description(self) -> tuple[str, str]:
        return "kg", "Knowledge Graph commands."

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        # Import concrete subgraph modules so they can register themselves.
        # This keeps the CLI generic while still enabling default subgraphs.

        @cli_app.command("add-doc")
        def add_doc(
            keys: Annotated[
                list[str],
                typer.Option(
                    "--key", "-k", help="Data key(s) to add to the EKG database (can be specified multiple times)"
                ),
            ],
            subgraph: Annotated[str, typer.Option("--subgraph", "-g", help="Subgraph type to use")],
        ) -> None:
            """Add one or more documents to the shared EKG database.

            Loads data from the key-value store and adds it to the
            shared EKG database, creating the database if it doesn't exist.
            Supports incremental additions - nodes with the same _name will be merged,
            preserving creation timestamps and updating modification timestamps.

            Examples:
                # Add single document
                cli kg add --key fake_cnes_1

                # Add multiple documents in one command
                cli kg add --key fake_cnes_1 --key cnes-venus-tma

                # Add documents sequentially (without deleting DB in between)
                cli kg add --key fake_cnes_1
                cli kg add --key cnes-venus-tma
            """
            from genai_graph.core.graph_backend import (
                create_backend_from_config,
                get_backend_storage_path_from_config,
            )
            from genai_graph.core.graph_documents import add_documents_to_graph
            from genai_graph.core.graph_registry import get_subgraph

            # Get subgraph implementation
            try:
                subgraph_impl = get_subgraph(subgraph)
            except ValueError as e:
                import traceback as tb

                console.print(f"[red]❌ {e}[/red]")
                console.print("[red]" + tb.format_exc() + "[/red]")
                raise typer.Exit(1) from e

            # Validate that we have at least one key
            if not keys:
                console.print("[red]❌ At least one --key must be provided[/red]")
                raise typer.Exit(1)

            console.print(Panel(f"[bold cyan]Adding {len(keys)} Document(s) to EKG[/bold cyan]"))

            # Create graph configuration once
            console.print("⚙️  Creating graph schema...")
            schema = subgraph_impl.build_schema()
            console.print(
                f"[green]✓[/green] Schema: {len(schema.nodes)} node types, {len(schema.relations)} relationships"
            )

            # Initialize or connect to database once
            console.print("🔧 Connecting to EKG database...")
            backend = create_backend_from_config("default")

            # Process documents with enhanced error handling and per-document status
            db_path = get_backend_storage_path_from_config("default")

            try:
                # Process each key sequentially with UI feedback
                total_docs_processed = 0
                total_docs_failed = 0

                for idx, key in enumerate(keys, 1):
                    console.print(f"\n[bold cyan]═══ Processing document {idx}/{len(keys)}: {key} ═══[/bold cyan]")

                    try:
                        # Load data
                        console.print("📁 Loading data...")
                        data = subgraph_impl.load_data(key)
                        if not data:
                            console.print(f"[red]❌ No {subgraph} data found for key: {key}[/red]")
                            total_docs_failed += 1
                            continue

                        entity_name = subgraph_impl.get_entity_name_from_data(data)
                        console.print(f"[green]✓[/green] Loaded: [bold]{entity_name}[/bold]")

                        # Delegate complete document ingestion logic to graph_documents
                        console.print("🚀 Merging into graph...")
                        try:
                            stats = add_documents_to_graph([key], subgraph_impl, backend, schema)
                        except Exception as e:
                            import traceback as tb

                            console.print(f"[red]❌ {e}[/red]")
                            console.print("[red]" + tb.format_exc() + "[/red]")
                            total_docs_failed += 1
                            continue

                        if stats.total_processed > 0:
                            total_docs_processed += 1
                            console.print(
                                f"[green]✓[/green] Document {idx} processed: {stats.nodes_created} nodes, {stats.relationships_created} relationships"
                            )
                        else:
                            total_docs_failed += 1
                            console.print(f"[red]❌ Failed to process document: {key}[/red]")

                    except Exception as e:
                        import traceback as tb

                        console.print(f"[red]❌ Error processing {key}: {e}[/red]")
                        console.print("[red]" + tb.format_exc() + "[/red]")
                        total_docs_failed += 1
                        continue

                # Display final summary
                console.print(f"\n{'═' * 60}")
                if total_docs_processed > 0:
                    console.print(
                        Panel(
                            f"[bold green]✅ Successfully processed {total_docs_processed}/{len(keys)} documents![/bold green]"
                        )
                    )
                else:
                    console.print(Panel("[bold red]❌ Failed to process any documents[/bold red]"))
                    raise typer.Exit(1)

                summary_table = Table(title="Final Summary")
                summary_table.add_column("Metric", style="cyan", no_wrap=True)
                summary_table.add_column("Value", justify="right", style="magenta")

                summary_table.add_row("Documents Processed", f"{total_docs_processed}/{len(keys)}")
                if total_docs_failed > 0:
                    summary_table.add_row("Documents Failed", str(total_docs_failed))
                summary_table.add_row("Database Path", str(db_path))

                console.print(summary_table)

                console.print("\n[green]💡 Next steps:[/green]")
                console.print("   • Query: [bold]cli kg query[/bold]")
                console.print("   • Info:  [bold]cli kg info[/bold]")
                console.print("   • Export: [bold]cli kg export-html[/bold]")

            except Exception as e:
                import traceback as tb

                console.print(f"[red]❌ Unexpected error during document ingestion: {e}[/red]")
                console.print("[red]" + tb.format_exc() + "[/red]")
                raise typer.Exit(1) from e

        @cli_app.command("delete")
        def delete_ekg(
            force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation prompts")] = False,
        ) -> None:
            """Delete the entire EKG database.

            Safely removes the shared database directory after confirmation.
            All opportunity data will be lost.
            """

            from genai_graph.core.graph_backend import (
                create_backend_from_config,
                delete_backend_storage_from_config,
            )

            try:
                # Try to get some basic stats
                backend = create_backend_from_config(GRAPH_DB_CONFIG)
                if backend:
                    try:
                        tables_result = backend.execute("CALL show_tables() RETURN *")
                        tables_df = tables_result.get_as_df()
                        node_count = len([row for _, row in tables_df.iterrows() if row.get("type") == "NODE"])
                        rel_count = len([row for _, row in tables_df.iterrows() if row.get("type") == "REL"])
                        console.print(f"📊 Contains {node_count} node tables and {rel_count} relationship tables")

                        # Try to get total record counts
                        total_nodes = 0
                        for _, row in tables_df.iterrows():
                            if row.get("type") == "NODE":
                                try:
                                    result = backend.execute(f"MATCH (n:{row['name']}) RETURN count(n) as count")
                                    count = result.get_as_df().iloc[0]["count"]
                                    total_nodes += count
                                except Exception:
                                    pass
                        console.print(f"📊 Total nodes in database: {total_nodes}")
                    except Exception:
                        console.print("📊 Database exists but couldn't read statistics")
            except Exception:
                pass

            # Confirmation
            if not force:
                if not Confirm.ask("[bold red]Are you sure you want to delete the ENTIRE EKG database?[/bold red]"):
                    console.print("[yellow]Operation cancelled[/yellow]")
                    raise typer.Exit(0)

                # Final confirmation for safety
                if not Confirm.ask(
                    "[bold red]This will delete ALL opportunity data. This action cannot be undone. Continue?[/bold red]"
                ):
                    console.print("[yellow]Operation cancelled[/yellow]")
                    raise typer.Exit(0)
            else:
                console.print("[yellow]⚠️  Force mode enabled - skipping confirmation prompts[/yellow]")

            # Delete the database
            console.print("🗑️  Deleting EKG database...")
            try:
                delete_backend_storage_from_config("default")
                console.print("[green]✅ EKG database deleted successfully[/green]")
                console.print(
                    "[green]You can now start fresh with [bold]cli kg add --key <opportunity_key>[/bold][/green]"
                )
            except Exception as e:
                import traceback as tb

                console.print(f"[red]❌ Error deleting database: {e}[/red]")
                console.print("[red]" + tb.format_exc() + "[/red]")
                raise typer.Exit(1) from e

        @cli_app.command("query")
        def query_ekg(
            query: str = typer.Argument(help="query to execute"),
            config_name: Annotated[
                str,
                typer.Option(
                    "--config",
                    help="Name of the structured config to use from yaml config",
                ),
            ] = "default",
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
                from genai_graph.core.graph_registry import GraphRegistry

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

            registry = GraphRegistry.get_instance()
            selected_subgraphs = subgraphs or registry.listsubgraphs()

            if not selected_subgraphs:
                console.print("[red]❌ No subgraphs are currently registered.[/red]")
                raise typer.Exit(1)

            setup_langchain(llm, lc_debug, lc_verbose)

            system_prompt = build_ekg_agent_system_prompt(selected_subgraphs)
            ekg_tool = create_ekg_cypher_tool(
                backend_config=GRAPH_DB_CONFIG,
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

            from genai_graph.core.graph_backend import (
                create_backend_from_config,
            )
            from genai_graph.core.graph_registry import GraphRegistry

            registry = GraphRegistry.get_instance()
            selected_subgraphs = subgraphs or registry.listsubgraphs()
            subgraph_label = ", ".join(selected_subgraphs) if selected_subgraphs else "<none>"

            console.print(
                Panel(f"[bold cyan]Querying EKG Database[/bold cyan]\n[dim]Subgraphs: {subgraph_label}[/dim]")
            )

            # Get database connection
            backend = create_backend_from_config(GRAPH_DB_CONFIG)
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

        @cli_app.command("info")
        def show_info(
            subgraphs: Annotated[
                list[str],
                typer.Option(
                    "--subgraph",
                    "-g",
                    help="Subgraph(s) to display info for; default is all registered",
                ),
            ] = [],
        ) -> None:
            """Display EKG database information, schema, and entity mapping.

            Shows comprehensive information about the EKG database including
            node/relationship counts, schema details, and semantic mapping.
            """

            from genai_graph.core.graph_backend import (
                create_backend_from_config,
                get_backend_storage_path_from_config,
            )
            from genai_graph.core.graph_registry import GraphRegistry

            registry = GraphRegistry.get_instance()
            selected_subgraphs = subgraphs or registry.listsubgraphs()

            try:
                schema = registry.build_combined_schema(selected_subgraphs)
            except ValueError as e:
                import traceback as tb

                console.print(f"[red]❌ {e}[/red]")
                console.print("[red]" + tb.format_exc() + "[/red]")
                raise typer.Exit(1) from e

            subgraph_title = ", ".join(selected_subgraphs) if selected_subgraphs else "ALL"

            console.print(
                Panel(
                    f"[bold cyan]{subgraph_title} EKG Database Information[/bold cyan]",
                )
            )

            # Get database connection
            backend = create_backend_from_config(GRAPH_DB_CONFIG)
            if not backend:
                console.print("[red]❌ No EKG database found[/red]")
                console.print("[yellow]💡 Add data first: [bold]cli kg add --key <data_key>[/bold][/yellow]")
                raise typer.Exit(1)

            console.print("[green]✅ Connected to EKG database[/green]\n")

            # Database location info
            db_path = get_backend_storage_path_from_config("default")
            info_table = Table(title="Database Information")
            info_table.add_column("Property", style="cyan", no_wrap=True)
            info_table.add_column("Value", style="green")

            info_table.add_row("Database Path", str(db_path))
            info_table.add_row("Database Type", "Cypher Graph Database")
            info_table.add_row("Backend", "Cypher (via GraphBackend abstraction)")
            info_table.add_row("Storage", "Persistent File Storage")
            info_table.add_row("Subgraph(s)", ", ".join(selected_subgraphs) if selected_subgraphs else "ALL")

            console.print(info_table)
            console.print()

            # Get schema information
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

                # When a subset of subgraphs is selected, restrict statistics
                # to node/relationship labels that appear in the combined
                # schema. This keeps counts aligned with the logical graph
                # being inspected.
                #
                # In addition, always surface the system-level Document node
                # and all of its SOURCE* relationships so callers can see
                # which documents were ingested, even when multiple subgraphs
                # are present and Kuzu requires type-specific relationship
                # tables such as SOURCE_ReviewedOpportunity or
                # SOURCE_SWArchitectureDocument.
                if schema:
                    allowed_node_labels = {n.baml_class.__name__ for n in schema.nodes}
                    allowed_rel_types = {r.name for r in schema.relations}

                    # Filter node and relationship tables to the configured schema
                    filtered_node_tables: list[str] = [t for t in node_tables if t in allowed_node_labels]
                    node_tables = filtered_node_tables

                    filtered_rel_tables: list[str] = [t for t in rel_tables if t in allowed_rel_types]
                    rel_tables = filtered_rel_tables

                # Schema overview
                schema_table = Table(title="Schema Overview")
                schema_table.add_column("Component", style="cyan", no_wrap=True)
                schema_table.add_column("Count", justify="right", style="magenta")

                schema_table.add_row("Node Tables", str(len(node_tables)))
                schema_table.add_row("Relationship Tables", str(len(rel_tables)))

                console.print(schema_table)
                console.print()

                # Node statistics
                if node_tables:
                    node_stats_table = Table(title="Node Counts")
                    node_stats_table.add_column("Node Type", style="cyan", no_wrap=True)
                    node_stats_table.add_column("Count", justify="right", style="magenta")

                    for node_type in sorted(node_tables):
                        try:
                            result = backend.execute(f"MATCH (n:{node_type}) RETURN count(n) as count")
                            count = result.get_as_df().iloc[0]["count"]
                            node_stats_table.add_row(node_type, str(count))
                        except Exception as e:
                            import traceback as tb

                            node_stats_table.add_row(node_type, f"[red]Error: {e}[/red]")
                            logger.debug(f"Failed to get count for {node_type}: {tb.format_exc()}")

                    console.print(node_stats_table)
                    console.print()

                # Relationship statistics
                if rel_tables:
                    rel_stats_table = Table(title="Relationship Counts")
                    rel_stats_table.add_column("Relationship Type", style="cyan", no_wrap=True)
                    rel_stats_table.add_column("Count", justify="right", style="magenta")

                    for rel_type in sorted(rel_tables):
                        try:
                            result = backend.execute(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count")
                            count = result.get_as_df().iloc[0]["count"]
                            rel_stats_table.add_row(rel_type, str(count))
                        except Exception as e:
                            import traceback as tb

                            rel_stats_table.add_row(rel_type, f"[red]Error: {e}[/red]")
                            logger.debug(f"Failed to get count for {rel_type}: {tb.format_exc()}")

                    console.print(rel_stats_table)
                    console.print()

            except Exception as e:
                import traceback as tb

                console.print(f"[red]Error retrieving schema information: {e}[/red]")
                console.print("[red]" + tb.format_exc() + "[/red]")

            # Node Mapping
            console.print(Panel(f"[bold cyan]{subgraph_title} Node Mapping[/bold cyan]"))

            mapping_table = Table(title="Node Type → Description and Deduplication")
            mapping_table.add_column("Graph Node Type", style="cyan", no_wrap=True)
            mapping_table.add_column("Description", style="yellow")
            mapping_table.add_column("Dedup Key", style="magenta")
            mapping_table.add_column("Alt Names Field", style="green")

            # Get node labels and deduplication strategy from the combined schema
            for node in schema.nodes:
                node_type = node.baml_class.__name__
                description = node.description or ""

                # Human-readable dedup key description
                if node.deduplication_key is None:
                    dedup_label = "_name (default)"
                elif isinstance(node.deduplication_key, str):
                    dedup_label = node.deduplication_key
                else:
                    # Callable – we do not introspect further to keep output stable
                    dedup_label = "callable"

                # Only highlight alternate_names when a custom deduplication strategy
                # is configured. For nodes using the default _name-based dedup, the
                # alternate name feature is typically not relevant in practice.
                if node.deduplication_key is None:
                    alt_label = ""
                else:
                    alt_label = "alternate_names"

                mapping_table.add_row(node_type, description, dedup_label, alt_label)

            console.print(mapping_table)
            console.print()

            # Relationship mapping
            rel_mapping_table = Table(title="Relationship Type → Semantic Meaning")
            rel_mapping_table.add_column("Relationship", style="cyan", no_wrap=True)
            rel_mapping_table.add_column("From → To", style="green")
            rel_mapping_table.add_column("Meaning", style="yellow")

            # Get relationship labels from the combined schema
            for relation in schema.relations:
                rel_type = relation.name
                direction = f"{relation.from_node.__name__} → {relation.to_node.__name__}"
                meaning = relation.description or ""
                rel_mapping_table.add_row(rel_type, direction, meaning)

            console.print(rel_mapping_table)

            # Indexed fields information
            console.print("\n[bold cyan]Indexed Fields (Vector Store)[/bold cyan]")
            indexed_fields_table = Table(title="Vector Store Indexed Fields")
            indexed_fields_table.add_column("Node Type", style="cyan", no_wrap=True)
            indexed_fields_table.add_column("Indexed Fields", style="yellow")

            has_indexed = False
            for node in schema.nodes:
                if node.index_fields:
                    has_indexed = True
                    fields_str = ", ".join(node.index_fields)
                    indexed_fields_table.add_row(node.baml_class.__name__, fields_str)

            if has_indexed:
                console.print(indexed_fields_table)
            else:
                console.print("[dim]No fields are configured for vector indexing[/dim]")

            # Embedded fields information
            console.print("\n[bold cyan]Embedded Fields[/bold cyan]")
            embedded_table = Table(title="Fields Embedded in Parent Nodes")
            embedded_table.add_column("Parent Node", style="cyan", no_wrap=True)
            embedded_table.add_column("Embedded Field", style="green")
            embedded_table.add_column("Embedded Class", style="magenta")

            has_embedded = False
            for node in schema.nodes:
                if node.embedded:
                    has_embedded = True
                    for field_name, embedded_class in node.embedded:
                        embedded_table.add_row(node.baml_class.__name__, field_name, embedded_class.__name__)

            if has_embedded:
                console.print(embedded_table)
            else:
                console.print("[dim]No embedded fields configured[/dim]")

            # Quick query suggestions
            console.print("\n[green]💡 Try these queries:[/green]")
            console.print('   • [bold]cli kg query --query "MATCH (n) RETURN labels(n)[0], count(n)"[/bold]')
            console.print("   • [bold]cli kg query[/bold] (interactive shell)")

        @cli_app.command("export-html")
        def export_html(
            output_dir: Annotated[str, typer.Option("--output-dir", "-o", help="Output directory")] = "/tmp",
            open_browser: Annotated[bool, typer.Option("--open/--no-open", help="Open in browser")] = True,
            subgraphs: Annotated[
                list[str],
                typer.Option(
                    "--subgraph",
                    "-g",
                    help="Subgraph(s) to visualise; default is all registered",
                ),
            ] = [],
        ) -> None:
            """Export EKG graph visualization as HTML and display clickable link.

            Creates an interactive D3.js visualization of the EKG database
            and saves it to the specified output directory.
            """
            from genai_graph.core.graph_backend import create_backend_from_config
            from genai_graph.core.graph_html import generate_html_visualization

            console.print(Panel("[bold cyan]Exporting EKG HTML Visualization[/bold cyan]"))

            # Get database connection
            backend = create_backend_from_config(GRAPH_DB_CONFIG)
            if not backend:
                console.print("[red]❌ No EKG database found[/red]")
                console.print("[yellow]💡 Add data first: [bold]cli kg add --key <opportunity_key>[/bold][/yellow]")
                raise typer.Exit(1)

            console.print("[green]✅ Connected to EKG database[/green]")

            # Prepare output path
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            html_filename = "ekg_graph_visualization.html"
            html_file_path = output_path / html_filename

            console.print(f"📁 Output location: [bold]{html_file_path}[/bold]")

            # Generate HTML visualization
            console.print("🎨 Generating interactive visualization...")
            try:
                from genai_graph.core.graph_registry import GraphRegistry

                registry = GraphRegistry.get_instance()
                selected_subgraphs = subgraphs or registry.listsubgraphs()

                with console.status("[bold green]Creating HTML visualization..."):
                    # Build a combined schema to inform which node/relationship
                    # types should be visualised. The HTML generator will use
                    # this to filter node tables and relationships.
                    try:
                        schema = registry.build_combined_schema(selected_subgraphs)
                    except ValueError as e:
                        import traceback as tb

                        console.print(f"[red]❌ {e}[/red]")
                        console.print("[red]" + tb.format_exc() + "[/red]")
                        raise typer.Exit(1) from e

                    generate_html_visualization(
                        backend,
                        str(html_file_path),
                        title="EKG Database Visualization",
                        node_configs=schema.nodes,
                        relation_configs=schema.relations,
                    )

                console.print("[green]✅ HTML visualization created successfully[/green]")

                # Get file size
                file_size = html_file_path.stat().st_size
                file_size_mb = file_size / (1024 * 1024)

                # Display export summary
                export_table = Table(title="Export Summary")
                export_table.add_column("Property", style="cyan", no_wrap=True)
                export_table.add_column("Value", style="green")

                export_table.add_row("File Location", str(html_file_path))
                export_table.add_row("File Size", f"{file_size_mb:.2f} MB")
                export_table.add_row("Format", "Interactive HTML + D3.js")
                export_table.add_row("Features", "Zoomable, draggable, hover tooltips")

                console.print(export_table)

                # Create clickable link panel
                file_url = f"file://{html_file_path.absolute()}"
                console.print(
                    Panel(
                        f"[bold green]🌐 Clickable Link:[/bold green]\n\n"
                        f"[link={file_url}]{file_url}[/link]\n\n"
                        f"[dim]Click the link above or copy-paste into your browser[/dim]",
                        title="HTML Visualization Ready",
                        border_style="green",
                    )
                )

                # Optionally open in browser
                if open_browser:
                    try:
                        console.print("🌐 Opening in default browser...")
                        webbrowser.open(file_url)
                        console.print("[green]✅ Opened in browser[/green]")
                    except Exception as e:
                        import traceback as tb

                        console.print(f"[yellow]⚠️  Could not open browser automatically: {e}[/yellow]")
                        logger.debug(f"Browser open failed: {tb.format_exc()}")
                        console.print("[yellow]Please open the file manually using the link above[/yellow]")

            except Exception as e:
                import traceback as tb

                console.print(f"[red]❌ Error generating visualization: {e}[/red]")
                console.print("[red]" + tb.format_exc() + "[/red]")
                raise typer.Exit(1) from e

        @cli_app.command("list-subgraphs")
        def listsubgraphs() -> None:
            """List all registered knowledge graph subgraphs."""
            from genai_graph.core.graph_registry import GraphRegistry

            registry = GraphRegistry.get_instance()
            names = registry.listsubgraphs()

            if not names:
                console.print("[yellow]No subgraphs are currently registered.[/yellow]")
                return

            table = Table(title="Registered Subgraphs")
            table.add_column("Name", style="cyan", no_wrap=True)

            for name in names:
                table.add_row(name)

            console.print(table)

        @cli_app.command("schema")
        def show_schema(
            subgraphs: Annotated[
                list[str],
                typer.Option(
                    "--subgraph",
                    "-g",
                    help="Subgraph(s) to display schema for; default is all registered",
                ),
            ] = [],
        ) -> None:
            """Display knowledge graph schema in Markdown format for LLM context.

            Generates a comprehensive Markdown description of the graph schema including
            node types, relationships, properties, and indexed fields. This output is
            designed to provide context to LLMs for generating correct Cypher queries.
            """

            from genai_graph.core.graph_registry import GraphRegistry
            from genai_graph.core.schema_doc_generator import generate_schema_description

            console.print(Panel("[bold cyan]Knowledge Graph Schema[/bold cyan]"))

            try:
                registry = GraphRegistry.get_instance()
                selected_subgraphs = subgraphs or registry.listsubgraphs()

                # Use combined-schema documentation when multiple (or zero)
                # subgraphs are requested; for exactly one, keep the
                # single-subgraph format.
                if not selected_subgraphs or len(selected_subgraphs) > 1:
                    desc = generate_schema_description(selected_subgraphs)
                else:
                    desc = generate_schema_description(selected_subgraphs[0])
                console.print(desc)

            except ValueError as e:
                import traceback as tb

                console.print(f"[red]❌ {e}[/red]")
                console.print("[red]" + tb.format_exc() + "[/red]")
                raise typer.Exit(1) from e
