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

from pathlib import Path
from typing import Annotated

import typer
from genai_tk.main.cli import CliTopCommand
from genai_tk.utils.config_mngr import global_config
from loguru import logger
from rich.console import Console

# Initialize Rich console (used by other commands)
console = Console()

# Configuration constants
KV_STORE_ID = "file"
GRAPH_DB_CONFIG = "default"


def get_kg_config_name(config_name: str | None) -> str:
    """Get the KG config name to use, with proper fallback logic.

    Sets the config in global_config if needed, and returns the effective config name.

    Args:
        config_name: Explicitly provided config name from CLI option

    Returns:
        The config name to use. Priority order:
        1. Explicitly provided config_name
        2. Value of 'default_kg_config' key from global config
        3. Hardcoded fallback 'default'
    """
    if config_name:
        # User explicitly provided a config, set it
        global_config().set("kg_config", config_name)
        return config_name

    # Get default from the new 'default_kg_config' key
    default_from_config = global_config().get("default_kg_config")
    if default_from_config:
        # Set it so GraphRegistry can pick it up
        global_config().set("kg_config", default_from_config)
        return default_from_config

    # Final fallback
    global_config().set("kg_config", "default")
    return "test1"


class EkgCommands(CliTopCommand):
    """Commands for interacting with a Knowledge Graph."""

    def get_description(self) -> tuple[str, str]:
        return "kg", "Knowledge Graph commands."

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        # Import concrete subgraph modules so they can register themselves.
        # This keeps the CLI generic while still enabling default subgraphs.

        @cli_app.command("delete")
        def delete_ekg(
            force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation prompts")] = False,
        ) -> None:
            """Delete the entire EKG database.

            Safely removes the shared database directory after confirmation.
            All opportunity data will be lost.
            """

            from rich.prompt import Confirm

            from genai_graph.core.graph_backend import (
                create_backend_from_config,
                delete_backend_storage_from_config,
            )

            # Get the KG config name
            kg_config_name = get_kg_config_name(config_name)

            try:
                # Try to get some basic stats
                backend = create_backend_from_config(GRAPH_DB_CONFIG, kg_config_name)
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
                delete_backend_storage_from_config("default", kg_config_name)
                console.print("[green]✅ EKG database deleted successfully[/green]")
                console.print(
                    "[green]You can now start fresh with [bold]cli kg add --key <opportunity_key>[/bold][/green]"
                )
            except Exception as e:
                import traceback as tb

                console.print(f"[red]❌ Error deleting database: {e}[/red]")
                console.print("[red]" + tb.format_exc() + "[/red]")
                raise typer.Exit(1) from e

        @cli_app.command("export-html")
        def export_html(
            config_name: Annotated[
                str | None,
                typer.Option(
                    "--config",
                    help="Name of the KG config to use from config/ekg.yaml (default: value of key 'kg_config')",
                ),
            ] = None,
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
            from rich.panel import Panel
            from rich.table import Table

            from genai_graph.core.graph_backend import create_backend_from_config
            from genai_graph.core.graph_html import generate_html_visualization
            from genai_graph.core.kg_manager import get_kg_manager

            console.print(Panel("[bold cyan]Exporting EKG HTML Visualization[/bold cyan]"))

            # Get the KG config name and KG manager
            kg_config_name = get_kg_config_name(config_name)
            manager = get_kg_manager()
            manager.activate(profile=kg_config_name)

            # Get database connection
            backend = create_backend_from_config(GRAPH_DB_CONFIG, kg_config_name)
            if not backend:
                console.print("[red]❌ No EKG database found[/red]")
                console.print("[yellow]💡 Add data first: [bold]cli kg add --key <opportunity_key>[/bold][/yellow]")
                raise typer.Exit(1)

            console.print("[green]✅ Connected to EKG database[/green]")

            # Use KgManager for output path if output_dir is default
            if output_dir == "/tmp":
                html_file_path = manager.get_html_export_path()
                console.print(f"📁 Using organized output: [bold]{html_file_path}[/bold]")
            else:
                # Use custom output directory
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                html_filename = "ekg_graph_visualization.html"
                html_file_path = output_path / html_filename
                console.print(f"📁 Custom output location: [bold]{html_file_path}[/bold]")

            # Generate HTML visualization
            console.print("🎨 Generating interactive visualization...")
            try:
                from genai_graph.core.graph_registry import GraphRegistry

                # Get the effective config name using centralized logic
                get_kg_config_name(config_name)

                # Create a fresh registry instance
                registry = GraphRegistry()
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
                        import webbrowser

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
