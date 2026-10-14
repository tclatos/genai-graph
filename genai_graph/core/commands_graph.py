"""CLI commands for interacting with and querying Graph databases (``cli graph ...``).

Provides:
- ``display`` : Generate an interactive HTML graph visualization (D3.js force-directed layout)
                and open it in the default web browser.
- ``query``   : Launch an interactive Textual TUI to test Cypher queries, explore
                schema tables, and run generic examples, or execute in batch mode.
"""

from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import Annotated

import typer
from genai_tk.main.cli import CliTopCommand
from rich.console import Console
from rich.table import Table

from genai_graph.kg.backend import KuzuBackend

console = Console()


def resolve_graph_db_path(db_path: str | None = None) -> str:
    """Resolve database path from parameter, graph_db.default, docgraph_profiles, or paths.kg_db.

    Args:
        db_path: Explicit database path. If provided, use it directly.

    Returns:
        Resolved filesystem path to the graph database.

    Raises:
        typer.Exit: If no path is provided and no default can be resolved from configuration.
    """
    if db_path:
        return db_path

    # 1. Try global_config()["graph_db"]["default"]
    try:
        from genai_tk.config_mgmt.config_mngr import global_config
        from genai_tk.config_mgmt.file_patterns import resolve_config_path

        cfg = global_config()
        gdb = cfg.get("graph_db.default") or cfg.get("graph_db", {}).get("default")
        if gdb:
            resolved = resolve_config_path(str(gdb))
            if resolved and Path(resolved).exists():
                return resolved
    except Exception:
        pass

    # 2. Try docgraph_profiles.default lookup via find_docgraph_db_path
    try:
        from genai_graph.agent.docgraph_agent import find_docgraph_db_path

        resolved = find_docgraph_db_path()
        if resolved and Path(resolved).exists():
            return resolved
    except Exception:
        pass

    # 3. Fallback to graph_db.default even if file does not exist yet
    try:
        from genai_tk.config_mgmt.config_mngr import global_config
        from genai_tk.config_mgmt.file_patterns import resolve_config_path

        cfg = global_config()
        gdb = cfg.get("graph_db.default") or cfg.get("graph_db", {}).get("default")
        if gdb:
            resolved = resolve_config_path(str(gdb))
            if resolved:
                return resolved
    except Exception:
        pass

    console.print(
        "[red]Error: No database path provided and could not resolve from config ('graph_db.default' or 'docgraph_profiles.default.paths.kg_db').[/red]\n"
        "  Use [bold cyan]--db <path>[/bold cyan] or configure [bold cyan]graph_db.default[/bold cyan] in your project YAML."
    )
    raise typer.Exit(1)


class GraphCommands(CliTopCommand):
    """Commands for inspecting, visualizing, and querying Graph databases."""

    def get_description(self) -> tuple[str, str]:  # type: ignore[override]
        return "graph", "Graph database visualization and interactive Cypher query commands."

    def register_sub_commands(self, cli_app: typer.Typer) -> None:  # type: ignore[override]
        """Register ``graph`` subcommands on the given Typer application."""

        @cli_app.command("display")
        def display(
            db_path: Annotated[
                str | None,
                typer.Option(
                    "--db",
                    "-d",
                    help="Path to Ladybug/Kùzu database directory. Defaults to graph_db.default.",
                ),
            ] = None,
            output: Annotated[
                str | None,
                typer.Option(
                    "--output",
                    "-o",
                    help="Destination path for generated HTML file. Defaults to ./<db_stem>_graph.html.",
                ),
            ] = None,
            query: Annotated[
                str,
                typer.Option(
                    "--query",
                    "-q",
                    help="Cypher query to fetch nodes and relationships for visualization.",
                ),
            ] = "MATCH (n)-[r]->(m) RETURN n, r, m",
            limit: Annotated[
                int,
                typer.Option(
                    "--limit",
                    "-l",
                    help="Maximum number of relationships to display (appended to query if LIMIT not specified).",
                ),
            ] = 1000,
            browser: Annotated[
                bool,
                typer.Option(
                    "--browser/--no-browser",
                    help="Automatically open generated HTML file in default web browser.",
                ),
            ] = True,
        ) -> None:
            """Create an interactive HTML graph visualization and open it in the browser."""
            from genai_graph.kg.export.html import generate_html

            resolved_db = resolve_graph_db_path(db_path)
            db_file = Path(resolved_db)
            if not db_file.exists():
                console.print(f"[red]Error: Graph database not found at:[/red] {resolved_db}")
                raise typer.Exit(1)

            # Determine HTML destination path
            dest_path = Path(output).resolve() if output else Path.cwd() / f"{db_file.stem}_graph.html"
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Ensure LIMIT is respected
            effective_query = query.strip()
            if "LIMIT" not in effective_query.upper():
                effective_query = f"{effective_query} LIMIT {limit}"

            console.print(f"[dim]Connecting to database: {resolved_db}[/dim]")
            console.print(f"[dim]Visualization query: {effective_query}[/dim]")

            backend = KuzuBackend()
            backend.connect(resolved_db, read_only=True)

            try:
                generate_html(
                    connection=backend,
                    destination_file_path=str(dest_path),
                    query=effective_query,
                )
            except Exception as exc:
                console.print(f"[red]Failed to generate HTML graph visualization: {exc}[/red]")
                raise typer.Exit(1) from exc

            console.print(f"[bold green]✓ HTML graph visualization created:[/bold green] {dest_path}")

            if browser:
                file_url = dest_path.resolve().as_uri()
                console.print(f"[dim]🌐 Opening in browser: {file_url}[/dim]")
                webbrowser.open(file_url)

        @cli_app.command("query")
        def query(
            db_path: Annotated[
                str | None,
                typer.Option(
                    "--db",
                    "-d",
                    help="Path to Ladybug/Kùzu database directory. Defaults to graph_db.default.",
                ),
            ] = None,
            query_str: Annotated[
                str | None,
                typer.Option(
                    "--query",
                    "-q",
                    help="Cypher query to execute directly or pre-load in the TUI.",
                ),
            ] = None,
            batch: Annotated[
                bool,
                typer.Option(
                    "--batch",
                    "-b",
                    help="Execute query non-interactively and display results in terminal table.",
                ),
            ] = False,
            read_only: Annotated[
                bool,
                typer.Option(
                    "--read-only/--write",
                    help="Open database in read-only mode (default: True).",
                ),
            ] = True,
        ) -> None:
            """Enter an interactive TUI to test Cypher queries with predefined generic examples.

            Use --batch with -q to run a query non-interactively from the command line.

            Examples:
                cli graph query
                cli graph query --db data/kg/rfq_tree.db
                cli graph query -q "CALL show_tables() RETURN *" --batch
            """
            resolved_db = resolve_graph_db_path(db_path)
            db_file = Path(resolved_db)
            if not db_file.exists():
                console.print(f"[red]Error: Graph database not found at:[/red] {resolved_db}")
                raise typer.Exit(1)

            if batch:
                if not query_str:
                    console.print("[red]Error: --query is required when using --batch mode.[/red]")
                    raise typer.Exit(1)

                backend = KuzuBackend()
                backend.connect(resolved_db, read_only=read_only)
                try:
                    df = backend.execute_get_as_df(query_str, union=True)
                    if df.empty:
                        console.print("[yellow]Query returned no results[/yellow]")
                        return

                    table = Table(title=f"Query Results ({len(df)} rows)")
                    for col in df.columns:
                        table.add_column(str(col), style="cyan")

                    max_rows = 50
                    for i, (_, row) in enumerate(df.iterrows()):
                        if i >= max_rows:
                            table.add_row(*["..." for _ in df.columns])
                            break
                        table.add_row(*[str(val) for val in row])

                    console.print(table)
                    if len(df) > max_rows:
                        console.print(f"[dim]Showing first {max_rows} of {len(df)} results[/dim]")
                except Exception as exc:
                    console.print(f"[red]Query error: {exc}[/red]")
                    raise typer.Exit(1) from exc
                return

            # Interactive TUI mode
            from genai_graph.kg.query.cypher_tui import run_cypher_tui

            run_cypher_tui(db_path=resolved_db, initial_query=query_str, read_only=read_only)
