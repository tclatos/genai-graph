"""ETL Commands for vector store and document processing.

This module provides CLI commands for managing ETL operations, particularly
for CocoIndex vector store integration.
"""

from typing import Annotated

import typer
from genai_tk.main.cli import CliTopCommand
from genai_tk.utils.config_mngr import global_config
from rich.console import Console
from rich.table import Table

# Initialize Rich console
console = Console()


def get_etl_config_name(config_name: str | None) -> str:
    """Get the ETL config name to use, with proper fallback logic.

    Args:
        config_name: Explicitly provided config name from CLI option

    Returns:
        The config name to use (defaults to 'default' if not specified)
    """
    if config_name:
        return config_name
    return "default"


class EtlCommands(CliTopCommand):
    """Commands for ETL operations and vector store management."""

    def get_description(self) -> tuple[str, str]:
        return "etl", "ETL commands for vector store management."

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        @cli_app.command("update")
        def update(
            config_name: Annotated[
                str | None,
                typer.Option(
                    "--config",
                    help="Name of the ETL config to use from config/ekg.yaml (default: 'default')",
                ),
            ] = None,
        ):
            """Update the vector store with documents from configured sources.

            This command:
            1. Initializes the PostgreSQL database with pgvector extension
            2. Processes documents according to the configuration
            3. Generates embeddings for text chunks
            4. Stores embeddings in the vector database

            Example:
                cli etl update
                cli etl update --config custom
            """
            import cocoindex

            from genai_graph.etl.cocoindex_start import update_vector_store

            # Get configuration
            cfg_name = get_etl_config_name(config_name)

            try:
                etl_config = global_config().get_dict(f"etl_configs.{cfg_name}.cocoindex")
                # database_url is already resolved from the YAML config (e.g., ${paths.postgres})
                database_url = etl_config.get("database_url")
                if not database_url:
                    raise ValueError("database_url not found in ETL configuration")
            except Exception as e:
                console.print(f"[red]Error loading configuration:[/red] {e}")
                console.print(f"[yellow]Ensure 'etl_configs.{cfg_name}' exists in config/ekg.yaml[/yellow]")
                raise typer.Exit(1) from e

            cocoindex.init(cocoindex.Settings(database=cocoindex.DatabaseConnectionSpec(url=database_url)))

            console.print(f"[bold blue]Starting ETL update with config:[/bold blue] {cfg_name}")
            console.print(f"[dim]Source path:[/dim] {etl_config.get('source', {}).get('path', 'N/A')}")
            console.print(f"[dim]Database:[/dim] {database_url}")

            # Run the update
            result = update_vector_store(etl_config, database_url)

            if result.get("success"):
                console.print("[bold green]✓ Vector store update completed successfully![/bold green]")
                if "stats" in result:
                    console.print(f"[dim]Stats:[/dim] {result['stats']}")
            else:
                console.print(f"[bold red]✗ Update failed:[/bold red] {result.get('error', 'Unknown error')}")
                raise typer.Exit(1) from None

        @cli_app.command("query")
        def query(
            text: Annotated[str, typer.Argument(help="Search query text")],
            config_name: Annotated[
                str | None,
                typer.Option(
                    "--config",
                    help="Name of the ETL config to use from config/ekg.yaml (default: 'default')",
                ),
            ] = None,
            top_k: Annotated[
                int | None,
                typer.Option(
                    "--top-k",
                    help="Number of results to return (overrides config)",
                ),
            ] = None,
            verbose: Annotated[
                bool,
                typer.Option(
                    "--verbose",
                    "-v",
                    help="Show full text content in results",
                ),
            ] = False,
        ):
            """Search the vector store for documents matching the query.

            This command:
            1. Converts the query text to embeddings
            2. Performs semantic search in the vector database
            3. Returns the most relevant document chunks

            Examples:
                cli etl query "What is machine learning?"
                cli etl query "Python best practices" --top-k 10
                cli etl query "API documentation" --verbose
            """
            import cocoindex

            from genai_graph.etl.cocoindex_start import search_vector_store

            # Get configuration
            cfg_name = get_etl_config_name(config_name)

            try:
                etl_config = global_config().get_dict(f"etl_configs.{cfg_name}.cocoindex")
                # database_url is already resolved from the YAML config (e.g., ${paths.postgres})
                database_url = etl_config.get("database_url")
                if not database_url:
                    raise ValueError("database_url not found in ETL configuration")
            except Exception as e:
                console.print(f"[red]Error loading configuration:[/red] {e}")
                console.print(f"[yellow]Ensure 'etl_configs.{cfg_name}' exists in config/ekg.yaml[/yellow]")
                raise typer.Exit(1) from e

            # Initialize cocoindex

            cocoindex.init(cocoindex.Settings(database=cocoindex.DatabaseConnectionSpec(url=database_url)))

            console.print(f"[bold blue]Searching for:[/bold blue] {text}")
            if top_k:
                console.print(f"[dim]Returning top {top_k} results[/dim]")

            try:
                # Perform search
                results = search_vector_store(text, etl_config, database_url, top_k)

                if not results:
                    console.print("[yellow]No results found.[/yellow]")
                    return

                # Display results in a table
                table = Table(title=f"Search Results ({len(results)} found)", show_header=True)
                table.add_column("Score", style="cyan", width=8)
                table.add_column("Filename", style="green", width=40)
                if verbose:
                    table.add_column("Content", style="white", width=80)
                else:
                    table.add_column("Preview", style="white", width=60)

                for result in results:
                    score = f"{result['score']:.3f}"
                    filename = result["filename"]

                    if verbose:
                        content = result["text"]
                    else:
                        content = result["text"][:200] + ("..." if len(result["text"]) > 200 else "")

                    table.add_row(score, filename, content)

                console.print(table)

            except Exception as e:
                console.print(f"[bold red]✗ Search failed:[/bold red] {e}")
                import traceback

                console.print("[dim]" + traceback.format_exc() + "[/dim]")
                raise typer.Exit(1) from e
