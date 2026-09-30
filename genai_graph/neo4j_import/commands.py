"""CLI commands for Neo4j to Ladybug import.

This module provides the ``neo4j`` top-level command for importing Neo4j JSONL exports
into Ladybug graph database (Kuzu-compatible, maintained fork).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from genai_tk.config_mgmt.config_mngr import global_config
from genai_tk.main.cli import CliTopCommand
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class Neo4jCommands(CliTopCommand):
    """Commands for importing Neo4j exports into Ladybug (Kuzu-compatible)."""

    def get_description(self) -> tuple[str, str]:
        return "neo4j", "Neo4j to Ladybug import commands (Ladybug is a maintained fork of Kuzu)."

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        """Register ``neo4j`` subcommands on the given Typer application."""

        @cli_app.command("analyze")
        def analyze(
            jsonl_file: Annotated[
                str,
                typer.Argument(help="Path to Neo4j JSONL export file."),
            ],
            output_schema: Annotated[
                str | None,
                typer.Option(
                    "--output",
                    "-o",
                    help="Output file for Ladybug schema statements.",
                ),
            ] = None,
            show_summary: Annotated[
                bool,
                typer.Option(
                    "--summary/--no-summary",
                    help="Show detailed summary of the schema.",
                ),
            ] = True,
        ) -> None:
            """Analyze a Neo4j JSONL export and generate Ladybug schema.

            Reads the JSONL file, extracts all node labels, relationship types,
            and their properties, then generates CREATE NODE TABLE and CREATE REL TABLE
            statements for Ladybug (Kuzu-compatible).

            Examples:
                ```bash
                # Analyze a JSONL file
                cli neo4j analyze ./export.jsonl

                # Analyze with schema output
                cli neo4j analyze ./export.jsonl -o schema.cypher

                # Use config variables
                cli neo4j analyze '${paths.stratnav_db}/somedb.jsonl' -o schema.cypher
                ```
            """
            from genai_tk.config_mgmt.file_patterns import resolve_config_path

            from genai_graph.neo4j_import.schema_analyzer import SchemaAnalyzer

            # Resolve YAML config variables in paths
            resolved_jsonl_file = resolve_config_path(jsonl_file)
            jsonl_path = Path(resolved_jsonl_file)

            if not jsonl_path.exists():
                console.print(
                    f"[red]Error: File not found: {jsonl_file}"
                    + (f"\n(Resolved to: {resolved_jsonl_file})" if resolved_jsonl_file != jsonl_file else "")
                    + "[/red]"
                )
                raise typer.Exit(1)

            console.print(f"[bold]Analyzing:[/bold] {jsonl_file}")

            analyzer = SchemaAnalyzer(jsonl_path)
            analyzer.analyze()

            if show_summary:
                analyzer.print_summary()

            # Generate schema statements
            statements = analyzer.generate_kuzu_schema()

            if output_schema:
                resolved_output_schema = resolve_config_path(output_schema)
                output_path = Path(resolved_output_schema)
                with output_path.open("w", encoding="utf-8") as f:
                    f.write("-- Kuzu schema generated from Neo4j JSONL export\n")
                    f.write(f"-- Source: {jsonl_file}\n\n")
                    for stmt in statements:
                        f.write(stmt + "\n\n")
                console.print(f"[green]✓ Schema written to: {output_schema}[/green]")
            else:
                console.print("\n[bold]Generated Kuzu Schema:[/bold]\n")
                for stmt in statements:
                    console.print(stmt)
                    console.print()

        @cli_app.command("convert")
        def convert(
            jsonl_file: Annotated[
                str,
                typer.Argument(help="Path to Neo4j JSONL export file."),
            ],
            output_dir: Annotated[
                str,
                typer.Argument(help="Output directory for JSON files."),
            ],
        ) -> None:
            """Convert Neo4j JSONL export to Ladybug-compatible JSON files.

            Creates separate JSON files for each node label and relationship type
            that can be imported into Ladybug using COPY FROM statements.

            Examples:
                ```bash
                # Convert a JSONL file
                cli neo4j convert ./export.jsonl ./output

                # Use config variables
                cli neo4j convert '${paths.stratnav_db}/somedb.jsonl' '${paths.data_root}/ladybug_import'
                ```
            """
            from genai_tk.config_mgmt.file_patterns import resolve_config_path

            from genai_graph.neo4j_import.converter import Neo4jToKuzuConverter

            # Resolve YAML config variables in paths
            resolved_jsonl_file = resolve_config_path(jsonl_file)
            jsonl_path = Path(resolved_jsonl_file)

            resolved_output_dir = resolve_config_path(output_dir)
            output_path = Path(resolved_output_dir)

            if not jsonl_path.exists():
                console.print(
                    f"[red]Error: File not found: {jsonl_file}"
                    + (f"\n(Resolved to: {resolved_jsonl_file})" if resolved_jsonl_file != jsonl_file else "")
                    + "[/red]"
                )
                raise typer.Exit(1)

            console.print(f"[bold]Converting:[/bold] {jsonl_file}")
            console.print(f"[bold]Output:[/bold] {output_dir}")
            if resolved_output_dir != output_dir:
                console.print(f"[dim](Resolved to: {resolved_output_dir})[/dim]")

            converter = Neo4jToKuzuConverter(jsonl_path)
            stats = converter.convert(output_path)

            console.print("\n[green]✓ Conversion complete[/green]")

            table = Table(title="Conversion Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Nodes processed", f"{stats.nodes_processed:,}")
            table.add_row("Relationships processed", f"{stats.relationships_processed:,}")
            table.add_row("Node files created", str(len(stats.node_files_created)))
            table.add_row("Relationship files created", str(len(stats.rel_files_created)))

            console.print(table)

            if stats.errors:
                console.print(f"\n[yellow]⚠️  {len(stats.errors)} errors encountered[/yellow]")
                for error in stats.errors[:5]:
                    console.print(f"  - {error}")

        @cli_app.command("subset")
        def subset(
            jsonl_file: Annotated[
                str,
                typer.Argument(help="Path to Neo4j JSONL export file."),
            ],
            output_file: Annotated[
                str,
                typer.Argument(help="Output path for the subset JSONL file."),
            ],
            max_nodes: Annotated[
                int,
                typer.Option(
                    "--max-nodes",
                    "-n",
                    help="Maximum nodes per label.",
                ),
            ] = 10,
            max_rels: Annotated[
                int,
                typer.Option(
                    "--max-rels",
                    "-r",
                    help="Maximum relationships per type.",
                ),
            ] = 20,
            anonymize: Annotated[
                bool,
                typer.Option(
                    "--anonymize/--no-anonymize",
                    help="Anonymize string properties with fake data.",
                ),
            ] = False,
            seed: Annotated[
                int | None,
                typer.Option(
                    "--seed",
                    "-s",
                    help="Random seed for reproducibility.",
                ),
            ] = None,
        ) -> None:
            """Create a subset of a Neo4j JSONL export for testing.

            Extracts a small sample of nodes and relationships, optionally
            anonymizing the data with fake values.

            Examples:
                ```bash
                # Create a subset with defaults
                cli neo4j subset ./export.jsonl ./subset.jsonl

                # Create small subset with anonymization
                cli neo4j subset ./export.jsonl ./subset.jsonl -n 5 -r 10 --anonymize

                # Use config variables
                cli neo4j subset '${paths.stratnav_db}/db.jsonl' '${paths.data_root}/subset.jsonl' -n 100
                ```
            """
            from genai_tk.config_mgmt.file_patterns import resolve_config_path

            from genai_graph.neo4j_import.converter import SubsetCreator

            # Resolve YAML config variables in paths
            resolved_jsonl_file = resolve_config_path(jsonl_file)
            jsonl_path = Path(resolved_jsonl_file)

            resolved_output_file = resolve_config_path(output_file)
            output_path = Path(resolved_output_file)

            if not jsonl_path.exists():
                console.print(
                    f"[red]Error: File not found: {jsonl_file}"
                    + (f"\n(Resolved to: {resolved_jsonl_file})" if resolved_jsonl_file != jsonl_file else "")
                    + "[/red]"
                )
                raise typer.Exit(1)

            console.print(f"[bold]Creating subset from:[/bold] {jsonl_file}")
            console.print(f"[bold]Output:[/bold] {output_file}")
            if resolved_output_file != output_file:
                console.print(f"[dim](Resolved to: {resolved_output_file})[/dim]")
            console.print(f"[dim]Max nodes per label: {max_nodes}[/dim]")
            console.print(f"[dim]Max rels per type: {max_rels}[/dim]")
            console.print(f"[dim]Anonymize: {anonymize}[/dim]")

            creator = SubsetCreator(jsonl_path)
            stats = creator.create_subset(
                output_path,
                max_nodes_per_label=max_nodes,
                max_rels_per_type=max_rels,
                anonymize=anonymize,
                seed=seed,
            )

            console.print("\n[green]✓ Subset created[/green]")

            table = Table(title="Subset Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Nodes included", f"{stats['nodes']:,}")
            table.add_row("Relationships included", f"{stats['relationships']:,}")
            table.add_row("Node labels", str(stats["node_labels"]))
            table.add_row("Relationship types", str(stats["rel_types"]))

            console.print(table)

        @cli_app.command("import")
        def import_to_kuzu(
            jsonl_file: Annotated[
                str,
                typer.Argument(help="Path to Neo4j JSONL export file."),
            ],
            db_path: Annotated[
                str | None,
                typer.Option(
                    "--db",
                    "-d",
                    help="Path for Kuzu database. Defaults to config path.",
                ),
            ] = None,
            json_dir: Annotated[
                str | None,
                typer.Option(
                    "--json-dir",
                    "-j",
                    help="Directory for intermediate JSON files.",
                ),
            ] = None,
            force: Annotated[
                bool,
                typer.Option(
                    "--force",
                    "-f",
                    help="Delete existing database before import.",
                ),
            ] = False,
        ) -> None:
            """Import a Neo4j JSONL export into a Kuzu database.

            This is the full pipeline that:
            1. Analyzes the JSONL to extract schema
            2. Converts to Kuzu-compatible JSON files
            3. Creates the Kuzu database with schema
            4. Imports all nodes and relationships

            Examples:
                ```bash
                # Import with default database path
                cli neo4j import ./export.jsonl

                # Import with custom database path
                cli neo4j import ./export.jsonl --db ./my_kuzu_db

                # Use config variables
                cli neo4j import '${paths.stratnav_db}/db.jsonl' --db '${paths.data_root}/kuzu_db'
                ```
            """
            from genai_tk.config_mgmt.file_patterns import resolve_config_path

            from genai_graph.neo4j_import.kuzu_manager import import_neo4j_to_kuzu

            # Resolve YAML config variables in paths
            resolved_jsonl_file = resolve_config_path(jsonl_file)
            jsonl_path = Path(resolved_jsonl_file)

            if not jsonl_path.exists():
                console.print(
                    f"[red]Error: File not found: {jsonl_file}"
                    + (f"\n(Resolved to: {resolved_jsonl_file})" if resolved_jsonl_file != jsonl_file else "")
                    + "[/red]"
                )
                raise typer.Exit(1)

            # Get default db path from config if not specified
            if db_path is None:
                # Try graph_db.default first, then fall back to neo4j_import path for backward compatibility
                db_path = global_config().get("graph_db.default", None)
                if db_path is None:
                    db_path = str(global_config().get_dir_path("paths.data_root") / "neo4j_import" / "kuzu_db")

            resolved_db_path = resolve_config_path(db_path)
            db_path_obj = Path(resolved_db_path)

            # Resolve json_dir if specified
            json_dir_obj = None
            if json_dir:
                resolved_json_dir = resolve_config_path(json_dir)
                json_dir_obj = Path(resolved_json_dir)

            console.print(
                Panel.fit(
                    f"[bold]Neo4j to Kuzu Import[/bold]\n\n"
                    f"Source: {jsonl_file}\n"
                    f"Database: {db_path}\n"
                    f"Delete existing: {force}",
                    border_style="blue",
                )
            )

            try:
                result = import_neo4j_to_kuzu(
                    jsonl_path=jsonl_path,
                    db_path=db_path_obj,
                    json_output_dir=json_dir_obj,
                    delete_existing=force,
                )

                console.print("\n[green]✓ Import complete![/green]\n")

                # Summary table
                table = Table(title="Import Summary")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")

                table.add_row("Database path", str(result["db_path"]))
                table.add_row("Node tables", str(len(result["schema_info"]["node_tables"])))
                table.add_row("Relationship tables", str(len(result["schema_info"]["rel_tables"])))
                table.add_row("Total nodes", f"{result['database']['total_nodes']:,}")
                table.add_row("Total relationships", f"{result['database']['total_relationships']:,}")

                console.print(table)

                if result["import"]["errors"]:
                    console.print(f"\n[yellow]⚠️  {len(result['import']['errors'])} errors[/yellow]")
                    for error in result["import"]["errors"][:5]:
                        console.print(f"  - {error}")

            except Exception as e:
                console.print(f"[red]Error during import: {e}[/red]")
                logger.exception("Import failed")
                raise typer.Exit(1) from e

        @cli_app.command("query")
        def query(
            cypher: Annotated[
                str,
                typer.Argument(help="Cypher query to execute."),
            ],
            db_path: Annotated[
                str | None,
                typer.Option(
                    "--db",
                    "-d",
                    help="Path to Kuzu database.",
                ),
            ] = None,
            limit: Annotated[
                int,
                typer.Option(
                    "--limit",
                    "-l",
                    help="Maximum rows to display.",
                ),
            ] = 20,
        ) -> None:
            """Run a Cypher query against the imported Kuzu database.

            Examples:
                ```bash
                # Query with default database
                cli neo4j query "MATCH (n) RETURN n LIMIT 10"

                # Query with custom database path
                cli neo4j query "MATCH (n) RETURN count(n)" --db ./my_kuzu_db

                # Use config variables
                cli neo4j query "MATCH (n) RETURN n" --db '${paths.data_root}/kuzu_db'
                ```
            """
            from genai_tk.config_mgmt.file_patterns import resolve_config_path

            from genai_graph.neo4j_import.kuzu_manager import KuzuImporter

            # Get default db path from config if not specified
            if db_path is None:
                # Try graph_db.default first, then fall back to neo4j_import path for backward compatibility
                db_path = global_config().get("graph_db.default", None)
                if db_path is None:
                    db_path = str(global_config().get_dir_path("paths.data_root") / "neo4j_import" / "kuzu_db")

            resolved_db_path = resolve_config_path(db_path)
            db_path_obj = Path(resolved_db_path)

            if not db_path_obj.exists():
                console.print(
                    f"[red]Error: Database not found: {db_path}"
                    + (f"\n(Resolved to: {resolved_db_path})" if resolved_db_path != db_path else "")
                    + "[/red]"
                )
                raise typer.Exit(1)

            importer = KuzuImporter(db_path_obj)
            importer.create_database(delete_existing=False)

            try:
                results = importer.run_query(cypher)

                if not results:
                    console.print("[dim]No results[/dim]")
                    return

                # Display as table
                table = Table(title=f"Query Results (showing up to {limit})")

                # Add columns
                for col in results[0]:
                    table.add_column(col, style="cyan")

                # Add rows
                for row in results[:limit]:
                    table.add_row(*[str(v)[:100] for v in row.values()])

                console.print(table)

                if len(results) > limit:
                    console.print(f"[dim]... and {len(results) - limit} more rows[/dim]")

            except Exception as e:
                console.print(f"[red]Query error: {e}[/red]")
                raise typer.Exit(1) from e

            finally:
                importer.close()

        @cli_app.command("info")
        def info(
            db_path: Annotated[
                str | None,
                typer.Option(
                    "--db",
                    "-d",
                    help="Path to Kuzu database.",
                ),
            ] = None,
        ) -> None:
            """Show information about an imported Kuzu database.

            Examples:
                ```bash
                # Show info for default database
                cli neo4j info

                # Show info for custom database path
                cli neo4j info --db ./my_kuzu_db

                # Use config variables
                cli neo4j info --db '${paths.data_root}/kuzu_db'
                ```
            """
            from genai_tk.config_mgmt.file_patterns import resolve_config_path

            from genai_graph.neo4j_import.kuzu_manager import KuzuImporter

            # Get default db path from config if not specified
            if db_path is None:
                # Try graph_db.default first, then fall back to neo4j_import path for backward compatibility
                db_path = global_config().get("graph_db.default", None)
                if db_path is None:
                    db_path = str(global_config().get_dir_path("paths.data_root") / "neo4j_import" / "kuzu_db")

            resolved_db_path = resolve_config_path(db_path)
            db_path_obj = Path(resolved_db_path)

            if not db_path_obj.exists():
                console.print(
                    f"[red]Error: Database not found: {db_path}"
                    + (f"\n(Resolved to: {resolved_db_path})" if resolved_db_path != db_path else "")
                    + "[/red]"
                )
                raise typer.Exit(1)

            importer = KuzuImporter(db_path_obj)
            importer.create_database(delete_existing=False)

            try:
                stats = importer.get_stats()

                console.print(
                    Panel.fit(
                        f"[bold]Kuzu Database Info[/bold]\n\nPath: {db_path}",
                        border_style="blue",
                    )
                )

                # Node tables
                if stats["node_tables"]:
                    table = Table(title="Node Tables")
                    table.add_column("Table", style="cyan")
                    table.add_column("Count", style="green", justify="right")

                    for name, count in sorted(stats["node_tables"].items()):
                        table.add_row(name, f"{count:,}")

                    table.add_row("[bold]Total[/bold]", f"[bold]{stats['total_nodes']:,}[/bold]")
                    console.print(table)

                # Relationship tables
                if stats["rel_tables"]:
                    table = Table(title="Relationship Tables")
                    table.add_column("Table", style="cyan")
                    table.add_column("Count", style="green", justify="right")

                    for name, count in sorted(stats["rel_tables"].items()):
                        table.add_row(name, f"{count:,}")

                    table.add_row(
                        "[bold]Total[/bold]",
                        f"[bold]{stats['total_relationships']:,}[/bold]",
                    )
                    console.print(table)

            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                raise typer.Exit(1) from e

            finally:
                importer.close()
