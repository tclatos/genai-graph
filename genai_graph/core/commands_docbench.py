"""CLI commands for Document Graph / DocBench browser (``cli docbench ...``).

Provides:
- ``web`` : launch the modern interactive Streamlit Document Graph explorer.
- ``cli`` : interactive TUI / outline browser in terminal.
- ``tui`` : launch the Textual interactive terminal browser.
- ``list`` : list ingested documents in the corpus.
- ``toc`` : show document table of contents outline.
- ``search`` : search sections across documents.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from genai_tk.cli.base import CliTopCommand
from rich.console import Console
from rich.table import Table

from genai_graph.core.commands_docgraph import _resolve_db_path

console = Console()


class DocBenchCommands(CliTopCommand):
    """Commands for browsing and evaluating Document Graphs (DocBench)."""

    description: str = "Document Graph browser and inspector (Streamlit webapp & CLI)"

    def get_description(self) -> tuple[str, str]:
        return "docbench", self.description

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        """Register ``docbench`` subcommands on the Typer application."""

        @cli_app.command("web")
        def web(
            db_path: Annotated[
                str | None,
                typer.Option(
                    "--db",
                    help="Path to the Ladybug database file. Uses docgraph_profiles.<profile>.paths.kg_db from config if omitted.",
                ),
            ] = None,
            profile: Annotated[
                str,
                typer.Option("--profile", "-p", help="DocGraph profile name (default: default)."),
            ] = "default",
            port: Annotated[
                int,
                typer.Option("--port", help="Port for the Streamlit web server."),
            ] = 8501,
            host: Annotated[
                str,
                typer.Option("--host", help="Host address for the Streamlit web server."),
            ] = "localhost",
            browser: Annotated[
                bool,
                typer.Option("--browser/--no-browser", help="Automatically open browser tab."),
            ] = True,
        ) -> None:
            """Launch the modern interactive Streamlit Document Graph explorer.

            Examples:
                cli docbench web
                cli docbench web --profile officeqa
                cli docbench web --db ./data/kg/docgraph.db --port 8502
            """
            resolved_db = _resolve_db_path(db_path, profile=profile) if db_path else None
            app_script = str(
                Path(__file__).resolve().parent.parent / "webapp" / "pages" / "demos" / "docgraph_browser.py"
            )

            cmd = [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                app_script,
                "--server.port",
                str(port),
                "--server.address",
                str(host),
            ]
            if not browser:
                cmd.extend(["--server.headless", "true"])

            env = os.environ.copy()
            if resolved_db:
                env["DOCGRAPH_DB_PATH"] = resolved_db
            if profile:
                env["DOCGRAPH_PROFILE"] = profile

            console.print(f"[bold green]Starting Document Graph Web Explorer on http://{host}:{port}...[/bold green]")
            subprocess.run(cmd, env=env)

        @cli_app.command("cli")
        def cli_browser(
            document: Annotated[
                str | None,
                typer.Argument(help="Document hash, filename, or folder (optional)."),
            ] = None,
            db_path: Annotated[
                str | None,
                typer.Option(
                    "--db",
                    help="Path to the Ladybug database file.",
                ),
            ] = None,
            profile: Annotated[
                str,
                typer.Option("--profile", "-p", help="DocGraph profile name (default: default)."),
            ] = "default",
            tui: Annotated[
                bool,
                typer.Option("--tui/--no-tui", "-t", help="Launch interactive Textual TUI browser."),
            ] = False,
        ) -> None:
            """CLI browser: show document listing / TOC or launch Textual TUI.

            Examples:
                cli docbench cli
                cli docbench cli --tui
                cli docbench cli guide.md
            """
            resolved_db = _resolve_db_path(db_path, profile=profile)

            if tui or document is None:
                from genai_graph.kg.query.document_graph_tui import run_document_graph_tui

                run_document_graph_tui(resolved_db)
                return

            from genai_graph.kg.backend import KuzuBackend
            from genai_graph.kg.query.document_graph_tools import get_document_toc, render_toc_outline

            backend = KuzuBackend()
            backend.connect(resolved_db)
            rows = get_document_toc(backend, document)
            if not rows:
                console.print(f"[yellow]No sections found for document: {document}[/yellow]")
                return
            console.print(render_toc_outline(rows))  # type: ignore[arg-type]

        @cli_app.command("tui")
        def tui(
            db_path: Annotated[
                str | None,
                typer.Option(
                    "--db",
                    help="Path to the Ladybug database file.",
                ),
            ] = None,
            profile: Annotated[
                str,
                typer.Option("--profile", "-p", help="DocGraph profile name (default: default)."),
            ] = "default",
        ) -> None:
            """Launch the interactive Textual TUI to browse the Document Graph."""
            resolved_db = _resolve_db_path(db_path, profile=profile)
            from genai_graph.kg.query.document_graph_tui import run_document_graph_tui

            run_document_graph_tui(resolved_db)

        @cli_app.command("list")
        def list_docs(
            folder: Annotated[
                str | None,
                typer.Option("--folder", help="Only show documents under this folder."),
            ] = None,
            db_path: Annotated[
                str | None,
                typer.Option(
                    "--db",
                    help="Path to the Ladybug database file.",
                ),
            ] = None,
            profile: Annotated[
                str,
                typer.Option("--profile", "-p", help="DocGraph profile name (default: default)."),
            ] = "default",
        ) -> None:
            """List ingested documents in the Document Graph."""
            resolved_db = _resolve_db_path(db_path, profile=profile)
            from genai_graph.kg.backend import KuzuBackend
            from genai_graph.kg.query.document_graph_tools import list_documents, resolve_folder_id

            backend = KuzuBackend()
            backend.connect(resolved_db)
            folder_id = resolve_folder_id(backend, folder) if folder else None
            rows = list_documents(backend, folder_id=folder_id)
            if not rows:
                console.print("[yellow]No documents ingested yet.[/yellow]")
                return

            table = Table(title="Documents in Document Graph")
            table.add_column("Filename", style="cyan")
            table.add_column("Sections", style="white")
            table.add_column("Tokens", style="dim")
            table.add_column("Description", style="green")
            table.add_column("Markdown Hash", style="dim")
            for r in rows:
                table.add_row(
                    str(r["filename"]),
                    str(r["section_count"]),
                    str(r.get("token_count") or 0),
                    str(r.get("description") or ""),
                    str(r["markdown_hash"]),
                )
            console.print(table)
