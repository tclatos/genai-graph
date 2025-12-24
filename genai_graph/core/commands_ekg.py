"""CLI commands for interacting with the Enterprise Knowledge Graph.

This module provides the ``kg`` top-level command (as configured via
``config/overrides.yaml``) and routes the ``create`` command through the
Prefect-based orchestration flow defined in ``genai_graph.orchestration``.
"""

from __future__ import annotations

from typing import Annotated

import os
import typer
from genai_tk.main.cli import CliTopCommand
from genai_tk.utils.config_mngr import global_config
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from genai_graph.orchestration.flows import create_kg_flow

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
                f"Path: {result.backend.db_path}",
            )

            if warnings:
                console.print(Panel.fit("[bold yellow]⚠️  Warnings[/bold yellow]", border_style="yellow"))
                for idx, warning in enumerate(warnings, 1):
                    console.print(f"  [yellow]{idx}.[/yellow] {warning}")
                console.print("")
            else:
                console.print("[green]✓ No warnings[/green]")

            if result.html_export and export_html:
                console.print(
                    f"[green]HTML export created at:[/green] [cyan]{result.html_export.output_path}[/cyan]",
                )

        # TODO: other commands (delete, export-html, info, query) can be
        # migrated to Prefect-based flows in a similar fashion if needed.
