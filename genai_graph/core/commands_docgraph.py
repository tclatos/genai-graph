"""CLI commands for the Document Graph (``cli docgraph ...``).

Provides:
- ``run``    : execute a document-graph workflow profile (markdownize sources,
               then run the configured sub-graph factories) via the workflow
               engine. Sources can be overridden ad-hoc with ``-s``.
- ``build``  : quick document-graph-only build (markdownize sources, then ingest
               the Folder → Document → Section structure) directly on a Ladybug DB.
- ``delete`` : delete all documents, folders, and sections from the graph.
- ``list`` / ``toc`` / ``cat`` / ``search`` / ``tui`` : navigate an ingested graph.

``run`` and ``kg create`` share the same workflow engine; ``run`` targets ad-hoc
sources while ``kg create`` targets a predefined, named set of documents.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Annotated, Any

import typer
from genai_tk.config_mgmt.config_mngr import global_config
from genai_tk.main.cli import CliTopCommand
from genai_tk.workflow.force import ForceStage
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

console = Console()


def _resolve_db_path(db_path: str | None = None) -> str:
    """Resolve database path from parameter or config default.

    Args:
        db_path: Explicit database path. If provided, use it.

    Returns:
        Resolved database path.

    Raises:
        typer.Exit: If no path provided and no config default found.
    """
    if db_path:
        return db_path

    # Try to get default from config
    default_db = global_config().get("graph_db.default", None)
    if default_db:
        return str(default_db)

    console.print(
        "[red]Error: No database path provided and no graph_db.default configured.[/red]\n"
        "  Use --db <path> or add graph_db.default to your config file."
    )
    raise typer.Exit(1)


def _validate_force(force: str | None) -> None:
    if force is None:
        return
    try:
        ForceStage(force)
    except ValueError as exc:
        stages = ", ".join(s.value for s in ForceStage)
        console.print(f"[red]Invalid --force stage '{force}'. Choose one of: {stages}[/red]")
        raise typer.Exit(1) from exc


def _resolve_folder_ref_or_exit(backend: Any, folder_ref: str | None) -> str | None:
    """Resolve a `--folder` option value to a `folder_id`, or exit with an error message."""
    if folder_ref is None:
        return None
    from genai_graph.kg.query.document_graph_tools import resolve_folder_id

    folder_id = resolve_folder_id(backend, folder_ref)
    if folder_id is None:
        console.print(f"[red]No folder found matching: {folder_ref}[/red]")
        raise typer.Exit(1)
    return folder_id


def _resolve_doc_ref_or_exit(backend: Any, doc_ref: str | None) -> str | None:
    """Resolve a `--doc` option value to a `markdown_hash`, or exit with an error message."""
    if doc_ref is None:
        return None
    from genai_graph.kg.query.document_graph_tools import resolve_document_id

    doc_id = resolve_document_id(backend, doc_ref)
    if doc_id is None:
        console.print(f"[red]No document found matching: {doc_ref}[/red]")
        raise typer.Exit(1)
    return doc_id


class DocGraphCommands(CliTopCommand):
    """Commands for building and navigating a Document Graph."""

    def get_description(self) -> tuple[str, str]:  # type: ignore[override]
        return "docgraph", "Document Graph commands."

    def register_sub_commands(self, cli_app: typer.Typer) -> None:  # type: ignore[override]
        """Register ``docgraph`` subcommands on the given Typer application."""

        @cli_app.command("run")
        def run(
            workflow: Annotated[
                str,
                typer.Option("--workflow", "-w", help="Document-graph workflow profile (e.g. 'rainbow_extract')."),
            ],
            source: Annotated[
                list[str] | None,
                typer.Option("--source", "-s", help="Ad-hoc source file(s)/dir(s)/zip(s); overrides profile sources."),
            ] = None,
            dry_run: Annotated[
                bool,
                typer.Option("--dry-run", help="Resolve the workflow plan without executing."),
            ] = False,
            force: Annotated[
                str | None,
                typer.Option("--force", help="Force-invalidate caches from this stage onward."),
            ] = None,
            delete_first: Annotated[
                bool,
                typer.Option("--delete-first/--no-delete-first", help="Delete existing graph before creation."),
            ] = False,
            export_html: Annotated[
                bool,
                typer.Option("--export-html/--no-export-html", help="Export HTML visualization after creation."),
            ] = True,
            set_values: Annotated[
                list[str] | None,
                typer.Option("--set", help="Override profile values as KEY=VALUE.", metavar="KEY=VALUE"),
            ] = None,
        ) -> None:
            """Run a document-graph workflow over ad-hoc or configured sources.

            Examples:
                cli docgraph run --workflow rainbow_extract -s "04...VENUS...pptx"
                cli docgraph run -w rainbow_extract -s ./ppt --force md
            """
            _validate_force(force)

            from genai_tk.workflow.executor import execute_workflow
            from genai_tk.workflow.resolver import (
                WorkflowResolutionError,
                parse_cli_overrides,
                resolve_workflow_invocation,
            )

            cli_overrides: dict[str, Any] = parse_cli_overrides(set_values) if set_values else {}
            cli_overrides.setdefault("force_stage", force)
            cli_overrides.setdefault("delete_first", delete_first)
            cli_overrides.setdefault("export_html", export_html)
            if source:
                cli_overrides["sources"] = list(source)

            try:
                invocation = resolve_workflow_invocation(workflow, cli_overrides=cli_overrides)
            except WorkflowResolutionError as exc:
                console.print(Panel(str(exc), title=f"Resolution Error: {workflow}", border_style="red"))
                raise typer.Exit(1) from exc

            _render_plan(invocation)
            if dry_run:
                console.print(Panel("Dry run complete — no execution performed.", border_style="green"))
                return

            try:
                results = execute_workflow(invocation)
            except Exception as exc:
                logger.debug("docgraph run error for {}: {}", workflow, exc, exc_info=True)
                console.print(Panel(str(exc), title=f"docgraph run failed: {workflow}", border_style="red"))
                raise typer.Exit(1) from exc
            console.print(f"[green]✓ {workflow}: workflow completed ({len(results)} step(s))[/green]")

        @cli_app.command("build")
        def build(
            source: Annotated[
                list[str],
                typer.Argument(help="Directories, files, or .zip archives to ingest (raw docs or Markdown)."),
            ],
            db_path: Annotated[
                str | None,
                typer.Option(
                    "--db", help="Path to the Ladybug database file. Uses graph_db.default from config if omitted."
                ),
            ] = None,
            md_output_dir: Annotated[
                str | None,
                typer.Option(
                    "--md-output-dir",
                    help="Where converted Markdown is written. Defaults to '<db_path stem>_markdown'.",
                ),
            ] = None,
            cache_dir: Annotated[
                str | None,
                typer.Option("--cache-dir", help="Intermediates directory (unzipped/pdf/manifest)."),
            ] = None,
            profile: Annotated[
                str,
                typer.Option("--profile", help="markdownize profile: fast, medium, best, or default."),
            ] = "default",
            include: Annotated[
                list[str] | None,
                typer.Option("--include", help="Glob pattern(s) to include (default '*.md')."),
            ] = None,
            exclude: Annotated[
                list[str] | None,
                typer.Option("--exclude", help="Glob pattern(s) to exclude."),
            ] = None,
            force: Annotated[
                str | None,
                typer.Option("--force", help="Force-invalidate caches from this stage onward."),
            ] = None,
            delete_first: Annotated[
                bool,
                typer.Option("--delete-first", help="Drop existing Section tables and rebuild all sections."),
            ] = False,
            llm: Annotated[
                str | None,
                typer.Option(
                    "--llm",
                    help="LLM id (name@provider) or config tag (e.g. default/flash) enabling the LLM build path: "
                    "a flash model discovers each document's structure and summarizes its sections in one call. "
                    "Omit for the fast algorithmic-only path. See kg_build.llms.* config tags.",
                ),
            ] = None,
            llm_max_tokens: Annotated[
                int | None,
                typer.Option(
                    "--llm-max-tokens",
                    help="Explicit max output tokens for the outline call; raise for reasoning models that "
                    "exhaust their completion budget ('length limit reached' errors).",
                ),
            ] = None,
            summary_min_tokens: Annotated[
                int,
                typer.Option(
                    "--summary-min-tokens", help="Prompt guidance for what counts as a 'substantial' section."
                ),
            ] = 800,
            outline_cache_dir: Annotated[
                str | None,
                typer.Option("--outline-cache-dir", help="Directory for the content-addressed outline JSON cache."),
            ] = None,
            workers: Annotated[int, typer.Option("--workers", help="Parallelism for the LLM outline pre-pass.")] = 4,
            context_safety_ratio: Annotated[
                float,
                typer.Option(
                    "--context-safety-ratio",
                    help="Degrade a document to algorithmic parsing (no LLM call, no summaries) when its token "
                    "count exceeds this fraction of the model's context window.",
                ),
            ] = 0.9,
        ) -> None:
            """Markdownize sources, then build (or update) a Document Graph.

            Without `--llm`, the build is algorithmic and fast (heading hierarchy
            only). With `--llm`, a flash model discovers each document's real
            structure (from its table of contents / style changes) AND summarizes
            each section in one call, producing descriptions + summaries in the
            graph. Documents over the model's context window degrade to the
            algorithmic path (no summaries) and are still ingested.

            Examples:
                cli docgraph build ./docs --db ./data/kg/tree.db
                cli docgraph build ./Alko.zip --db ./data/kg/tree.db --force md
                cli docgraph build ./docs --llm default --workers 8
                cli docgraph build ./docs --db ./data/kg/tree.db --llm-max-tokens 32000
            """
            _validate_force(force)
            db_path = _resolve_db_path(db_path)

            from genai_tk.config_mgmt.file_patterns import resolve_config_path
            from genai_tk.workflow.markdownize import markdownize_flow

            from genai_graph.orchestration.document_graph_flow import document_graph_flow

            resolved_md_output_dir = md_output_dir or str(Path(db_path).with_suffix("")) + "_markdown"

            # Markdownize each source into its own subdirectory (named after the source's
            # stem) so the Document Graph's top-level Folder is named after the original
            # zip/directory instead of the shared markdownize output directory.
            per_source_dirs: list[str] = []
            for src in source:
                stem = Path(resolve_config_path(src)).stem
                src_output_dir = str(Path(resolved_md_output_dir) / stem)
                src_cache_dir = str(Path(cache_dir) / stem) if cache_dir else None

                console.print(f"[dim]Markdownizing {src} -> {src_output_dir}[/dim]")
                markdownize_flow(
                    sources=[src],
                    md_output_dir=src_output_dir,
                    cache_dir=src_cache_dir,
                    profile=profile,
                    force_stage=force,
                )
                per_source_dirs.append(src_output_dir)

            result_dict = document_graph_flow(
                sources=per_source_dirs,
                db_path=db_path,
                include=include or ["*.md"],
                exclude=exclude or [],
                force_stage=force,
                delete_first=delete_first,
                llm=llm,
                llm_max_tokens=llm_max_tokens,
                summary_min_tokens=summary_min_tokens,
                outline_cache_dir=outline_cache_dir,
                workers=workers,
                context_safety_ratio=context_safety_ratio,
            )

            table = Table(title="Document Graph — Build Result")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")
            table.add_row("Processed", str(result_dict["documents_processed"]))
            table.add_row("Skipped (unchanged)", str(result_dict["documents_skipped"]))
            table.add_row("Failed", str(result_dict["documents_failed"]))
            table.add_row("Sections created", str(result_dict["sections_created"]))
            table.add_row("Sections summarized", str(result_dict.get("sections_summarized", 0)))
            table.add_row(
                "Files degraded to algo (over context window)",
                str(result_dict.get("files_degraded", 0)),
            )
            table.add_row("Relationships created", str(result_dict["relationships_created"]))
            console.print(table)
            for w in result_dict["warnings"]:
                console.print(f"[yellow]⚠ {w}[/yellow]")

        @cli_app.command("delete")
        def delete_db(
            db_path: Annotated[
                str | None,
                typer.Option(
                    "--db", help="Path to the Ladybug database file. Uses graph_db.default from config if omitted."
                ),
            ] = None,
            yes: Annotated[
                bool,
                typer.Option("--yes", "-y", help="Skip the confirmation prompt."),
            ] = False,
        ) -> None:
            """Delete all documents, folders, and sections from the graph (keeps the database file)."""
            from genai_graph.kg.backend import KuzuBackend
            from genai_graph.kg.document_graph.ingest import drop_document_graph

            db_path = _resolve_db_path(db_path)
            backend = KuzuBackend()
            backend.connect(db_path)

            if not yes:
                console.print("[bold red]This will delete all documents, folders, and sections.[/bold red]")
                if not typer.confirm("Continue?"):
                    console.print("[yellow]Aborted.[/yellow]")
                    raise typer.Exit(1)

            drop_document_graph(backend, drop_documents=True)
            console.print("[green]Deleted all documents, folders, and sections.[/green]")

        @cli_app.command("list")
        def list_docs(
            folder: Annotated[
                str | None,
                typer.Option("--folder", help="Only show documents under this folder (hash, prefix, or name)."),
            ] = None,
            db_path: Annotated[
                str | None,
                typer.Option(
                    "--db", help="Path to the Ladybug database file. Uses graph_db.default from config if omitted."
                ),
            ] = None,
        ) -> None:
            """List ingested documents, optionally filtered to one folder's subtree."""
            db_path = _resolve_db_path(db_path)
            from genai_graph.kg.backend import KuzuBackend
            from genai_graph.kg.query.document_graph_tools import list_documents

            backend = KuzuBackend()
            backend.connect(db_path)

            folder_id = _resolve_folder_ref_or_exit(backend, folder)
            rows = list_documents(backend, folder_id=folder_id)
            if not rows:
                console.print("[yellow]No documents ingested yet.[/yellow]")
                return

            table = Table(title="Documents")
            table.add_column("Filename", style="cyan")
            table.add_column("Folder", style="magenta")
            table.add_column("Sections", style="white")
            table.add_column("Markdown Hash", style="dim")
            table.add_column("Path", style="dim")
            for r in rows:
                breadcrumb = str(PurePosixPath(r["path"]).parent) if r.get("path") else "."
                table.add_row(
                    str(r["filename"]), breadcrumb, str(r["section_count"]), str(r["markdown_hash"]), str(r["path"])
                )
            console.print(table)

        @cli_app.command("toc")
        def toc(
            document: Annotated[str, typer.Argument(help="Document hash (or prefix), filename, or folder hash/name.")],
            db_path: Annotated[
                str | None,
                typer.Option(
                    "--db", help="Path to the Ladybug database file. Uses graph_db.default from config if omitted."
                ),
            ] = None,
            yaml_out: Annotated[
                bool,
                typer.Option("--yaml", help="Print as YAML (with descriptions), for feeding to an agent."),
            ] = False,
            summaries: Annotated[
                bool,
                typer.Option("--summaries", help="With --yaml, also include the fuller per-section summaries."),
            ] = False,
            max_level: Annotated[
                int | None,
                typer.Option("--max-level", help="With --yaml, only show sections down to this heading level."),
            ] = None,
        ) -> None:
            """Show the table of contents for one document, or list a folder's contents."""
            db_path = _resolve_db_path(db_path)
            from genai_graph.kg.backend import KuzuBackend
            from genai_graph.kg.query.document_graph_tools import (
                document_toc_yaml,
                folder_toc_yaml,
                get_document_toc,
                get_folder_tree,
                list_documents,
                render_toc_outline,
                resolve_folder_id,
            )

            backend = KuzuBackend()
            backend.connect(db_path)

            folder_id = resolve_folder_id(backend, document)
            if folder_id is not None:
                if yaml_out:
                    console.print(
                        folder_toc_yaml(
                            backend,
                            folder_id,
                            include_sections=True,
                            include_summaries=summaries,
                            max_level=max_level,
                        ),
                        soft_wrap=True,
                    )
                    return
                subfolders = [r for r in get_folder_tree(backend, folder_id) if r["parent_folder_id"] == folder_id]
                docs = [r for r in list_documents(backend, folder_id=folder_id) if r["folder_id"] == folder_id]
                if not subfolders and not docs:
                    console.print(f"[yellow]Folder is empty: {document}[/yellow]")
                    return
                for f in subfolders:
                    console.print(f"- \U0001f4c1 {f['name']} ({f['folder_id']}, {f['doc_count']} doc(s))")
                for d in docs:
                    console.print(f"- \U0001f4c4 {d['filename']} ({d['markdown_hash']})")
                return

            if yaml_out:
                console.print(
                    document_toc_yaml(backend, document, include_summaries=summaries, max_level=max_level),
                    soft_wrap=True,
                )
                return

            rows = get_document_toc(backend, document)
            if not rows:
                console.print(f"[yellow]No sections found for document: {document}[/yellow]")
                return
            console.print(render_toc_outline(rows))  # type: ignore[arg-type]

        @cli_app.command("folder-toc")
        def folder_toc(
            folder: Annotated[
                str | None,
                typer.Argument(help="Folder hash/name to root the TOC at. Covers every document if omitted."),
            ] = None,
            db_path: Annotated[
                str | None,
                typer.Option(
                    "--db", help="Path to the Ladybug database file. Uses graph_db.default from config if omitted."
                ),
            ] = None,
            sections: Annotated[
                bool,
                typer.Option("--sections", help="Also inline each document's section tree."),
            ] = False,
            summaries: Annotated[
                bool,
                typer.Option("--summaries", help="Also include the fuller summaries."),
            ] = False,
        ) -> None:
            """Print the documents under a folder's subtree as YAML, each with a one-line description.

            Sections are omitted by default — this is the orientation view: pick a document,
            then run `docgraph toc <id> --yaml` for its sections. Pass --sections to inline them.
            """
            db_path = _resolve_db_path(db_path)
            from genai_graph.kg.backend import KuzuBackend
            from genai_graph.kg.query.document_graph_tools import folder_toc_yaml

            backend = KuzuBackend()
            backend.connect(db_path)
            folder_id = _resolve_folder_ref_or_exit(backend, folder)
            console.print(
                folder_toc_yaml(backend, folder_id, include_sections=sections, include_summaries=summaries),
                soft_wrap=True,
            )

        @cli_app.command("cat")
        def cat(
            document: Annotated[
                str,
                typer.Argument(
                    help="Document hash (or prefix), filename, or a section id "
                    "(e.g. 'd9387cdaf256734a::1') to show just that section and its subsections."
                ),
            ],
            db_path: Annotated[
                str | None,
                typer.Option(
                    "--db", help="Path to the Ladybug database file. Uses graph_db.default from config if omitted."
                ),
            ] = None,
            cypher: Annotated[
                bool,
                typer.Option("--cypher", help="Print the Cypher query used to fetch the content."),
            ] = False,
            raw: Annotated[
                bool,
                typer.Option("--raw", help="Print raw Markdown text instead of rendering it."),
            ] = False,
        ) -> None:
            """Reconstruct and print a document's (or one section's) Markdown text from its sections."""
            db_path = _resolve_db_path(db_path)
            from genai_graph.kg.backend import KuzuBackend
            from genai_graph.kg.query.document_graph_tools import (
                reconstruct_document,
                reconstruct_section,
                resolve_folder_id,
            )

            backend = KuzuBackend()
            backend.connect(db_path)

            if "::" not in document and resolve_folder_id(backend, document) is not None:
                console.print(
                    f"[red]'{document}' is a folder, not a document.[/red] Use "
                    f"[cyan]docgraph list --folder {document}[/cyan] or [cyan]docgraph folders {document}[/cyan]."
                )
                raise typer.Exit(1)

            if "::" in document:
                text, query = reconstruct_section(backend, document, return_query=True)
            else:
                text, query = reconstruct_document(backend, document, return_query=True)

            if cypher:
                console.print(Panel(query, title="Cypher", border_style="cyan"))

            if text is None:
                console.print(f"[red]No document or section found matching: {document}[/red]")
                raise typer.Exit(1)

            if raw:
                console.print(text)
            else:
                from rich.markdown import Markdown as RichMarkdown

                console.print(RichMarkdown(text))

        @cli_app.command("search")
        def search(
            keyword: Annotated[str, typer.Argument(help="Keyword or query to search for in section titles/text.")],
            db_path: Annotated[
                str | None,
                typer.Option(
                    "--db", help="Path to the Ladybug database file. Uses graph_db.default from config if omitted."
                ),
            ] = None,
            limit: Annotated[int, typer.Option("--limit", "-l", help="Max number of matches.")] = 20,
            folder: Annotated[
                str | None,
                typer.Option(
                    "--folder", "-f", help="Restrict the search to this folder's subtree (hash, prefix, or name)."
                ),
            ] = None,
            doc: Annotated[
                str | None,
                typer.Option(
                    "--doc",
                    "--document",
                    "-d",
                    help="Restrict the search to this document (hash, prefix, filename, or path).",
                ),
            ] = None,
            node: Annotated[
                str | None,
                typer.Option(
                    "--node", "-n", help="Restrict the search to a folder or document node (hash, prefix, or name)."
                ),
            ] = None,
            mode: Annotated[
                str,
                typer.Option(
                    "--mode",
                    "-m",
                    help="Search mode: 'hybrid' (vector + BM25, default), 'vector' (semantic only), 'bm25' (FTS only), or 'cypher' (native CONTAINS search).",
                ),
            ] = "hybrid",
            embeddings: Annotated[
                str | None,
                typer.Option(
                    "--embeddings",
                    "--model",
                    help="Embeddings model ID or tag for vector/hybrid search (e.g. 'default', 'bge-small-en@local').",
                ),
            ] = None,
        ) -> None:
            """Search section titles and text across ingested documents, with hybrid, vector, BM25, or native Cypher mode."""
            db_path = _resolve_db_path(db_path)
            from genai_graph.kg.backend import KuzuBackend
            from genai_graph.kg.query.document_graph_tools import (
                get_available_indexes,
                resolve_node_ref,
                search_sections,
            )

            backend = KuzuBackend()
            backend.connect(db_path)

            folder_id: str | None = None
            doc_id: str | None = None

            if node is not None:
                node_type, n_id = resolve_node_ref(backend, node)
                if node_type == "folder":
                    folder_id = n_id
                elif node_type == "document":
                    doc_id = n_id
                else:
                    console.print(f"[red]No folder or document found matching --node: {node}[/red]")
                    raise typer.Exit(1)

            if folder is not None:
                folder_id = _resolve_folder_ref_or_exit(backend, folder)

            if doc is not None:
                doc_id = _resolve_doc_ref_or_exit(backend, doc)

            mode_norm = mode.lower().strip()
            valid_modes = {
                "hybrid",
                "all",
                "vector",
                "semantic",
                "bm25",
                "fts",
                "cypher",
                "native",
                "keyword",
                "contains",
            }
            if mode_norm not in valid_modes:
                console.print(
                    f"[red]Invalid search mode '{mode}'. Choose one of: 'hybrid' (default), 'vector', 'bm25', 'cypher'.[/red]"
                )
                raise typer.Exit(1)

            available = get_available_indexes(backend)

            # Inform user if indexes are unavailable for the requested/default mode
            if mode_norm in ("hybrid", "all"):
                if not available["vector"]:
                    console.print("[dim]ℹ Vector index not found in database; using BM25 keyword search.[/dim]")
                if not available["fts"]:
                    console.print("[dim]ℹ BM25 (FTS) index not found in database; using native Cypher search.[/dim]")
            elif mode_norm in ("vector", "semantic"):
                if not available["vector"]:
                    console.print(
                        "[yellow]⚠ Vector index ('chunk_embedding_index') not found in database. Search may return no results.[/yellow]"
                    )
            elif mode_norm in ("bm25", "fts"):
                if not available["fts"]:
                    console.print(
                        "[yellow]⚠ BM25 FTS index ('section_fts') not found in database; falling back to native Cypher search.[/yellow]"
                    )

            rows = search_sections(
                backend,
                keyword,
                limit=limit,
                folder_id=folder_id,
                document_id=doc_id,
                mode=mode_norm,
                embeddings_id=embeddings,
            )
            if not rows:
                console.print(f"[yellow]No sections matched query: {keyword!r}[/yellow]")
                return
            for r in rows:  # type: ignore[union-attr]
                score_str = f", score: {r['score']}" if r.get("score") and r["score"] > 0 else ""
                console.print(
                    f"- [{r['section_id']}] {r['title']} (line {r['line_start']}{score_str}) — {r['markdown_hash']}"
                )
                if r.get("matched_chunk"):
                    console.print(f"    [dim]Chunk: {r['matched_chunk']}[/dim]")

        @cli_app.command("folders")
        def folders(
            ref: Annotated[
                str | None,
                typer.Argument(help="Folder hash/name to root the tree at. Shows every source folder if omitted."),
            ] = None,
            db_path: Annotated[
                str | None,
                typer.Option(
                    "--db", help="Path to the Ladybug database file. Uses graph_db.default from config if omitted."
                ),
            ] = None,
        ) -> None:
            """Display the ingested folder hierarchy as a tree."""
            db_path = _resolve_db_path(db_path)
            from genai_graph.kg.backend import KuzuBackend
            from genai_graph.kg.query.document_graph_tools import get_folder_tree

            backend = KuzuBackend()
            backend.connect(db_path)
            root_id = _resolve_folder_ref_or_exit(backend, ref)

            rows = get_folder_tree(backend, root_id)
            if not rows:
                console.print("[yellow]No folders ingested yet.[/yellow]")
                return

            by_id = {r["folder_id"]: r for r in rows}
            by_parent: dict[str | None, list[dict[str, Any]]] = {}
            for r in rows:
                by_parent.setdefault(r["parent_folder_id"], []).append(r)

            root_rows = [by_id[root_id]] if root_id else [r for r in rows if r["parent_folder_id"] is None]

            def add_node(parent: Tree, row: dict[str, Any]) -> None:
                label = f"{row['name']} [dim]({row['folder_id']}, {row['doc_count']} doc(s))[/dim]"
                node = parent.add(label)
                for child in sorted(by_parent.get(row["folder_id"], []), key=lambda r: r["name"]):
                    add_node(node, child)

            tree = Tree("[bold]Folders[/bold]")
            for r in sorted(root_rows, key=lambda r: r["name"]):
                add_node(tree, r)
            console.print(tree)

        @cli_app.command("tui")
        def tui(
            db_path: Annotated[
                str | None,
                typer.Option(
                    "--db", help="Path to the Ladybug database file. Uses graph_db.default from config if omitted."
                ),
            ] = None,
        ) -> None:
            """Launch an interactive Textual TUI to browse the Document Graph."""
            db_path = _resolve_db_path(db_path)
            from genai_graph.kg.query.document_graph_tui import run_document_graph_tui

            run_document_graph_tui(db_path)

        @cli_app.command("agent")
        def agent(
            query: Annotated[
                str | None,
                typer.Argument(
                    help="Question to answer by navigating the Document Graph (omit with --chat for interactive mode)."
                ),
            ] = None,
            profile: Annotated[
                str,
                typer.Option("--profile", "-p", help="Agent profile key (default: docgraph)."),
            ] = "docgraph",
            llm: Annotated[
                str,
                typer.Option("--llm", "-m", help="LLM id (name@provider) or tag. Defaults to the profile's LLM."),
            ] = "default",
            db_path: Annotated[
                str | None,
                typer.Option(
                    "--db",
                    help="Path to the Ladybug database file. Uses graph_db.default from config if omitted.",
                ),
            ] = None,
            folder: Annotated[
                str | None,
                typer.Option("--folder", help="Folder to scope the agent to (hash, prefix, or name)."),
            ] = None,
            skill_dir: Annotated[
                list[str] | None,
                typer.Option("--skill-dir", help="Additional runtime skill directory (repeatable)."),
            ] = None,
            recursion_limit: Annotated[
                int, typer.Option("--recursion-limit", help="Max LangGraph steps per turn.")
            ] = 120,
            chat: Annotated[bool, typer.Option("--chat", help="Interactive multi-turn REPL (memory enabled).")] = False,
            trace: Annotated[bool, typer.Option("--trace", help="Print graph node trace lines.")] = False,
        ) -> None:
            """Run a deep agent that navigates the Document Graph to answer a query.

            Examples:
                cli docgraph agent "What IT services is Alko requesting?" --folder folder_273e65da416b2e72
                cli docgraph agent --chat --folder folder_273e65da416b2e72
                cli docgraph agent "Summarize the SLAs" --llm deepseek_v4flash@openrouter
            """
            import asyncio

            from genai_tk.agents.harness.profiles import load_langchain_profiles

            from genai_graph.agent import create_docgraph_agent, run_docgraph_agent
            from genai_graph.kg.query.document_graph_tools import DocumentGraphError

            profiles = load_langchain_profiles()
            if profile not in profiles:
                console.print(f"[red]Agent profile {profile!r} not found. Available: {sorted(profiles)}[/red]")
                raise typer.Exit(1)
            agent_profile = profiles[profile]
            agent_profile.recursion_limit = recursion_limit
            llm_id = llm if llm != "default" else None

            async def _run() -> None:
                try:
                    harness = create_docgraph_agent(
                        agent_profile,
                        llm=llm_id,
                        db_path=db_path,
                        folder_id=folder,
                        extra_skill_dirs=skill_dir,
                    )
                except DocumentGraphError as exc:
                    console.print(f"[red]{exc}[/red]")
                    raise typer.Exit(1) from exc

                try:
                    if chat:
                        from genai_tk.agents.harness.chat_repl import run_chat_repl

                        console.print(
                            f"[cyan]Document Graph agent ({profile}) — interactive mode. Type /quit to exit.[/cyan]\n"
                        )
                        await run_chat_repl(harness, initial_query=query, show_trace=trace)
                        return

                    if not query:
                        console.print(
                            "[red]A query is required in one-shot mode (or use --chat for interactive mode).[/red]"
                        )
                        raise typer.Exit(1)

                    label = llm_id or agent_profile.llm or "default"
                    console.print(f"[dim]Running {profile} agent (llm={label})…[/dim]\n")
                    await run_docgraph_agent(harness, query, show_trace=trace)
                    console.print()
                finally:
                    await harness.aclose()

            try:
                asyncio.run(_run())
            except typer.Exit:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.debug("docgraph agent error", exc_info=True)
                console.print(f"[red]Agent error: {exc}[/red]")
                raise typer.Exit(1) from exc

        logger.debug("Registered 'docgraph' CLI commands")


def _render_plan(invocation: Any) -> None:
    """Print a summary table of the resolved workflow plan."""
    import json as _json

    from genai_tk.workflow.models import ResolvedWorkflowInvocation

    if not isinstance(invocation, ResolvedWorkflowInvocation):
        return

    summary = Table(title="Workflow Plan", show_header=True, header_style="bold cyan")
    summary.add_column("Property", style="cyan", no_wrap=True)
    summary.add_column("Value", style="white")
    summary.add_row("Workflow", invocation.workflow_name)
    summary.add_row("Profile", invocation.profile_name or "<none>")
    summary.add_row("Force stage", invocation.force_stage or "<none>")
    summary.add_row("Steps", str(len(invocation.workflow.steps)))
    console.print(summary)

    if invocation.values:
        console.print(Panel(_json.dumps(invocation.values, indent=2, default=str), title="Effective Values"))
