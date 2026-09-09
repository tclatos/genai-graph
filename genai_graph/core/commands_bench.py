"""CLI commands for benchmark evaluation and inspection (``cli bench ...``)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from genai_tk.cli.base import CliTopCommand
from rich.console import Console
from rich.table import Table

from genai_graph.bench.adapters.base import get_benchmark_adapter
from genai_graph.bench.config import (
    ALL_STEPS,
    list_bench_profiles,
    load_bench_profile,
    load_env,
)
from genai_graph.bench.flows import full_bench_flow
from genai_graph.bench.judge import load_existing_scores
from genai_graph.bench.summary import (
    compute_bench_summary,
    display_bench_summary,
    save_bench_summary,
)
from genai_graph.bench.tui import (
    display_questions_table,
    display_single_question_panel,
    load_bench_dataset_with_results,
    run_bench_tui,
)

console = Console()


class BenchCommands(CliTopCommand):
    """Benchmark commands for graph agent evaluation."""

    description: str = "Run and inspect benchmark evaluations across datasets"

    def get_description(self) -> tuple[str, str]:
        return "bench", self.description

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        @cli_app.command("list")
        def list_profiles(
            config_path: Annotated[
                str | None,
                typer.Option("-c", "--config", help="Path to bench YAML configuration file"),
            ] = None,
        ) -> None:
            """List configured benchmark run profiles."""
            cfg_p = Path(config_path) if config_path else None
            profiles = list_bench_profiles(cfg_p)
            if not profiles:
                console.print("[yellow]No benchmark profiles found in configuration.[/yellow]")
                return

            table = Table(title="Benchmark Run Profiles")
            table.add_column("Profile", style="bold cyan")
            table.add_column("Description", style="white")
            table.add_column("Markdownize", style="magenta")
            table.add_column("Agent LLM", style="green")
            table.add_column("Judge LLM", style="blue")
            table.add_column("Files / Pathspecs", style="yellow")

            for name, data in profiles.items():
                desc = data.get("description", "")
                md_prof = data.get("markdownize_profile", "medium")
                llms = data.get("llms", {}) or {}
                agent_llm = llms.get("agent", "default")
                judge_llm = llms.get("judge", "default")
                files = data.get("files", {}) or data.get("questions", {}) or {}
                pathspecs = files.get("pathspecs", [])
                docs = files.get("docs", [])
                if pathspecs:
                    files_src = f"specs: {', '.join(pathspecs)}"
                elif docs:
                    files_src = f"{len(docs)} doc(s)"
                else:
                    files_src = "all docs"
                table.add_row(name, desc, md_prof, agent_llm, judge_llm, files_src)

            console.print(table)

        @cli_app.command("run")
        def run(
            profile: Annotated[
                str | None,
                typer.Option(
                    "-p",
                    "--profile",
                    help="Bench run profile key (defaults to default_profile in config)",
                ),
            ] = None,
            pathspecs: Annotated[
                str | None,
                typer.Option(
                    "-f",
                    "--files",
                    "--pathspecs",
                    help="Comma-separated pathspec patterns to filter documents (e.g. 'BESTBUY*,Pfizer*')",
                ),
            ] = None,
            docs: Annotated[
                str | None,
                typer.Option(
                    "-d",
                    "--docs",
                    help="Comma-separated doc_names overriding config files.docs",
                ),
            ] = None,
            question_ids: Annotated[
                str | None,
                typer.Option(
                    "-q",
                    "--question-ids",
                    help="Comma-separated question IDs (e.g. 'officeqa_12,officeqa_15')",
                ),
            ] = None,
            judge: Annotated[
                bool | None,
                typer.Option(
                    "--judge/--no-judge",
                    help="Enable or disable LLM-as-judge analysis (default: from config)",
                ),
            ] = None,
            step: Annotated[
                str | None,
                typer.Option("--step", help="Run only this one step (fetch, build, run, grade)"),
            ] = None,
            skip: Annotated[
                list[str] | None,
                typer.Option("--skip", help="Skip step(s) (fetch, build, run, grade)"),
            ] = None,
            limit: Annotated[
                int | None,
                typer.Option("-n", "--limit", help="Run only the first N questions"),
            ] = None,
            monitoring: Annotated[
                str | None,
                typer.Option(
                    "-m",
                    "--monitoring",
                    help="Tracing monitoring method ('none', 'langchain', 'langsmith', 'langfuse', 'local')",
                ),
            ] = None,
            force: Annotated[
                bool,
                typer.Option("--force", help="Force rebuild of OCR and document graph"),
            ] = False,
            rerun: Annotated[
                bool,
                typer.Option("--rerun", "--force-run", help="Force re-running agent turns for questions"),
            ] = False,
            config_path: Annotated[
                str | None,
                typer.Option("-c", "--config", help="Path to bench YAML configuration file"),
            ] = None,
        ) -> None:
            """Execute a benchmark evaluation run.

            Examples:
                cli bench run
                cli bench run -p mistral_glm -n 3
                cli bench run --no-judge
                cli bench run -f 'BESTBUY*,Pfizer*'
                cli bench run --skip fetch --skip build
                cli bench run --step run -n 1
            """
            load_env()
            if step and step not in ALL_STEPS:
                console.print(f"[red]Error:[/red] Invalid step '{step}'. Choose from: {ALL_STEPS}")
                raise typer.Exit(1)

            cfg_p = Path(config_path) if config_path else None
            cfg = load_bench_profile(
                profile_name=profile,
                config_path=cfg_p,
                build_force=force or None,
                force_run=rerun or None,
                limit=limit if limit is not None else None,
                judge_enabled=judge if judge is not None else None,
                monitoring=monitoring if monitoring is not None else None,
            )

            # Parse pathspecs, docs, and question_ids overrides
            p_specs = [s.strip() for s in pathspecs.split(",") if s.strip()] if pathspecs else None
            d_list = [d.strip() for d in docs.split(",") if d.strip()] if docs else None
            if question_ids:
                cfg.question_ids = [q.strip() for q in question_ids.split(",") if q.strip()]

            adapter = get_benchmark_adapter(cfg.adapter, project_root=cfg.project_root)
            available_docs = adapter.get_available_docs()
            cfg.docs = cfg.resolve_docs(
                available_docs=available_docs,
                docs_override=d_list,
                pathspecs_override=p_specs,
            )

            console.print(f"[bold green]Starting Benchmark Run:[/bold green] profile={cfg.profile_name}")
            console.print(
                f"Target documents ({len(cfg.docs)}): {', '.join(cfg.docs[:5])}{'...' if len(cfg.docs) > 5 else ''}"
            )
            console.print(f"Agent LLM: [cyan]{cfg.agent_llm}[/cyan] | Judge LLM: [magenta]{cfg.judge_llm}[/magenta]")

            full_bench_flow(cfg, step=step, skip=skip)

        @cli_app.command("grade")
        def grade(
            profile: Annotated[
                str | None,
                typer.Option("-p", "--profile", help="Bench run profile key"),
            ] = None,
            config_path: Annotated[
                str | None,
                typer.Option("-c", "--config", help="Path to bench YAML configuration file"),
            ] = None,
        ) -> None:
            """Grade existing question runs with LLM-as-judge without re-running agent turns."""
            load_env()
            cfg_p = Path(config_path) if config_path else None
            cfg = load_bench_profile(profile_name=profile, config_path=cfg_p)
            from genai_graph.bench.flows import grade_flow

            scores = grade_flow(cfg)
            summary = compute_bench_summary(
                scores,
                profile_name=cfg.profile_name,
                agent_llm=cfg.agent_llm,
                judge_llm=cfg.judge_llm,
            )
            save_bench_summary(summary, Path(cfg.scores_summary))
            display_bench_summary(summary)

        @cli_app.command("report")
        def report(
            profile: Annotated[
                str | None,
                typer.Option("-p", "--profile", help="Bench run profile key"),
            ] = None,
            config_path: Annotated[
                str | None,
                typer.Option("-c", "--config", help="Path to bench YAML configuration file"),
            ] = None,
        ) -> None:
            """Display summary metrics and error breakdown for an existing scored run."""
            cfg_p = Path(config_path) if config_path else None
            cfg = load_bench_profile(profile_name=profile, config_path=cfg_p)
            summary_p = Path(cfg.scores_summary)
            if summary_p.exists():
                from genai_graph.bench.models import BenchSummary

                summary = BenchSummary.model_validate_json(summary_p.read_text(encoding="utf-8"))
                display_bench_summary(summary)
            else:
                scores_p = Path(cfg.scores)
                if not scores_p.exists():
                    console.print(
                        f"[red]Error:[/red] No scores or summary found for profile '{cfg.profile_name}'. Run 'cli bench run' first."
                    )
                    raise typer.Exit(1)
                scores = list(load_existing_scores(scores_p).values())
                summary = compute_bench_summary(
                    scores,
                    profile_name=cfg.profile_name,
                    agent_llm=cfg.agent_llm,
                    judge_llm=cfg.judge_llm,
                )
                display_bench_summary(summary)

        @cli_app.command("questions")
        def questions(
            profile: Annotated[
                str | None,
                typer.Option("-p", "--profile", help="Bench run profile key"),
            ] = None,
            question_id: Annotated[
                str | None,
                typer.Option("-q", "--question-id", help="Select and inspect a single question by ID"),
            ] = None,
            tui: Annotated[
                bool,
                typer.Option("--tui/--no-tui", "-t", help="Launch interactive Textual TUI browser"),
            ] = False,
            limit: Annotated[
                int | None,
                typer.Option("-n", "--limit", help="Limit number of questions displayed in table mode"),
            ] = None,
            config_path: Annotated[
                str | None,
                typer.Option("-c", "--config", help="Path to bench YAML configuration file"),
            ] = None,
        ) -> None:
            """List questions with gold answers, agent outputs, and grader comments.

            Examples:
                cli bench questions
                cli bench questions -q FB_001
                cli bench questions --tui
                cli bench questions -n 20
            """
            load_env()
            cfg_p = Path(config_path) if config_path else None
            cfg = load_bench_profile(profile_name=profile, config_path=cfg_p)

            if tui:
                run_bench_tui(cfg, initial_question_id=question_id)
                return

            items = load_bench_dataset_with_results(cfg, question_id=question_id)
            if not items:
                if question_id:
                    console.print(f"[yellow]No question found with ID:[/yellow] {question_id}")
                else:
                    console.print("[yellow]No questions found in benchmark dataset.[/yellow]")
                return

            if question_id or len(items) == 1:
                display_single_question_panel(items[0])
            else:
                if limit and limit > 0:
                    items = items[:limit]
                display_questions_table(items, title=f"Benchmark Questions: {cfg.profile_name} ({len(items)} items)")
                console.print(
                    "[dim](Tip: run [bold cyan]cli bench questions -q <ID>[/bold cyan] to inspect one question, "
                    "or [bold cyan]cli bench tui[/bold cyan] for interactive Textual browser)[/dim]"
                )

        @cli_app.command("tui")
        def tui_command(
            profile: Annotated[
                str | None,
                typer.Option("-p", "--profile", help="Bench run profile key"),
            ] = None,
            question_id: Annotated[
                str | None,
                typer.Option("-q", "--question-id", help="Focus on specific question ID on startup"),
            ] = None,
            config_path: Annotated[
                str | None,
                typer.Option("-c", "--config", help="Path to bench YAML configuration file"),
            ] = None,
        ) -> None:
            """Launch the interactive Textual TUI to navigate the benchmark dataset.

            Examples:
                cli bench tui
                cli bench tui -p glm_5.3_Flash
                cli bench tui -q UID0056
            """
            load_env()
            cfg_p = Path(config_path) if config_path else None
            cfg = load_bench_profile(profile_name=profile, config_path=cfg_p)
            run_bench_tui(cfg, initial_question_id=question_id)
