"""Interactive Textual TUI and CLI viewer for benchmark dataset exploration.

Displays questions, gold reference answers, agent execution responses,
and LLM-as-judge evaluations with search, filtering, and detailed inspection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import DataTable, Footer, Header, Input, Markdown, Select, Static

from genai_graph.bench.adapters.base import resolve_benchmark_adapter
from genai_graph.bench.config import BenchConfig
from genai_graph.bench.judge import load_existing_scores
from genai_graph.bench.models import BenchQuestion
from genai_graph.bench.runner import load_existing_runs

console = Console()


class BenchQuestionDetail(BaseModel):
    """Joined question item with execution run and grader verdict if present."""

    id: str
    doc_name: str = ""
    doc_names: list[str] = Field(default_factory=list)
    question: str = ""
    gold_answer: str = ""
    justification: str | None = None
    evidence: list[Any] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Run execution information
    has_run: bool = False
    agent_answer: str | None = None
    agent_thinking: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    n_tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    run_error: str | None = None
    run_llm: str | None = None
    started_at: str | None = None

    # Grader evaluation information
    has_score: bool = False
    correctness: str | None = None  # "correct" | "partial" | "incorrect"
    numeric_match: bool | None = None
    groundedness: str | None = None  # "grounded" | "partial" | "ungrounded"
    error_category: str | None = None
    rationale: str | None = None
    judge_llm: str | None = None
    scored_at: str | None = None

    @property
    def status_label(self) -> str:
        if self.has_score:
            if self.correctness == "correct":
                return "✓ Correct"
            if self.correctness == "partial":
                return "~ Partial"
            return "✗ Incorrect"
        if self.has_run:
            return "● Run"
        return "- Unrun"

    @property
    def status_style(self) -> str:
        if self.has_score:
            if self.correctness == "correct":
                return "green"
            if self.correctness == "partial":
                return "yellow"
            return "red"
        if self.has_run:
            return "cyan"
        return "dim"


def load_bench_dataset_with_results(
    cfg: BenchConfig,
    question_id: str | None = None,
) -> list[BenchQuestionDetail]:
    """Load and join dataset questions with existing run records and grader verdicts."""
    adapter = resolve_benchmark_adapter(cfg.adapter, project_root=cfg.project_root)
    questions: list[BenchQuestion] = adapter.load_dataset()

    runs = load_existing_runs(Path(cfg.runs))
    scores = load_existing_scores(Path(cfg.scores))

    items: list[BenchQuestionDetail] = []
    for q in questions:
        q_id = q.id
        if question_id and q_id.lower() != question_id.lower():
            continue

        run_rec = runs.get(q_id)
        score_data = scores.get(q_id)

        detail = BenchQuestionDetail(
            id=q_id,
            doc_name=q.doc_name,
            doc_names=q.doc_names or ([q.doc_name] if q.doc_name else []),
            question=q.question,
            gold_answer=q.gold_answer,
            justification=q.justification,
            evidence=q.evidence,
            metadata=q.metadata,
        )

        if run_rec:
            detail.has_run = True
            detail.agent_answer = run_rec.agent_answer
            detail.agent_thinking = run_rec.agent_thinking
            detail.tool_calls = run_rec.tool_calls
            detail.tool_results = run_rec.tool_results
            detail.n_tool_calls = run_rec.n_tool_calls or len(run_rec.tool_calls)
            detail.input_tokens = run_rec.input_tokens
            detail.output_tokens = run_rec.output_tokens
            detail.run_error = run_rec.error
            detail.run_llm = run_rec.llm
            detail.started_at = run_rec.started_at

        if score_data:
            detail.has_score = True
            detail.correctness = score_data.get("correctness")
            detail.numeric_match = score_data.get("numeric_match")
            detail.groundedness = score_data.get("groundedness")
            detail.error_category = score_data.get("error_category")
            detail.rationale = score_data.get("rationale")
            detail.judge_llm = score_data.get("judge_llm")
            detail.scored_at = score_data.get("scored_at")

            # If score row contains run details that weren't in runs
            if not detail.has_run and score_data.get("agent_answer"):
                detail.has_run = True
                detail.agent_answer = score_data.get("agent_answer")
                detail.agent_thinking = score_data.get("agent_thinking")
                detail.tool_calls = score_data.get("tool_calls") or []
                detail.tool_results = score_data.get("tool_results") or []
                detail.n_tool_calls = int(score_data.get("n_tool_calls") or len(detail.tool_calls))
                detail.input_tokens = int(score_data.get("input_tokens") or 0)
                detail.output_tokens = int(score_data.get("output_tokens") or 0)
                detail.run_error = score_data.get("error")

        items.append(detail)

    return items


def format_question_markdown(q: BenchQuestionDetail, show_tool_calls: bool = True) -> str:
    """Render a comprehensive Markdown representation of a question detail."""
    import json

    lines: list[str] = []

    # Status header
    status_icon = (
        "🟢"
        if q.correctness == "correct"
        else ("🟡" if q.correctness == "partial" else ("🔴" if q.correctness == "incorrect" else "⚪"))
    )
    lines.append(f"# {status_icon} Question `{q.id}`")
    lines.append("")
    docs_str = ", ".join(f"`{d}`" for d in q.doc_names) if q.doc_names else f"`{q.doc_name}`"
    lines.append(f"**Document(s):** {docs_str}  ")
    lines.append(f"**Status:** `{q.status_label}`  ")

    if q.metadata:
        meta_items = [
            f"**{k}:** {v}"
            for k, v in q.metadata.items()
            if v and k not in ("source_files", "doc_names", "evidence_text_full_page", "financebench_id", "officeqa_id")
        ]
        if meta_items:
            lines.append(f"**Metadata:** {', '.join(meta_items[:4])}  ")

    lines.append("")
    lines.append("---")
    lines.append("## ❓ Question")
    lines.append(f"> {q.question}")
    lines.append("")

    # Gold Reference
    lines.append("## 🎯 Gold Answer")
    lines.append(f"```text\n{q.gold_answer}\n```")
    if q.justification:
        lines.append(f"**Justification:** {q.justification}  ")
    if q.evidence:
        lines.append("**Evidence citations:**")
        for ev in q.evidence:
            ev_text = ev.get("evidence_text") or str(ev) if isinstance(ev, dict) else str(ev)
            lines.append(f"- {ev_text}")
    lines.append("")

    # Agent Response
    lines.append("---")
    lines.append("## 🤖 Agent Response")
    if q.has_run or q.agent_answer:
        lines.append(f"```text\n{q.agent_answer or '(empty response)'}\n```")
        lines.append(
            f"**Model:** `{q.run_llm or 'default'}` | "
            f"**Tool Calls:** `{q.n_tool_calls}` | "
            f"**Tokens:** In `{q.input_tokens:,}` / Out `{q.output_tokens:,}`"
        )
        if q.run_error:
            lines.append(f"\n⚠️ **Error:** `{q.run_error}`")

        # Agent Thinking / Reasoning
        if q.agent_thinking:
            lines.append("\n### 🧠 Agent Reasoning / Thinking")
            lines.append(f"```text\n{q.agent_thinking.strip()}\n```")

        # Recorded Trajectory
        if q.tool_calls:
            lines.append(
                f"\n### 🛠️ Recorded Execution Trajectory ({len(q.tool_calls)} step{'s' if len(q.tool_calls) != 1 else ''})"
            )
            for idx, tc in enumerate(q.tool_calls, 1):
                name = tc.get("tool") or tc.get("name") or "tool"
                args = tc.get("args") or tc.get("arguments") or {}
                args_str = json.dumps(args, indent=2, ensure_ascii=False) if isinstance(args, dict) else str(args)
                lines.append(f"#### Step {idx}: `{name}`")
                lines.append(f"**Arguments:**\n```json\n{args_str}\n```")

                if idx - 1 < len(q.tool_results):
                    res_item = q.tool_results[idx - 1]
                    content = res_item.get("content") or res_item.get("result") or res_item.get("output") or ""
                    content_str = str(content)
                    max_len = 1500 if show_tool_calls else 200
                    if len(content_str) > max_len:
                        content_str = content_str[:max_len] + f"\n... [{len(content_str):,} characters total]"
                    lines.append(f"**Result:**\n```text\n{content_str}\n```")
                lines.append("")
    else:
        lines.append("*Question has not been executed yet.*")
    lines.append("")

    # Grader Evaluation
    lines.append("---")
    lines.append("## ⚖️ Grader Evaluation")
    if q.has_score:
        correctness_badge = f"**`{q.correctness.upper()}`**" if q.correctness else "`N/A`"
        lines.append(f"- **Verdict:** {correctness_badge}")
        if q.numeric_match is not None:
            num_str = "Yes (Match)" if q.numeric_match else "No (Mismatch)"
            lines.append(f"- **Numeric Match:** `{num_str}`")
        if q.groundedness:
            lines.append(f"- **Groundedness:** `{q.groundedness}`")
        if q.error_category:
            lines.append(f"- **Error Category:** `{q.error_category}`")
        if q.judge_llm:
            lines.append(f"- **Judge LLM:** `{q.judge_llm}`")
        if q.rationale:
            lines.append(f"\n### 📝 Grader Comment & Rationale\n> {q.rationale}")
    else:
        lines.append("*No grader evaluation recorded.*")

    return "\n".join(lines)

    return "\n".join(lines)


def display_questions_table(
    questions: list[BenchQuestionDetail],
    title: str = "Benchmark Questions",
) -> None:
    """Print an overview table of questions with status, gold answer, and grader comments."""
    table = Table(title=title, show_lines=True)
    table.add_column("ID", style="bold cyan", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Calls", style="cyan", justify="right", no_wrap=True)
    table.add_column("Doc", style="yellow")
    table.add_column("Question", style="white", ratio=3)
    table.add_column("Gold Answer", style="green", ratio=2)
    table.add_column("Agent Answer", style="magenta", ratio=2)
    table.add_column("Grader Comment", style="blue", ratio=3)

    for q in questions:
        status_styled = f"[{q.status_style}]{q.status_label}[/{q.status_style}]"
        agent_snippet = (
            (q.agent_answer[:80] + "...") if q.agent_answer and len(q.agent_answer) > 80 else (q.agent_answer or "-")
        )
        gold_snippet = (q.gold_answer[:80] + "...") if len(q.gold_answer) > 80 else q.gold_answer
        q_snippet = (q.question[:100] + "...") if len(q.question) > 100 else q.question
        comment_snippet = (
            (q.rationale[:100] + "...") if q.rationale and len(q.rationale) > 100 else (q.rationale or "-")
        )
        calls_str = str(q.n_tool_calls) if q.has_run or q.n_tool_calls > 0 else "-"

        table.add_row(
            q.id,
            status_styled,
            calls_str,
            q.doc_name or (q.doc_names[0] if q.doc_names else "-"),
            q_snippet,
            gold_snippet,
            agent_snippet,
            comment_snippet,
        )

    console.print(table)


def display_single_question_panel(q: BenchQuestionDetail, show_trajectory: bool = True) -> None:
    """Print full detailed Rich view for a single question including execution trajectory."""
    import json

    status_styled = f"[{q.status_style}]{q.status_label}[/{q.status_style}]"
    doc_str = ", ".join(q.doc_names) if q.doc_names else q.doc_name

    body = f"""\
[bold]Question:[/bold]
{q.question}

[bold green]Gold Answer:[/bold green]
{q.gold_answer}
"""
    if q.justification:
        body += f"\n[dim bold]Justification:[/dim bold] {q.justification}\n"
    if q.evidence:
        ev_lines = [f"- {e.get('evidence_text') or str(e)}" if isinstance(e, dict) else f"- {e}" for e in q.evidence]
        body += "\n[dim bold]Evidence:[/dim bold]\n" + "\n".join(ev_lines) + "\n"

    body += "\n[bold magenta]Agent Answer:[/bold magenta]\n"
    if q.has_run or q.agent_answer:
        body += f"{q.agent_answer or '(empty response)'}\n"
        body += f"[dim]Model: {q.run_llm or 'default'} | Tools: {q.n_tool_calls} | Tokens: In {q.input_tokens:,} / Out {q.output_tokens:,}[/dim]\n"
        if q.run_error:
            body += f"[bold red]Error:[/bold red] {q.run_error}\n"

        if q.agent_thinking:
            body += f"\n[dim italic]Agent Reasoning:[/dim italic]\n[dim]{q.agent_thinking.strip()}[/dim]\n"

        if show_trajectory and q.tool_calls:
            body += f"\n[bold cyan]Recorded Execution Trajectory ({len(q.tool_calls)} step{'s' if len(q.tool_calls) != 1 else ''}):[/bold cyan]\n"
            for idx, tc in enumerate(q.tool_calls, 1):
                name = tc.get("tool") or tc.get("name") or "tool"
                args = tc.get("args") or tc.get("arguments") or {}
                args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
                body += f"  [bold cyan]Step {idx}:[/bold cyan] [yellow]{name}[/yellow]({args_str})\n"

                if idx - 1 < len(q.tool_results):
                    res_item = q.tool_results[idx - 1]
                    content = str(res_item.get("content") or res_item.get("result") or res_item.get("output") or "")
                    preview = content.replace("\n", " ")
                    if len(preview) > 140:
                        preview = preview[:140] + "..."
                    body += f"    [dim]↳ Result: {preview}[/dim]\n"
    else:
        body += "[dim]Not executed yet.[/dim]\n"

    body += "\n[bold blue]Grader Evaluation & Comment:[/bold blue]\n"
    if q.has_score:
        body += f"[bold]Verdict:[/bold] [{q.status_style}]{q.correctness}[/{q.status_style}]"
        if q.numeric_match is not None:
            body += f" | [bold]Numeric Match:[/bold] {q.numeric_match}"
        if q.groundedness:
            body += f" | [bold]Groundedness:[/bold] {q.groundedness}"
        if q.error_category:
            body += f" | [bold]Error Category:[/bold] {q.error_category}"
        body += f"\n[bold]Grader Comment:[/bold] {q.rationale or '(none)'}\n"
        if q.judge_llm:
            body += f"[dim]Judge LLM: {q.judge_llm}[/dim]\n"
    else:
        body += "[dim]No grader evaluation recorded.[/dim]\n"

    panel = Panel(
        body,
        title=f"[bold cyan]{q.id}[/bold cyan] ({status_styled})",
        subtitle=f"Doc: [yellow]{doc_str}[/yellow]",
        border_style="cyan",
        expand=False,
    )
    console.print(panel)


class BenchViewerApp(App):
    """Textual application for interactive benchmark inspection."""

    CSS = """
    Screen {
        layout: vertical;
        background: $surface;
    }
    #main-container {
        layout: horizontal;
        height: 1fr;
    }
    #left-panel {
        width: 42%;
        min-width: 40;
        border-right: vkey $primary;
        padding: 1;
        layout: vertical;
    }
    #right-panel {
        width: 58%;
        padding: 1;
        layout: vertical;
    }
    #summary-bar {
        height: auto;
        margin-bottom: 1;
        background: $panel;
        padding: 1;
        border: round $primary;
    }
    #filter-row {
        layout: horizontal;
        height: auto;
        margin-bottom: 1;
    }
    #search-input {
        width: 60%;
        margin-right: 1;
    }
    #status-select {
        width: 40%;
    }
    #questions-table {
        height: 1fr;
    }
    #detail-container {
        height: 1fr;
        border: round $primary;
        padding: 1 2;
        background: $panel;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("/", "focus_search", "Search", show=True),
        Binding("escape", "clear_search", "Back to Table", show=True),
        Binding("t", "toggle_tools", "Toggle Tool Calls", show=True),
        Binding("r", "refresh_data", "Reload Data", show=True),
    ]

    def __init__(
        self,
        cfg: BenchConfig,
        initial_question_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cfg = cfg
        self.initial_question_id = initial_question_id
        self.all_questions: list[BenchQuestionDetail] = []
        self.filtered_questions: list[BenchQuestionDetail] = []
        self.selected_question: BenchQuestionDetail | None = None
        self.show_tool_calls: bool = True

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-container"):
            with Vertical(id="left-panel"):
                yield Static(id="summary-bar")
                with Horizontal(id="filter-row"):
                    yield Input(placeholder="🔍 Search (ID, doc, text)...", id="search-input")
                    yield Select(
                        options=[
                            ("All Questions", "all"),
                            ("✓ Correct", "correct"),
                            ("~ Partial", "partial"),
                            ("✗ Incorrect", "incorrect"),
                            ("● Executed", "run"),
                            ("○ Unrun", "unrun"),
                        ],
                        value="all",
                        id="status-select",
                    )
                yield DataTable(id="questions-table", cursor_type="row")
            with Vertical(id="right-panel"):
                with VerticalScroll(id="detail-container"):
                    yield Markdown(id="detail-markdown")
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"Benchmark Viewer: {self.cfg.profile_name}"
        self.sub_title = "Browse questions, gold answers, agent traces, and evaluations"
        table = self.query_one(DataTable)
        table.add_columns("ID", "Status", "Calls", "Doc", "Question")
        self.load_data()

    def load_data(self) -> None:
        self.all_questions = load_bench_dataset_with_results(self.cfg)
        self.apply_filter()

    def update_summary(self) -> None:
        total = len(self.all_questions)
        correct = sum(1 for q in self.all_questions if q.correctness == "correct")
        partial = sum(1 for q in self.all_questions if q.correctness == "partial")
        incorrect = sum(1 for q in self.all_questions if q.correctness == "incorrect")
        run_count = sum(1 for q in self.all_questions if q.has_run)
        scored_count = sum(1 for q in self.all_questions if q.has_score)

        summary_widget = self.query_one("#summary-bar", Static)
        summary_widget.update(
            f"[bold cyan]Profile:[/bold cyan] {self.cfg.profile_name} | "
            f"[bold]Total:[/bold] {total} | "
            f"[green]✓ {correct}[/green] [yellow]~ {partial}[/yellow] [red]✗ {incorrect}[/red] | "
            f"[dim]Run: {run_count} / Scored: {scored_count}[/dim]"
        )

    def apply_filter(self) -> None:
        search_term = self.query_one("#search-input", Input).value.strip().lower()
        status_filter = self.query_one("#status-select", Select).value

        results: list[BenchQuestionDetail] = []
        for q in self.all_questions:
            # Status filter
            if status_filter == "correct" and q.correctness != "correct":
                continue
            if status_filter == "partial" and q.correctness != "partial":
                continue
            if status_filter == "incorrect" and q.correctness != "incorrect":
                continue
            if status_filter == "run" and not q.has_run:
                continue
            if status_filter == "unrun" and q.has_run:
                continue

            # Text filter
            if search_term:
                match = (
                    search_term in q.id.lower()
                    or search_term in q.doc_name.lower()
                    or any(search_term in d.lower() for d in q.doc_names)
                    or search_term in q.question.lower()
                    or search_term in q.gold_answer.lower()
                    or (q.agent_answer and search_term in q.agent_answer.lower())
                    or (q.rationale and search_term in q.rationale.lower())
                )
                if not match:
                    continue

            results.append(q)

        self.filtered_questions = results
        self.populate_table()
        self.update_summary()

    def populate_table(self) -> None:
        table = self.query_one(DataTable)
        table.clear()

        initial_row_index = 0
        for idx, q in enumerate(self.filtered_questions):
            status_text = q.status_label
            q_snip = (q.question[:45] + "...") if len(q.question) > 45 else q.question
            doc_snip = q.doc_name or (q.doc_names[0] if q.doc_names else "")
            calls_text = str(q.n_tool_calls) if q.has_run or q.n_tool_calls > 0 else "-"
            table.add_row(q.id, status_text, calls_text, doc_snip, q_snip, key=q.id)

            if self.initial_question_id and q.id.lower() == self.initial_question_id.lower():
                initial_row_index = idx

        if self.filtered_questions:
            target_idx = min(initial_row_index, len(self.filtered_questions) - 1)
            table.move_cursor(row=target_idx)
            self.selected_question = self.filtered_questions[target_idx]
            self.render_detail(self.selected_question)
        else:
            self.selected_question = None
            md = self.query_one("#detail-markdown", Markdown)
            md.update("*No matching questions found.*")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        row_key = event.row_key.value
        for q in self.filtered_questions:
            if q.id == row_key:
                self.selected_question = q
                self.render_detail(q)
                break

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key:
            row_key = event.row_key.value
            for q in self.filtered_questions:
                if q.id == row_key:
                    self.selected_question = q
                    self.render_detail(q)
                    break

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "search-input":
            self.apply_filter()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "status-select":
            self.apply_filter()

    def render_detail(self, q: BenchQuestionDetail) -> None:
        md = self.query_one("#detail-markdown", Markdown)
        content = format_question_markdown(q, show_tool_calls=self.show_tool_calls)
        md.update(content)

    def action_focus_search(self) -> None:
        self.query_one("#search-input", Input).focus()

    def action_clear_search(self) -> None:
        search_input = self.query_one("#search-input", Input)
        if search_input.has_focus:
            self.query_one(DataTable).focus()
        else:
            search_input.value = ""
            self.apply_filter()

    def action_toggle_tools(self) -> None:
        self.show_tool_calls = not self.show_tool_calls
        if self.selected_question:
            self.render_detail(self.selected_question)
        status = "expanded (full output)" if self.show_tool_calls else "compact preview"
        self.notify(f"Trajectory view: {status}", timeout=2)

    def action_refresh_data(self) -> None:
        self.load_data()
        self.notify("Reloaded benchmark dataset & results from disk", timeout=2)


def run_bench_tui(cfg: BenchConfig, initial_question_id: str | None = None) -> None:
    """Launch the Textual interactive benchmark dataset browser."""
    app = BenchViewerApp(cfg, initial_question_id=initial_question_id)
    app.run()
