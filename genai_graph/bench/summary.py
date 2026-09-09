"""Summary metrics aggregation and terminal/file reporting for benchmarks."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from genai_graph.bench.models import BenchScoreRecord, BenchSummary

console = Console()


def compute_bench_summary(
    scores: list[BenchScoreRecord | dict[str, Any]],
    profile_name: str = "default",
    agent_llm: str = "",
    judge_llm: str = "",
) -> BenchSummary:
    """Compute aggregate accuracy, numeric match, groundedness, and token stats."""
    total = len(scores)
    if total == 0:
        return BenchSummary(
            profile=profile_name,
            total_questions=0,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    correct = 0
    partial = 0
    incorrect = 0
    numeric_total = 0
    numeric_matched = 0
    grounded_count = 0
    error_counter: Counter[str] = Counter()
    total_tool_calls = 0
    total_in_tokens = 0
    total_out_tokens = 0

    for s in scores:
        data = s if isinstance(s, dict) else s.to_legacy_dict()
        corr = data.get("correctness")
        if corr == "correct":
            correct += 1
        elif corr == "partial":
            partial += 1
        else:
            incorrect += 1

        num = data.get("numeric_match")
        if num is True:
            numeric_total += 1
            numeric_matched += 1
        elif num is False:
            numeric_total += 1

        ground = data.get("groundedness")
        if ground == "grounded":
            grounded_count += 1
        elif ground == "partial":
            grounded_count += 0.5

        err = data.get("error_category")
        if err:
            error_counter[str(err)] += 1

        total_tool_calls += int(data.get("n_tool_calls") or len(data.get("tool_calls") or []))
        total_in_tokens += int(data.get("input_tokens") or 0)
        total_out_tokens += int(data.get("output_tokens") or 0)

        if not agent_llm and data.get("llm"):
            agent_llm = str(data.get("llm"))
        if not judge_llm and data.get("judge_llm"):
            judge_llm = str(data.get("judge_llm"))

    acc = round(correct / total, 4)
    part_acc = round((correct + 0.5 * partial) / total, 4)
    num_rate = round(numeric_matched / numeric_total, 4) if numeric_total > 0 else None
    ground_rate = round(grounded_count / total, 4)

    return BenchSummary(
        profile=profile_name,
        total_questions=total,
        correct=correct,
        partial=partial,
        incorrect=incorrect,
        accuracy=acc,
        partial_accuracy=part_acc,
        numeric_match_rate=num_rate,
        numeric_total=numeric_total,
        numeric_matched=numeric_matched,
        grounded_rate=ground_rate,
        error_breakdown=dict(error_counter),
        total_tool_calls=total_tool_calls,
        avg_tool_calls=round(total_tool_calls / total, 2),
        total_input_tokens=total_in_tokens,
        total_output_tokens=total_out_tokens,
        agent_llm=agent_llm,
        judge_llm=judge_llm,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def save_bench_summary(summary: BenchSummary, output_path: Path) -> None:
    """Save summary to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")


def display_bench_summary(summary: BenchSummary) -> None:
    """Print a rich terminal summary table of benchmark results."""
    table = Table(title=f"Benchmark Evaluation Summary: {summary.profile}", show_lines=True)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Questions", str(summary.total_questions))
    table.add_row("Correct", f"{summary.correct} ({summary.accuracy:.1%})")
    table.add_row("Partial", f"{summary.partial}")
    table.add_row("Incorrect", f"{summary.incorrect}")
    table.add_row("Strict Accuracy", f"{summary.accuracy:.1%}")
    table.add_row("Weighted Accuracy (corr + 0.5*part)", f"{summary.partial_accuracy:.1%}")

    if summary.numeric_match_rate is not None:
        table.add_row(
            "Numeric Match Rate",
            f"{summary.numeric_match_rate:.1%} ({summary.numeric_matched}/{summary.numeric_total})",
        )

    table.add_row("Groundedness Rate", f"{summary.grounded_rate:.1%}")
    table.add_row("Avg Tool Calls / Turn", str(summary.avg_tool_calls))
    table.add_row(
        "Total Tokens",
        f"In: {summary.total_input_tokens:,} | Out: {summary.total_output_tokens:,}",
    )
    table.add_row("Agent LLM", summary.agent_llm or "n/a")
    table.add_row("Judge LLM", summary.judge_llm or "n/a")

    console.print(table)

    if summary.error_breakdown:
        err_table = Table(title="Error Breakdown", show_lines=True)
        err_table.add_column("Error Category", style="yellow")
        err_table.add_column("Count", style="red")
        for cat, cnt in summary.error_breakdown.items():
            err_table.add_row(cat, str(cnt))
        console.print(err_table)
