"""Unit tests for the reusable Benchmark Outcomes browser component."""

from __future__ import annotations

from pathlib import Path

from genai_graph.ui.streamlit.components.bench_browser import (
    _compute_bench_metrics,
    discover_bench_profiles,
)
from genai_graph.ui.textual.apps.bench_tui import BenchQuestionDetail


def test_compute_bench_metrics_empty() -> None:
    """Empty list returns zeroed metrics without ZeroDivisionError."""
    metrics = _compute_bench_metrics([])
    assert metrics["total"] == 0
    assert metrics["strict_acc"] == 0.0
    assert metrics["weighted_acc"] == 0.0
    assert metrics["groundedness"] == 0.0


def test_compute_bench_metrics_mixed() -> None:
    """Correctly computes strict, weighted, and tool averages on mixed questions."""
    q1 = BenchQuestionDetail(
        id="q1",
        question="What is X?",
        gold_answer="X is 1",
        has_run=True,
        has_score=True,
        correctness="correct",
        groundedness="grounded",
        n_tool_calls=2,
        input_tokens=100,
        output_tokens=50,
    )
    q2 = BenchQuestionDetail(
        id="q2",
        question="What is Y?",
        gold_answer="Y is 2",
        has_run=True,
        has_score=True,
        correctness="partial",
        groundedness="grounded",
        n_tool_calls=4,
        input_tokens=200,
        output_tokens=100,
    )
    q3 = BenchQuestionDetail(
        id="q3",
        question="What is Z?",
        gold_answer="Z is 3",
        has_run=True,
        has_score=True,
        correctness="incorrect",
        groundedness="ungrounded",
        n_tool_calls=6,
        input_tokens=300,
        output_tokens=150,
    )
    q4 = BenchQuestionDetail(
        id="q4",
        question="What is W?",
        gold_answer="W is 4",
        has_run=False,
        has_score=False,
    )

    items = [q1, q2, q3, q4]
    m = _compute_bench_metrics(items)

    assert m["total"] == 4
    assert m["evaluated"] == 3
    assert m["correct"] == 1
    assert m["partial"] == 1
    assert m["incorrect"] == 1
    assert m["unrun"] == 1

    # strict_acc = 1 / 3 = 33.33%
    assert abs(m["strict_acc"] - 33.33) < 0.1
    # weighted_acc = (1 + 0.5) / 3 = 50.0%
    assert abs(m["weighted_acc"] - 50.0) < 0.1
    # groundedness = 2 / 3 = 66.67%
    assert abs(m["groundedness"] - 66.67) < 0.1
    # avg_tools = (2 + 4 + 6) / 3 = 4.0
    assert m["avg_tools"] == 4.0


def test_discover_bench_profiles(tmp_path: Path) -> None:
    """Discovers configured profile keys from YAML config."""
    yaml_content = """
profiles:
  decision_fast:
    agent_llm: glm_5.3_flash
  default:
    agent_llm: gpt_4o
"""
    cfg_file = tmp_path / "bench.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    profiles = discover_bench_profiles(cfg_file)
    assert "decision_fast" in profiles
    assert "default" in profiles
