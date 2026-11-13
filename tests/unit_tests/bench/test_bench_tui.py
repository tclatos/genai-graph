"""Unit tests for bench TUI and dataset loading with results."""

from __future__ import annotations

import json
from pathlib import Path

from genai_graph.bench.adapters.base import BaseBenchmarkAdapter
from genai_graph.bench.config import BenchConfig, BenchFilesConfig
from genai_graph.bench.models import BenchQuestion
from genai_graph.ui.textual.apps.bench_tui import load_bench_dataset_with_results


class DummyAdapter(BaseBenchmarkAdapter):
    """Dummy adapter for testing question filtering."""

    name: str = "dummy"

    def load_dataset(self, split: str | None = None, cache_dir: Path | None = None) -> list[BenchQuestion]:
        return [
            BenchQuestion(
                id="Q1",
                doc_name="doc_a",
                doc_names=["doc_a"],
                question="Question about doc A?",
                gold_answer="Gold A",
            ),
            BenchQuestion(
                id="Q2",
                doc_name="doc_b",
                doc_names=["doc_b"],
                question="Question about doc B?",
                gold_answer="Gold B",
            ),
            BenchQuestion(
                id="Q3",
                doc_name="doc_c",
                doc_names=["doc_c"],
                question="Question about doc C?",
                gold_answer="Gold C",
            ),
        ]

    def fetch_document(self, doc_name: str, output_dir: Path) -> Path:
        return output_dir / f"{doc_name}.pdf"

    def get_judge_rubric(self) -> str:
        return "Dummy rubric"


def test_load_bench_dataset_filtering(tmp_path: Path, monkeypatch) -> None:
    """Test that load_bench_dataset_with_results filters questions by files.docs."""
    import genai_graph.ui.textual.apps.bench_tui as tui_module

    monkeypatch.setattr(tui_module, "resolve_benchmark_adapter", lambda adapter_id, **kw: DummyAdapter())

    runs_file = tmp_path / "runs.jsonl"
    runs_file.parent.mkdir(parents=True, exist_ok=True)
    # Q2 has a run record even if not in docs
    run_rec = {
        "id": "Q2",
        "doc_name": "doc_b",
        "question": "Question about doc B?",
        "agent_answer": "Agent answer B",
    }
    runs_file.write_text(json.dumps(run_rec) + "\n", encoding="utf-8")

    scores_file = tmp_path / "scores.jsonl"
    scores_file.write_text("", encoding="utf-8")

    # Config restricts docs to only ["doc_a"]
    cfg = BenchConfig(
        profile_name="test_prof",
        dataset_adapter="dummy",
        files=BenchFilesConfig(docs=["doc_a"]),
        runs=str(runs_file),
        scores=str(scores_file),
        project_root=tmp_path,
    )

    items = load_bench_dataset_with_results(cfg)
    # Q1 is included because doc_a is in files.docs
    # Q2 is included because it has an existing run record in runs.jsonl
    # Q3 is excluded because doc_c is not in files.docs and has no run/score
    ids = [item.id for item in items]
    assert "Q1" in ids
    assert "Q2" in ids
    assert "Q3" not in ids
    assert len(items) == 2


def test_load_bench_dataset_specific_question(tmp_path: Path, monkeypatch) -> None:
    """Test that passing question_id bypasses files.docs filtering."""
    import genai_graph.ui.textual.apps.bench_tui as tui_module

    monkeypatch.setattr(tui_module, "resolve_benchmark_adapter", lambda adapter_id, **kw: DummyAdapter())

    runs_file = tmp_path / "runs.jsonl"
    runs_file.write_text("", encoding="utf-8")
    scores_file = tmp_path / "scores.jsonl"
    scores_file.write_text("", encoding="utf-8")

    cfg = BenchConfig(
        profile_name="test_prof",
        dataset_adapter="dummy",
        files=BenchFilesConfig(docs=["doc_a"]),
        runs=str(runs_file),
        scores=str(scores_file),
        project_root=tmp_path,
    )

    items = load_bench_dataset_with_results(cfg, question_id="Q3")
    assert len(items) == 1
    assert items[0].id == "Q3"
