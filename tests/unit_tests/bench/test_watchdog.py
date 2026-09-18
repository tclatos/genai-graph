"""Unit tests for the bench suspend/resume watchdog and tool-error counting."""

from __future__ import annotations

import time as real_time
import types

import pytest

import genai_graph.bench.watchdog as watchdog_module
from genai_graph.bench.runner import BenchRunRecord, count_error_tool_results
from genai_graph.bench.watchdog import SuspendWatchdog, check_suspend


class FakeTime:
    """Controllable stand-in for the ``time`` module inside the watchdog."""

    def __init__(self) -> None:
        self.wall = 1_000.0
        self.mono = 1_000.0

    def time(self) -> float:
        return self.wall

    def monotonic(self) -> float:
        return self.mono


def test_watchdog_quiet_on_steady_clocks():
    w = SuspendWatchdog(interval=0.02, threshold=1.0)
    w.start()
    try:
        real_time.sleep(0.1)
        assert not w.triggered
        check_suspend()  # no raise
    finally:
        w.stop()


def test_watchdog_detects_suspend():
    fake = FakeTime()
    monkey_time = types.SimpleNamespace(time=fake.time, monotonic=fake.monotonic)
    original = watchdog_module.time
    watchdog_module.time = monkey_time  # type: ignore[assignment]
    w = SuspendWatchdog(interval=0.01, threshold=10.0)
    w.start()
    try:
        fake.wall += 300.0  # machine suspended for 5 minutes
        deadline = real_time.time() + 2.0
        while not w.triggered and real_time.time() < deadline:
            real_time.sleep(0.01)
        assert w.triggered
        with pytest.raises(RuntimeError, match="suspend/resume"):
            check_suspend()
    finally:
        w.stop()
        watchdog_module.time = original  # type: ignore[assignment]


def test_check_suspend_noop_without_watchdog():
    # No active watchdog registered in this process (or stopped earlier).
    check_suspend()  # no raise


def _record(run_id: str, tool_results: list[dict]) -> BenchRunRecord:
    return BenchRunRecord(
        id=run_id,
        doc_name="doc.pdf",
        doc_names=["doc.pdf"],
        question="q",
        gold_answer="a",
        agent_answer="ans",
        tool_calls=[],
        tool_results=tool_results,
        n_tool_calls=0,
        input_tokens=0,
        output_tokens=0,
        error=None,
        thread_id=run_id,
        llm="test",
        started_at=None,
    )


def test_count_error_tool_results():
    records = [
        _record("q1", [{"tool": "get_section_content", "content": "Error: buffer pool is full and no memory could be freed"}]),
        _record("q2", [{"tool": "get_section_content", "content": "### [s0] Title\n\nNormal text"}]),
        _record("q3", [{"tool": "query_image", "content": "Error analyzing image"}, {"tool": "search_sections", "content": "Traceback (most recent call last):"}]),
    ]
    stats = count_error_tool_results(records)
    assert stats["runs_with_errors"] == 2
    assert stats["total_error_results"] == 3
    assert stats["run_ids"] == ["q1", "q3"]
    assert stats["by_marker"]["Error:"] == 1
    assert stats["by_marker"]["buffer pool"] == 1
    assert stats["by_marker"]["Traceback"] == 1


def test_count_error_tool_results_empty():
    stats = count_error_tool_results([])
    assert stats["runs_with_errors"] == 0
    assert stats["total_error_results"] == 0
