"""Question execution runner for docgraph deep agents.

Runs questions through the Document Graph agent, streaming tokens and capturing
tool invocations, thinking, token metrics, and execution status into JSONL run records.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from genai_graph.bench.models import BenchQuestion, BenchRunRecord

_RUNS_WRITE_LOCK = threading.Lock()


async def run_one_question(
    harness: Any,
    q: BenchQuestion | dict[str, Any],
    llm: str,
) -> BenchRunRecord:
    """Stream one question through the agent harness and return a structured BenchRunRecord."""
    from genai_tk.agents.harness import (
        EndEvent,
        ErrorEvent,
        ThinkingEvent,
        TokenEvent,
        ToolCallEvent,
        ToolResultEvent,
        UsageEvent,
    )
    from genai_tk.core.messages import strip_reasoning_tags

    q_dict = q if isinstance(q, dict) else q.to_legacy_dict()
    q_id = str(q_dict.get("id") or q_dict.get("financebench_id") or q_dict.get("officeqa_id") or "")
    q_text = str(q_dict.get("question", ""))
    gold_answer = str(q_dict.get("gold_answer") or q_dict.get("answer", ""))
    doc_name = str(q_dict.get("doc_name", ""))
    doc_names = list(q_dict.get("doc_names") or ([doc_name] if doc_name else []))

    all_tokens: list[str] = []
    final_turn_tokens: list[str] = []
    thinking_tokens: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    input_tokens = 0
    output_tokens = 0
    error: str | None = None
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        async for event in harness.astream(q_text, thread_id=q_id):
            if isinstance(event, TokenEvent):
                all_tokens.append(event.text)
                final_turn_tokens.append(event.text)
            elif isinstance(event, ThinkingEvent):
                thinking_tokens.append(event.text)
            elif isinstance(event, ToolCallEvent):
                tool_calls.append({"tool": event.tool_name, "args": event.args})
                final_turn_tokens = []
            elif isinstance(event, ToolResultEvent):
                tool_results.append({"tool": event.tool_name, "content": (event.content or "")[:1500]})
            elif isinstance(event, UsageEvent):
                input_tokens += event.input_tokens
                output_tokens += event.output_tokens
            elif isinstance(event, ErrorEvent):
                error = event.message
            elif isinstance(event, EndEvent):
                pass
    except Exception as exc:
        error = str(exc)

    final_text = "".join(final_turn_tokens).strip()
    if not final_text:
        final_text = "".join(all_tokens).strip()
    final_text = strip_reasoning_tags(final_text)

    return BenchRunRecord(
        id=q_id,
        doc_name=doc_name,
        doc_names=doc_names,
        question=q_text,
        gold_answer=gold_answer,
        agent_answer=final_text,
        agent_thinking="".join(thinking_tokens).strip() or None,
        justification=q_dict.get("justification"),
        evidence=q_dict.get("evidence") or [],
        tool_calls=tool_calls,
        tool_results=tool_results,
        n_tool_calls=len(tool_calls),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        error=error,
        thread_id=q_id,
        llm=llm,
        started_at=started_at,
        metadata={
            k: v
            for k, v in q_dict.items()
            if k
            not in (
                "id",
                "financebench_id",
                "officeqa_id",
                "question_id",
                "question",
                "answer",
                "gold_answer",
                "justification",
                "evidence",
                "doc_name",
                "doc_names",
            )
        },
    )


def append_run_record(record: BenchRunRecord, output_file: Path) -> None:
    """Thread-safe append of a run record to a JSONL file."""
    with _RUNS_WRITE_LOCK:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_legacy_dict(), ensure_ascii=False) + "\n")


def load_existing_runs(runs_path: Path) -> dict[str, BenchRunRecord]:
    """Load existing run records from JSONL file indexed by question ID."""
    if not runs_path.exists():
        return {}
    records: dict[str, BenchRunRecord] = {}
    with runs_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                q_id = str(data.get("id") or data.get("financebench_id") or data.get("officeqa_id") or "")
                if q_id:
                    records[q_id] = BenchRunRecord(
                        id=q_id,
                        doc_name=str(data.get("doc_name", "")),
                        doc_names=list(data.get("doc_names") or []),
                        question=str(data.get("question", "")),
                        gold_answer=str(data.get("gold_answer") or data.get("answer", "")),
                        agent_answer=str(data.get("agent_answer", "")),
                        agent_thinking=data.get("agent_thinking"),
                        justification=data.get("justification"),
                        evidence=data.get("evidence") or [],
                        tool_calls=data.get("tool_calls") or [],
                        tool_results=data.get("tool_results") or [],
                        n_tool_calls=int(data.get("n_tool_calls") or len(data.get("tool_calls") or [])),
                        input_tokens=int(data.get("input_tokens") or 0),
                        output_tokens=int(data.get("output_tokens") or 0),
                        error=data.get("error"),
                        thread_id=data.get("thread_id"),
                        llm=str(data.get("llm", "")),
                        started_at=data.get("started_at"),
                        metadata={
                            k: v
                            for k, v in data.items()
                            if k
                            not in (
                                "id",
                                "financebench_id",
                                "officeqa_id",
                                "question",
                                "gold_answer",
                                "agent_answer",
                                "agent_thinking",
                                "tool_calls",
                                "tool_results",
                                "n_tool_calls",
                                "input_tokens",
                                "output_tokens",
                                "error",
                                "thread_id",
                                "llm",
                                "started_at",
                            )
                        },
                    )
            except Exception as exc:
                logger.warning("Failed to parse run line in {}: {}", runs_path, exc)
    return records
