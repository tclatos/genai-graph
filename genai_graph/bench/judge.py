"""LLM-as-judge evaluation engine for benchmark responses.

Compares agent answers against gold answers and evidence, applying domain-specific
rubrics (FinanceBench, OfficeQA, etc.) and returning structured verdicts.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from genai_tk.core.factories.llm_factory import get_llm
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from genai_graph.bench.models import BenchRunRecord, BenchScoreRecord, JudgeVerdict

_SCORES_WRITE_LOCK = threading.Lock()


def _format_user_prompt(run: BenchRunRecord | dict[str, Any]) -> str:
    """Format the question, gold answer, evidence, and agent response for evaluation."""
    r = run if isinstance(run, dict) else run.to_legacy_dict()
    q_text = r.get("question", "")
    gold_answer = r.get("gold_answer") or r.get("answer", "")
    justification = r.get("justification") or ""
    evidence = r.get("evidence") or []
    agent_answer = r.get("agent_answer", "")

    evidence_str = "\n".join(f"- {e}" for e in evidence) if evidence else "(none provided)"
    return f"""\
QUESTION:
{q_text}

GOLD ANSWER:
{gold_answer}

GOLD JUSTIFICATION:
{justification or "(none provided)"}

GOLD EVIDENCE:
{evidence_str}

AGENT ANSWER:
{agent_answer or "(empty response)"}
"""


def _parse_judge_json(raw: str) -> dict[str, Any]:
    """Extract and parse JSON object from LLM output."""
    cleaned = raw.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


async def evaluate_single_run(
    judge_llm: str,
    run: BenchRunRecord | dict[str, Any],
    system_rubric: str,
    max_retries: int = 5,
) -> BenchScoreRecord:
    """Grade a single execution run with the judge LLM and return BenchScoreRecord."""
    model = get_llm(judge_llm)
    user_prompt = _format_user_prompt(run)
    messages = [
        SystemMessage(content=system_rubric),
        HumanMessage(content=user_prompt),
    ]

    r_record = (
        run
        if isinstance(run, BenchRunRecord)
        else BenchRunRecord(
            id=str(run.get("id") or run.get("financebench_id") or run.get("officeqa_id") or ""),
            doc_name=str(run.get("doc_name", "")),
            doc_names=list(run.get("doc_names") or []),
            question=str(run.get("question", "")),
            gold_answer=str(run.get("gold_answer") or run.get("answer", "")),
            agent_answer=str(run.get("agent_answer", "")),
            agent_thinking=run.get("agent_thinking"),
            justification=run.get("justification"),
            evidence=run.get("evidence") or [],
            tool_calls=run.get("tool_calls") or [],
            tool_results=run.get("tool_results") or [],
            n_tool_calls=int(run.get("n_tool_calls") or 0),
            input_tokens=int(run.get("input_tokens") or 0),
            output_tokens=int(run.get("output_tokens") or 0),
            error=run.get("error"),
            thread_id=run.get("thread_id"),
            llm=str(run.get("llm", "")),
            started_at=run.get("started_at"),
        )
    )

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = await model.ainvoke(messages)
            content = resp.content if hasattr(resp, "content") else str(resp)
            if isinstance(content, list):
                content = "".join(str(c) for c in content)

            data = _parse_judge_json(str(content))
            verdict = JudgeVerdict.model_validate(data)
            return BenchScoreRecord(
                run=r_record,
                verdict=verdict,
                judge_llm=judge_llm,
                scored_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "[{}] Judge evaluation attempt {}/{} failed: {}",
                r_record.id,
                attempt,
                max_retries,
                exc,
            )
            if attempt < max_retries:
                await asyncio.sleep(2**attempt)

    # Fallback verdict if grading repeatedly fails
    logger.error("[{}] All judge attempts failed. Marking incorrect.", r_record.id)
    return BenchScoreRecord(
        run=r_record,
        verdict=JudgeVerdict(
            correctness="incorrect",
            numeric_match=False,
            groundedness="ungrounded",
            error_category="halted_or_empty_response",
            rationale=f"Judge evaluation failed after {max_retries} attempts: {last_exc}",
        ),
        judge_llm=judge_llm,
        scored_at=datetime.now(timezone.utc).isoformat(),
    )


def append_score_record(score: BenchScoreRecord, output_file: Path) -> None:
    """Thread-safe append of a scored record to a JSONL file."""
    with _SCORES_WRITE_LOCK:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(score.to_legacy_dict(), ensure_ascii=False) + "\n")


def load_existing_scores(scores_path: Path) -> dict[str, dict[str, Any]]:
    """Load scored rows from JSONL file indexed by question ID."""
    if not scores_path.exists():
        return {}
    scores: dict[str, dict[str, Any]] = {}
    with scores_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                q_id = str(data.get("id") or data.get("financebench_id") or data.get("officeqa_id") or "")
                if q_id:
                    scores[q_id] = data
            except Exception as exc:
                logger.warning("Failed to parse score row in {}: {}", scores_path, exc)
    return scores
