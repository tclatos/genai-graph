"""Adapter for MMLongBench-Doc (Multi-modal Long Document Benchmark).

Reference: https://github.com/mayubo2333/MMLongBench-Doc
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from genai_graph.bench.adapters.base import BaseBenchmarkAdapter
from genai_graph.bench.models import BenchQuestion

MMLONGBENCH_RUBRIC = """\
You are a strict-but-fair grader for MMLongBench-Doc, a benchmark evaluating multi-modal comprehension of very long documents.
Compare the agent's answer to the gold answer using the provided evidence and justification.

Return ONLY a JSON object with exactly these keys:
{
  "correctness": "correct" | "partial" | "incorrect",
  "numeric_match": true | false | null,
  "groundedness": "grounded" | "partial" | "ungrounded",
  "error_category": "missing_ocr_or_visual_chart" | "calculation_or_math_error" | "retrieval_or_lookup_error" | "halted_or_empty_response" | null,
  "rationale": "<one sentence>"
}

Tiers:
- "correct" = agent answer matches gold answer substance and requirements.
- "partial" = partially complete, right direction but missing key aspects or minor errors.
- "incorrect" = wrong fact, incorrect interpretation, or missing answer.
"""


class MMLongBenchDocAdapter(BaseBenchmarkAdapter):
    """Adapter for MMLongBench-Doc datasets."""

    name: str = "mmlongbench"

    def load_dataset(self, split: str | None = None, cache_dir: Path | None = None) -> list[BenchQuestion]:
        """Load questions from local JSON/JSONL or Hugging Face cache."""
        target_dir = cache_dir or (Path.cwd() / "data" / "mmlongbench")
        target_dir.mkdir(parents=True, exist_ok=True)
        q_file = target_dir / "questions.jsonl"

        questions: list[BenchQuestion] = []
        if q_file.exists():
            with q_file.open(encoding="utf-8") as fh:
                for idx, line in enumerate(fh):
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    q_id = str(item.get("id") or item.get("question_id") or f"mmlong_{idx}")
                    doc_name = str(item.get("doc_name") or item.get("file_name") or "unknown")
                    questions.append(
                        BenchQuestion(
                            id=q_id,
                            doc_name=doc_name,
                            doc_names=[doc_name],
                            question=str(item.get("question", "")),
                            gold_answer=str(item.get("answer") or item.get("gold_answer", "")),
                            evidence=item.get("evidence", []),
                            justification=item.get("justification"),
                            metadata=item,
                        )
                    )
            return questions

        logger.info("MMLongBench questions.jsonl not found at {}. Returning empty list.", q_file)
        return []

    def fetch_document(self, doc_name: str, output_dir: Path) -> Path:
        """Locate or fetch MMLongBench document."""
        output_dir.mkdir(parents=True, exist_ok=True)
        norm = self.resolve_doc_name(doc_name)
        target = output_dir / f"{norm}.pdf"
        if target.exists():
            return target
        logger.warning("Document {} not present in {}.", norm, output_dir)
        return target

    def get_judge_rubric(self) -> str:
        return MMLONGBENCH_RUBRIC
