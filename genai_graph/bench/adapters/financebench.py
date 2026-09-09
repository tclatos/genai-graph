"""FinanceBench dataset adapter for SEC financial filings."""

from __future__ import annotations

import math
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from loguru import logger

from genai_graph.bench.adapters.base import BaseBenchmarkAdapter
from genai_graph.bench.models import BenchQuestion

DATASET_ID = "PatronusAI/financebench"
PDF_RAW_BASE = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs"

FINANCEBENCH_JUDGE_RUBRIC = """\
You are a strict-but-fair grader for FinanceBench, a benchmark of financial
questions answered from SEC 10-K, 10-Q, 8-K filings and earnings reports.
Compare the agent's answer to the gold answer using the gold evidence and
justification. Financial answers are often a number or a short factual claim.

Return ONLY a JSON object with exactly these keys:
{
  "correctness": "correct" | "partial" | "incorrect",
  "numeric_match": true | false | null,
  "groundedness": "grounded" | "partial" | "ungrounded",
  "error_category": "missing_ocr_or_visual_chart" | "calculation_or_math_error" | "retrieval_or_lookup_error" | "halted_or_empty_response" | null,
  "rationale": "<one sentence>"
}

Equivalence rules (adopted from Mafin2.5):
- Numerical accuracy: rounding differences are IGNORED when they do not change
  the conclusion. Allow flexibility: 1.2 is similar to 1.23 (one rounds to the
  other). Fractions, percentages, and decimals can be equivalent: "11 of 14" is
  equivalent to 79% and to 0.79.
- The agent answer is CORRECT if the gold answer, or any of its equivalences,
  can be INFERRED or generated from the agent's answer, or implicitly exists in
  it.
- If the agent answer is a SUPERSET of the gold answer, it is correct.
- If the agent answer conveys the same or similar meaning, conclusion, or
  rationale as the gold, it is correct.
- A reasonable alternative interpretation (justifiable vs the gold) is correct.
- Otherwise it is incorrect.

Tiers:
- "correct" = the agent answer matches the gold answer's substance under the
  equivalence rules above (number within rounding/fraction equivalence, or same
  factual claim). "partial" = right direction but wrong value/units, incomplete,
  or only partly substantiated. "incorrect" = wrong or missing.
- "numeric_match" = true if a number was expected and the agent's number matches
  the gold under the equivalence rules (rounding/fraction/percent); false if a
  number was expected and it does not match; null if no specific number expected.
- "groundedness" = whether the agent's answer is supported by the cited/source
  text rather than invented. "ungrounded" if it states facts not in the filing.
- "error_category" = when correctness is "partial" or "incorrect", categorize the primary root cause:
  * "missing_ocr_or_visual_chart": question requires reading a visual chart, line plot, graph, or diagram missing/unreadable in text OCR transcript.
  * "calculation_or_math_error": agent found the relevant figures, but made an arithmetic, formula, or rounding calculation mistake.
  * "retrieval_or_lookup_error": agent retrieved or referenced the wrong table, row, date, or failed to find the relevant section.
  * "halted_or_empty_response": agent timed out, looped, hit tool recursion limits, or returned an empty/aborted response.
"""


def _clean(value: Any) -> Any:
    """Return JSON-safe value: pandas NaN -> None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


class FinanceBenchAdapter(BaseBenchmarkAdapter):
    """Adapter for PatronusAI/financebench."""

    name: str = "financebench"

    def load_dataset(self, split: str | None = None, cache_dir: Path | None = None) -> list[BenchQuestion]:
        """Load PatronusAI/financebench and convert to BenchQuestion items."""
        target_dir = cache_dir or (Path.cwd() / "data" / "financebench")
        target_dir.mkdir(parents=True, exist_ok=True)
        parquet_cache = target_dir / "financebench_merged.parquet"

        if parquet_cache.exists():
            logger.info("Loading cached FinanceBench dataset from {}", parquet_cache)
            df = pd.read_parquet(parquet_cache)
        else:
            logger.info("Downloading {} from Hugging Face", DATASET_ID)
            ds = load_dataset(DATASET_ID, split="train")
            df = ds.to_pandas()
            df.to_parquet(parquet_cache, index=False)
            logger.info("Cached {} rows to {}", len(df), parquet_cache)

        questions: list[BenchQuestion] = []
        for _, row in df.iterrows():
            q_id = str(row.get("financebench_id", ""))
            doc_name = str(row.get("doc_name", ""))
            q_text = str(row.get("question", ""))
            gold_ans = str(row.get("answer", ""))
            justification = _clean(row.get("justification"))
            raw_evidence = row.get("evidence")
            if raw_evidence is None or (isinstance(raw_evidence, float) and math.isnan(raw_evidence)):
                evidence = []
            elif hasattr(raw_evidence, "__len__") and len(raw_evidence) == 0:
                evidence = []
            elif hasattr(raw_evidence, "tolist"):
                evidence = raw_evidence.tolist()
            else:
                evidence = [raw_evidence] if raw_evidence else []

            metadata = {
                "financebench_id": q_id,
                "company": _clean(row.get("company")),
                "doc_type": _clean(row.get("doc_type")),
                "doc_period": _clean(row.get("doc_period")),
                "gics_sector": _clean(row.get("gics_sector")),
                "question_type": _clean(row.get("question_type")),
                "question_reasoning": _clean(row.get("question_reasoning")),
                "evidence_text": _clean(row.get("evidence_text")),
                "evidence_doc_name": _clean(row.get("evidence_doc_name")),
                "evidence_page_num": _clean(row.get("evidence_page_num")),
                "evidence_text_full_page": _clean(row.get("evidence_text_full_page")),
            }
            questions.append(
                BenchQuestion(
                    id=q_id,
                    doc_name=doc_name,
                    doc_names=[doc_name] if doc_name else [],
                    question=q_text,
                    gold_answer=gold_ans,
                    evidence=evidence,
                    justification=justification,
                    metadata=metadata,
                )
            )
        return questions

    def fetch_document(self, doc_name: str, output_dir: Path) -> Path:
        """Download document PDF from the FinanceBench GitHub repository."""
        output_dir.mkdir(parents=True, exist_ok=True)
        norm_name = self.resolve_doc_name(doc_name)
        target = output_dir / f"{norm_name}.pdf"
        if target.exists() and target.stat().st_size > 0:
            logger.debug("PDF already present: {}", target)
            return target

        url = f"{PDF_RAW_BASE}/{norm_name}.pdf"
        logger.info("Downloading PDF for {} from {}", norm_name, url)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp, target.open("wb") as fh:
                shutil.copyfileobj(resp, fh)
        except Exception as exc:
            if target.exists():
                target.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to download PDF for {norm_name} from {url}: {exc}") from exc

        logger.info("Saved PDF ({} bytes) -> {}", target.stat().st_size, target)
        return target

    def get_judge_rubric(self) -> str:
        return FINANCEBENCH_JUDGE_RUBRIC
