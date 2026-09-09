"""OfficeQA dataset adapter for US Treasury Bulletins and government finance reports."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from loguru import logger

from genai_graph.bench.adapters.base import BaseBenchmarkAdapter
from genai_graph.bench.models import BenchQuestion

DATASET_ID = "databricks/officeqa"

OFFICEQA_JUDGE_RUBRIC = """\
You are a strict-but-fair grader for OfficeQA, a benchmark on US Treasury Bulletins and government financial reports.
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


def _parse_source_files(val: str | None) -> list[str]:
    """Parse source_files string into a list of normalized doc stems."""
    if not val or not isinstance(val, str):
        return []
    lines = [line.strip() for line in val.replace("\r", "\n").split("\n") if line.strip()]
    cleaned: list[str] = []
    for line in lines:
        s = line.strip()
        if s.endswith((".txt", ".pdf", ".md")):
            cleaned.append(Path(s).stem)
        elif s:
            cleaned.append(s)
    return cleaned


class OfficeQAAdapter(BaseBenchmarkAdapter):
    """Adapter for databricks/officeqa."""

    name: str = "officeqa"

    def load_dataset(self, split: str | None = "pro", cache_dir: Path | None = None) -> list[BenchQuestion]:
        """Load Databricks OfficeQA dataset and convert to BenchQuestion models."""
        target_dir = cache_dir or (Path.cwd() / "data" / "officeqa")
        target_dir.mkdir(parents=True, exist_ok=True)
        split_name = "pro" if split in ("pro", None) else "full"
        cache_file = target_dir / f"officeqa_{split_name}.parquet"

        if cache_file.exists():
            logger.info("Loading cached OfficeQA dataset from {}", cache_file)
            df = pd.read_parquet(cache_file)
        else:
            csv_name = "officeqa_pro.csv" if split_name == "pro" else "officeqa_full.csv"
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            logger.info("Downloading {}/{} from Hugging Face", DATASET_ID, csv_name)

            csv_path = hf_hub_download(
                repo_id=DATASET_ID,
                filename=csv_name,
                repo_type="dataset",
                token=token,
            )
            df = pd.read_csv(csv_path)
            df.to_parquet(cache_file, index=False)
            logger.info("Cached {} rows to {}", len(df), cache_file)

        questions: list[BenchQuestion] = []
        for idx, row in df.iterrows():
            q_id = str(row.get("officeqa_id") or row.get("id") or f"officeqa_{idx}")
            raw_sources = row.get("source_files")
            doc_names = _parse_source_files(raw_sources)
            doc_name = doc_names[0] if doc_names else "unknown"
            q_text = str(row.get("question", ""))
            gold_ans = str(row.get("answer", ""))
            justification = row.get("justification") if pd.notna(row.get("justification")) else None
            evidence = [row.get("evidence")] if pd.notna(row.get("evidence")) else []

            metadata = {
                "officeqa_id": q_id,
                "financebench_id": q_id,  # backward compatibility alias
                "source_files": raw_sources,
                "doc_names": doc_names,
                "difficulty": row.get("difficulty") if pd.notna(row.get("difficulty")) else None,
            }

            questions.append(
                BenchQuestion(
                    id=q_id,
                    doc_name=doc_name,
                    doc_names=doc_names,
                    question=q_text,
                    gold_answer=gold_ans,
                    evidence=evidence,
                    justification=justification,
                    metadata=metadata,
                )
            )
        return questions

    def fetch_document(self, doc_name: str, output_dir: Path) -> Path:
        """Download document from OfficeQA / Hugging Face documents repository."""
        output_dir.mkdir(parents=True, exist_ok=True)
        norm_name = self.resolve_doc_name(doc_name)
        target = output_dir / f"{norm_name}.pdf"
        if target.exists() and target.stat().st_size > 0:
            return target

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        try:
            downloaded = hf_hub_download(
                repo_id=DATASET_ID,
                filename=f"documents/{norm_name}.pdf",
                repo_type="dataset",
                token=token,
                local_dir=str(output_dir),
            )
            return Path(downloaded)
        except Exception as exc:
            logger.warning("Could not fetch PDF from HF for {}: {}. Creating placeholder.", norm_name, exc)
            target.write_text(f"Placeholder for {norm_name}", encoding="utf-8")
            return target

    def get_judge_rubric(self) -> str:
        return OFFICEQA_JUDGE_RUBRIC
