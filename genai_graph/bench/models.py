"""Pydantic v2 data models for benchmark dataset questions, execution runs, and scoring."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ErrorCategory = Literal[
    "missing_ocr_or_visual_chart",
    "calculation_or_math_error",
    "retrieval_or_lookup_error",
    "halted_or_empty_response",
]


class BenchQuestion(BaseModel):
    """Unified representation of a benchmark question across datasets."""

    id: str = Field(description="Unique question identifier (e.g. financebench_id, officeqa_id)")
    question: str = Field(description="The question text")
    gold_answer: str = Field(description="The gold standard answer")
    doc_names: list[str] = Field(default_factory=list, description="Associated document name stems")
    doc_name: str = Field(default="", description="Primary document name stem (first in doc_names)")
    evidence: list[Any] = Field(default_factory=list, description="Gold evidence snippets or references")
    justification: str | None = Field(default=None, description="Gold justification text")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary dataset-specific metadata")

    def model_post_init(self, __context: Any) -> None:
        if not self.doc_name and self.doc_names:
            self.doc_name = self.doc_names[0]
        elif self.doc_name and not self.doc_names:
            self.doc_names = [self.doc_name]

    def to_legacy_dict(self) -> dict[str, Any]:
        """Convert to a legacy dictionary format compatible with original financebench/officeqa rows."""
        d: dict[str, Any] = {
            "financebench_id": self.id,
            "officeqa_id": self.id,
            "question_id": self.id,
            "id": self.id,
            "doc_name": self.doc_name,
            "doc_names": self.doc_names,
            "question": self.question,
            "answer": self.gold_answer,
            "gold_answer": self.gold_answer,
            "evidence": self.evidence,
            "justification": self.justification,
        }
        d.update(self.metadata)
        return d


class BenchRunRecord(BaseModel):
    """Execution record for a single question evaluated through the agent."""

    id: str = Field(description="Question ID")
    doc_name: str = Field(default="", description="Primary document name stem")
    doc_names: list[str] = Field(default_factory=list, description="All associated document name stems")
    question: str = Field(description="Question text")
    gold_answer: str = Field(description="Gold answer text")
    agent_answer: str = Field(default="", description="Final answer produced by agent")
    agent_thinking: str | None = Field(default=None, description="Reasoning / thinking tokens if available")
    justification: str | None = Field(default=None, description="Gold justification if present")
    evidence: list[Any] = Field(default_factory=list, description="Gold evidence if present")
    tool_calls: list[dict[str, Any]] = Field(default_factory=list, description="Tool calls made during the turn")
    tool_results: list[dict[str, Any]] = Field(default_factory=list, description="Tool execution results")
    n_tool_calls: int = Field(default=0, description="Total count of tool calls")
    input_tokens: int = Field(default=0, description="Prompt / input tokens consumed")
    output_tokens: int = Field(default=0, description="Completion / output tokens consumed")
    error: str | None = Field(default=None, description="Error message if execution failed")
    thread_id: str | None = Field(default=None, description="Harness thread identifier")
    llm: str = Field(default="", description="Agent LLM model ID")
    started_at: str | None = Field(default=None, description="ISO timestamp of run start")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Dataset specific metadata")

    def to_legacy_dict(self) -> dict[str, Any]:
        """Output legacy JSONL dictionary preserving both financebench and officeqa keys."""
        d = {
            "financebench_id": self.id,
            "officeqa_id": self.id,
            "question_id": self.id,
            "id": self.id,
            "doc_name": self.doc_name,
            "doc_names": self.doc_names,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "justification": self.justification,
            "evidence": self.evidence,
            "agent_answer": self.agent_answer,
            "agent_thinking": self.agent_thinking,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "n_tool_calls": self.n_tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "error": self.error,
            "thread_id": self.thread_id,
            "llm": self.llm,
            "started_at": self.started_at,
        }
        d.update(self.metadata)
        return d


class JudgeVerdict(BaseModel):
    """Structured evaluation verdict returned by the LLM-as-judge."""

    correctness: Literal["correct", "partial", "incorrect"]
    numeric_match: bool | None = None
    groundedness: Literal["grounded", "partial", "ungrounded"] = "partial"
    error_category: ErrorCategory | None = None
    rationale: str = ""


class BenchScoreRecord(BaseModel):
    """Full scored record joining execution trajectory and LLM-as-judge verdict."""

    run: BenchRunRecord
    verdict: JudgeVerdict
    judge_llm: str
    scored_at: str | None = None

    def to_legacy_dict(self) -> dict[str, Any]:
        """Flatten into the legacy scored JSONL row format."""
        d = self.run.to_legacy_dict()
        d["judge_llm"] = self.judge_llm
        d["correctness"] = self.verdict.correctness
        d["numeric_match"] = self.verdict.numeric_match
        d["groundedness"] = self.verdict.groundedness
        d["error_category"] = self.verdict.error_category
        d["rationale"] = self.verdict.rationale
        if self.scored_at:
            d["scored_at"] = self.scored_at
        return d


class BenchSummary(BaseModel):
    """Aggregated metrics summary for a benchmark run."""

    profile: str = "default"
    total_questions: int = Field(default=0, alias="n")
    correct: int = 0
    partial: int = 0
    incorrect: int = 0
    accuracy: float = Field(default=0.0, alias="accuracy_correct")
    partial_accuracy: float = Field(default=0.0, alias="accuracy_correct_or_partial")
    numeric_match_rate: float | None = None
    numeric_total: int = Field(default=0, alias="numeric_questions")
    numeric_matched: int = 0
    grounded_rate: float = Field(default=0.0, alias="groundedness_rate")
    error_breakdown: dict[str, int] = Field(default_factory=dict)
    total_tool_calls: int = 0
    avg_tool_calls: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    agent_llm: str = ""
    judge_llm: str = ""
    generated_at: str = ""

    model_config = {"populate_by_name": True, "extra": "allow"}
