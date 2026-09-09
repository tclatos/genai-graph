"""Abstract Base Class for benchmark dataset adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pathspec

from genai_graph.bench.models import BenchQuestion


def match_docs_by_pathspecs(all_docs: list[str], pathspecs: list[str]) -> list[str]:
    """Filter doc_names using gitwildmatch/gitignore style pathspecs.

    Args:
        all_docs: List of candidate document names.
        pathspecs: List of pathspec patterns (supports ``!`` for exclusion).

    Returns:
        Filtered and stable list of matching document names.
    """
    if not pathspecs:
        return all_docs
    spec = pathspec.PathSpec.from_lines("gitwildmatch", pathspecs)
    matched = [d for d in all_docs if spec.match_file(d)]
    return matched


class BaseBenchmarkAdapter(ABC):
    """Base adapter defining contracts for dataset loading, fetching, and judge rubrics."""

    name: str = "base"

    @abstractmethod
    def load_dataset(self, split: str | None = None, cache_dir: Path | None = None) -> list[BenchQuestion]:
        """Load and return all questions from the dataset converted to standard BenchQuestion models."""

    @abstractmethod
    def fetch_document(self, doc_name: str, output_dir: Path) -> Path:
        """Download or fetch a single raw document (PDF or text) to output_dir and return its path."""

    @abstractmethod
    def get_judge_rubric(self) -> str:
        """Return the domain-specific grading system prompt rubric."""

    def resolve_doc_name(self, raw_name: str) -> str:
        """Normalize a raw document reference into a clean stem."""
        s = raw_name.strip().replace("\r", "")
        if s.endswith((".pdf", ".txt", ".md", ".html", ".csv", ".xlsx")):
            return Path(s).stem
        return s

    def get_available_docs(self, split: str | None = None, cache_dir: Path | None = None) -> list[str]:
        """Return a sorted unique list of all document names referenced in the dataset."""
        questions = self.load_dataset(split=split, cache_dir=cache_dir)
        docs: set[str] = set()
        for q in questions:
            for d in q.doc_names:
                if d:
                    docs.add(d)
        return sorted(docs)
