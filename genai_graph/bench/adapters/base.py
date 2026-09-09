"""Abstract Base Class and helper utilities for benchmark dataset adapters."""

from __future__ import annotations

import importlib
import os
import shutil
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Type

import pathspec
from loguru import logger

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


def download_hf_file(
    repo_id: str,
    filename: str,
    output_dir: Path | None = None,
    repo_type: str = "dataset",
    token: str | None = None,
) -> Path:
    """Download a single file from Hugging Face Hub.

    Args:
        repo_id: Hugging Face repo ID (e.g. 'databricks/officeqa').
        filename: Relative file path in repo (e.g. 'officeqa_pro.csv' or 'documents/doc1.pdf').
        output_dir: Optional directory to place the downloaded file.
        repo_type: 'dataset' or 'model'.
        token: Optional Hugging Face token (defaults to HF_TOKEN env var).

    Returns:
        Path to the downloaded local file.
    """
    from huggingface_hub import hf_hub_download

    auth_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    local_dir_arg = str(output_dir) if output_dir else None
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        token=auth_token,
        local_dir=local_dir_arg,
    )
    return Path(downloaded)


def download_http_file(
    url: str,
    output_path: Path,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> Path:
    """Download a file over HTTP/HTTPS.

    Args:
        url: Remote URL to download.
        output_path: Target local file path.
        headers: Optional HTTP headers dictionary.
        timeout: Socket timeout in seconds.

    Returns:
        Path to the written local file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    req_headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    if headers:
        req_headers.update(headers)

    req = urllib.request.Request(url, headers=req_headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, output_path.open("wb") as fh:
            shutil.copyfileobj(resp, fh)
    except Exception as exc:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url} -> {output_path}: {exc}") from exc

    logger.info("Saved HTTP file ({} bytes) -> {}", output_path.stat().st_size, output_path)
    return output_path


def load_hf_dataset_to_pandas(
    dataset_id: str,
    split: str = "train",
    cache_file: Path | None = None,
    token: str | None = None,
) -> Any:
    """Load a Hugging Face dataset and return it as a pandas DataFrame.

    If cache_file is provided and exists as parquet, loads directly from disk.
    """
    import pandas as pd

    if cache_file and cache_file.exists():
        logger.info("Loading cached dataset from {}", cache_file)
        return pd.read_parquet(cache_file)

    from datasets import load_dataset

    auth_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    logger.info("Downloading dataset {} (split='{}') from Hugging Face", dataset_id, split)
    ds = load_dataset(dataset_id, split=split, token=auth_token)
    df = ds.to_pandas()
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file, index=False)
        logger.info("Cached {} rows to {}", len(df), cache_file)
    return df


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


def resolve_benchmark_adapter(
    adapter_spec: str | Type[BaseBenchmarkAdapter] | BaseBenchmarkAdapter | None = None,
    project_root: Path | None = None,
) -> BaseBenchmarkAdapter:
    """Resolve and return an instantiated benchmark adapter.

    Args:
        adapter_spec: Qualified class name (e.g. 'financebench.adapter.FinanceBenchAdapter'),
                      alias ('mmlongbench'), class, or already instantiated adapter.
        project_root: Optional project root path for context.

    Returns:
        Instantiated BaseBenchmarkAdapter.
    """
    if isinstance(adapter_spec, BaseBenchmarkAdapter):
        return adapter_spec
    if isinstance(adapter_spec, type) and issubclass(adapter_spec, BaseBenchmarkAdapter):
        return adapter_spec()

    if adapter_spec:
        spec_str = str(adapter_spec).strip()
        # Handle built-in mmlongbench in genai-graph
        if spec_str.lower() in ("mmlongbench", "mmlongbench_doc", "mmlongbenchdocadapter"):
            from genai_graph.bench.adapters.mmlongbench import MMLongBenchDocAdapter

            return MMLongBenchDocAdapter()

        # Handle dotted class path, e.g. "financebench.adapter.FinanceBenchAdapter"
        if "." in spec_str:
            module_name, class_name = spec_str.rsplit(".", 1)
            try:
                mod = importlib.import_module(module_name)
                cls = getattr(mod, class_name)
                if isinstance(cls, type) and issubclass(cls, BaseBenchmarkAdapter):
                    return cls()
                return cls()
            except Exception as exc:
                raise ImportError(
                    f"Failed to load benchmark adapter '{spec_str}' (module: '{module_name}', class: '{class_name}'): {exc}"
                ) from exc

    root = project_root or Path.cwd()
    raise ValueError(
        f"No benchmark adapter specified or resolved for project in {root}. "
        f"Please define 'adapter: <dotted.path.to.AdapterClass>' (e.g. 'financebench.adapter.FinanceBenchAdapter') "
        f"in your bench.yaml configuration."
    )


# Backward-compatible alias
get_benchmark_adapter = resolve_benchmark_adapter
