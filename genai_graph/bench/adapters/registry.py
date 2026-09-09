"""Benchmark adapter registry and factory resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Type

from loguru import logger

from genai_graph.bench.adapters.base import BaseBenchmarkAdapter
from genai_graph.bench.adapters.financebench import FinanceBenchAdapter
from genai_graph.bench.adapters.mmlongbench import MMLongBenchDocAdapter
from genai_graph.bench.adapters.officeqa import OfficeQAAdapter

_REGISTRY: dict[str, Type[BaseBenchmarkAdapter]] = {
    "financebench": FinanceBenchAdapter,
    "finance_bench": FinanceBenchAdapter,
    "officeqa": OfficeQAAdapter,
    "office_qa": OfficeQAAdapter,
    "mmlongbench": MMLongBenchDocAdapter,
    "mmlongbench_doc": MMLongBenchDocAdapter,
}


def register_benchmark_adapter(name: str, adapter_cls: Type[BaseBenchmarkAdapter]) -> None:
    """Register a custom benchmark adapter."""
    _REGISTRY[name.lower()] = adapter_cls


def get_benchmark_adapter(
    name_or_alias: str | None = None,
    project_root: Path | None = None,
) -> BaseBenchmarkAdapter:
    """Resolve and return an instantiated benchmark adapter.

    If name_or_alias is None, attempts to auto-detect from project folder name
    or configuration files.
    """
    if name_or_alias:
        key = name_or_alias.lower().replace("-", "_")
        if key in _REGISTRY:
            return _REGISTRY[key]()
        for reg_key, cls in _REGISTRY.items():
            if reg_key in key or key in reg_key:
                return cls()

    # Auto-detection from folder name or cwd
    root = project_root or Path.cwd()
    folder_name = root.name.lower()
    if "finance" in folder_name:
        return FinanceBenchAdapter()
    if "office" in folder_name:
        return OfficeQAAdapter()
    if "mmlong" in folder_name:
        return MMLongBenchDocAdapter()

    # Default to FinanceBench adapter
    logger.debug("Could not auto-detect benchmark adapter for '{}', defaulting to FinanceBenchAdapter", folder_name)
    return FinanceBenchAdapter()
