"""Textual application implementations for genai-graph."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from genai_graph.ui.textual.apps.bench_tui import BenchViewerApp
    from genai_graph.ui.textual.apps.cypher_tui import CypherQueryApp
    from genai_graph.ui.textual.apps.document_graph_tui import DocumentGraphApp

    BenchTuiApp = BenchViewerApp
    CypherTuiApp = CypherQueryApp
    DocumentGraphTuiApp = DocumentGraphApp

_EXPORTS: dict[str, tuple[str, str]] = {
    "BenchTuiApp": ("genai_graph.ui.textual.apps.bench_tui", "BenchViewerApp"),
    "BenchViewerApp": ("genai_graph.ui.textual.apps.bench_tui", "BenchViewerApp"),
    "CypherQueryApp": ("genai_graph.ui.textual.apps.cypher_tui", "CypherQueryApp"),
    "CypherTuiApp": ("genai_graph.ui.textual.apps.cypher_tui", "CypherQueryApp"),
    "DocumentGraphApp": ("genai_graph.ui.textual.apps.document_graph_tui", "DocumentGraphApp"),
    "DocumentGraphTuiApp": ("genai_graph.ui.textual.apps.document_graph_tui", "DocumentGraphApp"),
}

__all__ = [
    "BenchTuiApp",
    "BenchViewerApp",
    "CypherQueryApp",
    "CypherTuiApp",
    "DocumentGraphApp",
    "DocumentGraphTuiApp",
]


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_path, attr_name = _EXPORTS[name]
        mod = importlib.import_module(module_path)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
