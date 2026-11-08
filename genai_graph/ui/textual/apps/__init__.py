"""Textual application implementations for genai-graph."""

from __future__ import annotations

from genai_graph.ui.textual.apps.bench_tui import BenchViewerApp
from genai_graph.ui.textual.apps.cypher_tui import CypherQueryApp
from genai_graph.ui.textual.apps.document_graph_tui import DocumentGraphApp

# Convenience aliases
BenchTuiApp = BenchViewerApp
CypherTuiApp = CypherQueryApp
DocumentGraphTuiApp = DocumentGraphApp

__all__ = [
    "BenchTuiApp",
    "BenchViewerApp",
    "CypherQueryApp",
    "CypherTuiApp",
    "DocumentGraphApp",
    "DocumentGraphTuiApp",
]
