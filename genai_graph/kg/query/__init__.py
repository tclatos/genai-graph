"""Query utilities for Knowledge Graphs.

This package provides:
- Text-to-Cypher translation
- LangChain agent tools for KG querying
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "build_kg_agent_system_prompt": "genai_graph.kg.query.agent",
    "create_kg_cypher_tool": "genai_graph.kg.query.agent",
    "SYSTEM_PROMPT": "genai_graph.kg.query.text2cypher",
    "query_kg": "genai_graph.kg.query.text2cypher",
    "text2cypher_chain": "genai_graph.kg.query.text2cypher",
}

__all__ = [
    "SYSTEM_PROMPT",
    "text2cypher_chain",
    "query_kg",
    "build_kg_agent_system_prompt",
    "create_kg_cypher_tool",
]


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        mod = importlib.import_module(_EXPORTS[name])
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
