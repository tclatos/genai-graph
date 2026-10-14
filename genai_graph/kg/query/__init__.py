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
    "CypherQueryApp": "genai_graph.kg.query.cypher_tui",
    "run_cypher_tui": "genai_graph.kg.query.cypher_tui",
    "DEFAULT_CYPHER_EXAMPLES": "genai_graph.kg.query.cypher_tui",
}

__all__ = [
    "CypherQueryApp",
    "DEFAULT_CYPHER_EXAMPLES",
    "SYSTEM_PROMPT",
    "build_kg_agent_system_prompt",
    "create_kg_cypher_tool",
    "query_kg",
    "run_cypher_tui",
    "text2cypher_chain",
]


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        mod = importlib.import_module(_EXPORTS[name])
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
