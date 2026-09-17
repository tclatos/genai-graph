"""Deep agent that navigates the Ladybug Document Graph to answer queries.

Exposes a thin builder over genai-tk's ``type: deep`` agent + the document-graph
navigation tools, plus the co-located runtime skills under ``agent/skills/``.
The CLI (``cli docgraph agent``) and downstream projects (e.g. rfq_pricing's
``cli agent extract``) both go through :func:`create_docgraph_agent`.
"""

from genai_graph.agent.docgraph_agent import (
    DEFAULT_LLM,
    DEFAULT_PROFILE,
    build_docgraph_system_prompt,
    create_docgraph_agent,
    create_document_graph_tools_from_config,
    find_docgraph_db_path,
    prepare_docgraph_profile,
    resolve_db_path,
    run_docgraph_agent,
)

__all__ = [
    "DEFAULT_LLM",
    "DEFAULT_PROFILE",
    "build_docgraph_system_prompt",
    "create_docgraph_agent",
    "create_document_graph_tools_from_config",
    "find_docgraph_db_path",
    "prepare_docgraph_profile",
    "resolve_db_path",
    "run_docgraph_agent",
]
