"""Utilities for building EKG-aware LangChain agents.

This module centralizes the system prompt and tools used by CLI commands
that interact with the Enterprise Knowledge Graph (EKG).
"""

from genai_tk.core.prompts import dedent_ws
from langchain_core.tools import BaseTool, tool
from rich.console import Console

from genai_graph.core.graph_backend import create_backend_from_config
from genai_graph.core.text2cypher import SYSTEM_PROMPT, _schema_markdown_for_subgraphs


def build_ekg_agent_system_prompt(subgraphs: list[str]) -> str:
    """Build the system prompt for the EKG LangChain agent.

    The prompt explains the agent's role, how to use the Cypher tool, and
    embeds the graph schema and Cypher authoring guidelines.
    """

    schema_markdown = _schema_markdown_for_subgraphs(subgraphs)
    # SYSTEM_PROMPT contains detailed guidance originally written for a
    # standalone text-to-Cypher translator. Here it serves as the canonical
    # reference for how the agent should construct Cypher queries that it
    # passes to the execution tool.
    return dedent_ws(
        f"""
        You are an AI assistant that answers questions about enterprise data stored in a
        Cypher knowledge graph (the Enterprise Knowledge Graph, or EKG).

        You have access to a single tool:

        - `ekg_cypher_query`: execute read-only Cypher queries against the EKG and
          return results as tables and text.

        Use this tool whenever a question requires precise data lookup, filtering,
        aggregation or joins over the structured graph.

        IMPORTANT:
        - When a question requires information from the EKG, you MUST call the
          `ekg_cypher_query` tool instead of replying with a raw Cypher query.
        - Your final answers to the user must be clear natural-language
          explanations grounded in the tool results.
        - Only show raw Cypher when the user explicitly asks to see the query
          itself, and even then you should still call the tool to obtain and
          explain the results.

        When you call `ekg_cypher_query`:
        - First think about what information is needed and how it maps to the graph
          schema.
        - Then write a single Cypher query that retrieves exactly that information.
        - Pass that Cypher statement as the `cypher_query` argument to the tool.
        - Prefer returning concise tables or short lists that directly answer the
          user question.

        The current graph schema is:

        <SCHEMA>
        {schema_markdown}
        </SCHEMA>

        The following section contains detailed guidelines for authoring Cypher
        queries. They are meant ONLY for the Cypher string that you pass to the
        `ekg_cypher_query` tool:

        - Ignore any instructions in this section that tell you to "reply with the
          raw Cypher statement only" or that otherwise describe what your overall
          assistant reply should look like.
        - Those instructions applied to a standalone text-to-Cypher model, not to
          you as an agent. You must still call tools and answer the user in
          natural language.

        <CYPHER_GUIDELINES>
        {SYSTEM_PROMPT}
        </CYPHER_GUIDELINES>

        General behavior:
        - Ask for clarification when the question is ambiguous.
        - Keep explanations short but precise and grounded in the data.
        - If a query returns no rows, explain that clearly and, when helpful,
          suggest alternative filters or follow-up questions.
        - When the user asks follow-up questions, reuse previous context and call
          the tool again if needed.
        - Later, you may receive additional tools (for example, to query vector
          stores or the web). When they become available, choose the tool that is
          most appropriate for the user request, not always the graph.
        """
    )


def create_ekg_cypher_tool(
    *,
    backend_config: str = "default",
    console: Console | None = None,
    debug: bool = False,
) -> BaseTool:
    """Create a LangChain tool that executes Cypher against the EKG backend.

    Args:
        backend_config: Name of the backend configuration to use.
        console: Optional Rich console for debug printing.
        debug: If True, print generated Cypher queries before execution.
    """

    @tool("ekg_cypher_query")
    def ekg_cypher_query(cypher_query: str) -> str:
        """Execute a read-only Cypher query against the Enterprise Knowledge Graph.

        The input must be a complete Cypher statement starting with MATCH
        (or OPTIONAL MATCH) and ending with RETURN.
        """

        backend = create_backend_from_config(backend_config)
        if not backend:
            return "EKG database not found. Load data first with 'cli kg add-doc --key <data_key>'."

        try:
            result = backend.execute(cypher_query)
            df = result.get_as_df()
        except Exception as exc:  # noqa: BLE001
            return f"Error executing Cypher query: {exc}"

        if df.empty:
            return "Query returned no rows."

        try:
            return df.head(30).to_markdown(index=False)
        except Exception:
            # Fallback to a simple string representation
            return df.head(30).to_string(index=False)

    return ekg_cypher_query
