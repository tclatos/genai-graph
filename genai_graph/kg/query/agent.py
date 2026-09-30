"""Utilities for building KG-aware LangChain agents.

This module centralizes the system prompt and tools used by CLI commands
and web applications that interact with the Knowledge Graph (KG).
"""

from genai_tk.core.prompts import dedent_ws
from langchain_core.tools import BaseTool, tool
from rich.console import Console

from genai_graph.kg.backend import create_backend_from_config
from genai_graph.kg.manager import get_kg_manager
from genai_graph.kg.query.text2cypher import SYSTEM_PROMPT, _embed_query_vector, _ensure_vector_indexes


def build_kg_agent_system_prompt(single_tool_mode: bool = False, kg_config_name: str | None = None) -> str:
    """Build the system prompt for the KG LangChain agent.

    The prompt explains the agent's role, how to use the Cypher tool, and
    embeds the graph schema and Cypher authoring guidelines.

    Args:
        single_tool_mode: If True, adjusts prompt for single tool call behavior.
        kg_config_name: KG config profile to load schema from. Defaults to the
            active manager profile.
    """
    manager = get_kg_manager()
    profile = kg_config_name if kg_config_name is not None else manager.profile

    # Prefer loading from the canonical JSON (has structured vector_indexes).
    # Fall back to the markdown text file if JSON is absent.
    json_path = manager.get_schema_json_path_for(profile)
    txt_path = manager.get_schema_path_for(profile)

    if json_path.exists():
        from genai_graph.kg.schema.resolved import ResolvedSchema

        resolved = ResolvedSchema.from_json_file(str(json_path))
        schema_markdown = resolved.to_markdown()
        vector_indexes_section = resolved.to_vector_section_markdown()
    elif txt_path.exists():
        schema_markdown = txt_path.read_text(encoding="utf-8")
        _vi_marker = "### Vector-Indexed Fields"
        _vi_idx = schema_markdown.find(_vi_marker)
        vector_indexes_section = schema_markdown[_vi_idx:] if _vi_idx >= 0 else ""
    else:
        raise FileNotFoundError(
            f"No schema file found for profile '{profile}'. "
            f"Run 'cli kg create' or 'cli kg schema --regen --kg {profile}'."
        )

    # SYSTEM_PROMPT contains detailed guidance originally written for a
    # standalone text-to-Cypher translator. Here it serves as the canonical
    # reference for how the agent should construct Cypher queries that it
    # passes to the execution tool.

    if single_tool_mode:
        # In single-tool mode, be extremely directive
        instructions = dedent_ws(
            """
            CRITICAL INSTRUCTION:
            You MUST call the `kg_cypher_query` tool to execute the query.
            DO NOT respond with just the Cypher query text.
            Your ONLY job is to:
            1. Generate the appropriate Cypher query
            2. Call the kg_cypher_query tool with that query
            3. Let the tool return the results

            You will be stopped after the first tool call, so make it count.
            """
        )
    else:
        instructions = dedent_ws(
            """
            IMPORTANT:
            - When a question requires information from the KG, you MUST call the
              `kg_cypher_query` tool instead of replying with a raw Cypher query.
            - Your final answers to the user must be clear natural-language
              explanations grounded in the tool results.
            - Only show raw Cypher when the user explicitly asks to see the query
              itself, and even then you should still call the tool to obtain and
              explain the results.
            """
        )

    return dedent_ws(
        f"""
        You are an AI assistant that answers questions about data stored in a
        Cypher knowledge graph (KG).

        You have access to a single tool:

        - `kg_cypher_query`: execute read-only Cypher queries against the KG and
          return results as tables and text.

        Use this tool whenever a question requires precise data lookup, filtering,
        aggregation or joins over the structured graph.

        {instructions}

        When you call `kg_cypher_query`:
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

        <VECTOR_INDEXES>
        {vector_indexes_section}
        </VECTOR_INDEXES>

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


def create_kg_cypher_tool(
    *,
    backend_config: str = "default",
    kg_config_name: str | None = None,
    console: Console | None = None,
    debug: bool = False,
) -> BaseTool:
    """Create a LangChain tool that executes Cypher against the KG backend.

    Args:
        backend_config: Name of the backend configuration to use.
        kg_config_name: Name of the KG configuration to use (e.g., "simple").
        console: Optional Rich console for debug printing.
        debug: If True, print generated Cypher queries before execution.
    """

    @tool("kg_cypher_query")
    def kg_cypher_query(cypher_query: str, question: str = "") -> str:
        """Execute a read-only Cypher query against the Knowledge Graph.

        The input must be a complete Cypher statement starting with MATCH
        (or OPTIONAL MATCH) or CALL QUERY_VECTOR_INDEX, and ending with RETURN.

        If the query contains $query_vector, provide the original user question
        in the `question` parameter so the system can compute the embedding.
        """

        backend = create_backend_from_config(backend_config, kg_config_name)
        if not backend:
            return "KG database not found. Load data first with 'cli kg add-doc --key <data_key>'."

        if hasattr(backend, "ensure_vector_extension"):
            backend.ensure_vector_extension()
        _ensure_vector_indexes(kg_config_name, backend)

        # Detect $query_vector and compute embedding if needed
        params = _embed_query_vector(cypher_query, question) if question else None

        try:
            result = backend.execute(cypher_query, parameters=params) if params else backend.execute(cypher_query)
            df = result.get_as_df()
        except Exception as exc:
            return f"Error executing Cypher query: {exc}"

        if df.empty:
            return "Query returned no rows."

        try:
            return df.head(30).to_markdown(index=False)
        except Exception:
            # Fallback to a simple string representation
            return df.head(30).to_string(index=False)

    return kg_cypher_query
